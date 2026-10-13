# -*- coding: utf-8 -*-
"""E-NeRV 训练与评估的入口脚本。

整体运行流程（自顶向下）：
    main()                                    本文件的主流程
      ├─ utils.misc.init_distributed_mode()   初始化分布式训练环境（单机时直接返回）
      ├─ utils.misc.load_yaml_as_dict()       读取 yaml 实验配置（超参数唯一来源）
      ├─ model_dict[cfg.model.model_name]     按配置名实例化模型（NeRV_Generator / E_NeRV_Generator）
      ├─ dataset_dict[cfg.dataset_type]       构建训练/验证数据集（CustomDataSet）与 DataLoader
      ├─ optim.Adam                           构建优化器
      └─ 逐 epoch 训练循环：
            ├─ engine.train_one_epoch()       单轮训练（内部调用模型 forward、损失计算、调 lr）
            ├─ utils.misc.save_on_master()    保存 checkpoint（仅主进程保存）
            └─ engine.evaluate()              按 eval_freq 周期性在验证集上评估 PSNR / MS-SSIM
"""
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import os
from model import model_dict           # 模型注册表 {"NeRV": NeRV_Generator, "E_NeRV": E_NeRV_Generator}
from datasets import dataset_dict      # 数据集注册表 {"all": CustomDataSet}
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, dataloader
from torch.utils.data.distributed import DistributedSampler
from engine import train_one_epoch, evaluate   # 训练 / 评估引擎
from torch.utils.tensorboard import SummaryWriter
import utils.misc as utils


def get_args_parse():
    """解析命令行参数（仅包含启动脚本层面的参数，模型超参数全部放在 yaml 配置中）"""
    parser = argparse.ArgumentParser('Dense NeRV', add_help=False)

    parser.add_argument('--cfg_path', default='', type=str, help='path to specific cfg yaml file path')  # 实验配置 yaml 文件路径
    parser.add_argument('--output_dir', default='', type=str, help='path to save the log and other files')  # 输出目录（日志 / checkpoint / tensorboard）
    parser.add_argument('--time_str', default='', type=str, help='just for tensorboard dir name')  # 时间戳字符串，仅用于 tensorboard 目录命名
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')  # 训练 / 测试使用的设备
    parser.add_argument('--port', default=29500, type=int, help='port number')  # 分布式训练的通信端口

    return parser


def main(args):
    """主流程：分布式初始化 -> 加载配置 -> 构建模型 / 数据 / 优化器 -> 训练与评估循环"""
    utils.init_distributed_mode(args)   # 依据环境变量（RANK/SLURM）决定是否启用分布式；单机单卡时 args.rank=None
    print('git:\n {}\n'.format(utils.get_sha()))  # 打印当前代码 git 版本信息，便于复现实验

    # get cfg yaml file
    cfg = utils.load_yaml_as_dict(args.cfg_path)  # 读取 yaml 配置为字典
    # dump the cfg yaml file in output dir
    utils.dump_cfg_yaml(cfg, args.output_dir)    # 把配置原样写入输出目录，便于追溯该次实验的全部超参数
    print(cfg)

    device = torch.device(args.device)

    # fix the seed
    seed = cfg['seed']  # 固定随机种子，保证可复现（与原版 NeRV 一致取 seed=1）
    # seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model_dict[cfg['model']['model_name']](cfg=cfg['model'])  # 从注册表按 model_name 实例化生成器（工厂模式）
    model.to(device)   # 模型搬至 GPU

    model_without_ddp = model  # 保存 DDP 包装前的模型引用（保存 checkpoint 时需要裸模型的 state_dict）
    # get model params
    if args.rank in [0, None]:  # 仅主进程（或单机模式）执行打印与日志记录
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6  # 统计参数量（百万）
        print(f'{args}\n {model}\n Model Params: {params}M')
        # tensorboard writer
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_{}'.format(args.time_str)))  # tensorboard 写入器

    img_transform = transforms.ToTensor()  # PIL 图像 -> [0,1] 归一化的 float 张量
    dataset_train = dataset_dict[cfg['dataset_type']](main_dir=cfg['dataset_path'], transform=img_transform, train=True)   # 训练集（视频全部帧）
    dataset_val = dataset_dict[cfg['dataset_type']](main_dir=cfg['dataset_path'], transform=img_transform, train=False)    # 验证集（同一视频，衡量重建质量）
    # follow nerv implementation on sampler and dataloader
    sampler_train = DistributedSampler(dataset_train) if args.distributed else None  # 分布式采样器（单机为 None，用 shuffle）
    sampler_val = DistributedSampler(dataset_val) if args.distributed else None

    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg['train_batchsize'], shuffle=(sampler_train is None), num_workers=cfg['workers'], 
        pin_memory=True, sampler=sampler_train, drop_last=True, worker_init_fn=utils.worker_init_fn  # worker_init_fn 固定每个 worker 的种子保证可复现
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=cfg['val_batchsize'], shuffle=False, num_workers=cfg['workers'], 
        pin_memory=True, sampler=sampler_val, drop_last=False, worker_init_fn=utils.worker_init_fn
    )

    datasize = len(dataset_train)  # 单个 epoch 的迭代步数 = 视频帧数 / batchsize（用于按迭代调度学习率）

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad],
            "lr": cfg['optim']['lr'],  # 基准学习率（实际每步由 utils.adjust_lr 覆盖）
        }
    ]

    optim_cfg = cfg['optim']
    if optim_cfg['optim_type'] == 'Adam':  # 目前仅实现了 Adam 优化器
        optimizer = optim.Adam(param_dicts, lr=optim_cfg['lr'], betas=(optim_cfg['beta1'], optim_cfg['beta2']))
    else:
        optimizer = None
    assert optimizer is not None, "No implementation of Optimizer!"

    if args.distributed:  # 分布式模式下用 DDP 包装模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module  # DDP 包装后通过 .module 取回裸模型

    output_dir = Path(args.output_dir)

    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]  # 记录历史最优指标

    print('Start training')
    start_time = datetime.now()
    for epoch in range(cfg['epoch']):  # 主训练循环：每个视频训练 cfg['epoch']（默认 300）个 epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)  # 保证分布式采样器每个 epoch 打乱顺序
        train_stats = train_one_epoch(  # 跳转到 engine.py 的 train_one_epoch，完成一轮前向/反向/指标统计
            model, dataloader_train, optimizer, device, epoch, cfg, args, datasize, start_time, writer
        )

        train_best_psnr = train_stats['train_psnr'][-1] if train_stats['train_psnr'][-1] > train_best_psnr else train_best_psnr  # 更新训练 PSNR 最优值
        train_best_msssim = train_stats['train_msssim'][-1] if train_stats['train_msssim'][-1] > train_best_msssim else train_best_msssim  # 更新训练 MS-SSIM 最优值
        if args.rank in [0, None]:  # 仅主进程打印
            print_str = '\ttraining: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(train_stats['train_psnr'][-1].item(), 
            train_best_psnr.item(), train_best_msssim.item())
            print(print_str, flush=True)
        
        checkpoint_paths = [output_dir / 'checkpoint.pth']  # save one per epoch（每个 epoch 覆盖保存同一个文件）
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({   # 仅主进程保存：模型权重 + 优化器状态 + 训练进度与最优指标
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': cfg,
                'train_best_psnr': train_best_psnr,
                'train_best_msssim': train_best_msssim,
                'val_best_psnr': val_best_psnr,
                'val_best_msssim': val_best_msssim,
            }, checkpoint_path)
        
        # evaluation
        if (epoch + 1) % cfg['eval_freq'] == 0 or epoch > cfg['epoch'] - 10:  # 每 eval_freq 个 epoch 或最后 10 个 epoch 评估一次
            val_stats = evaluate(  # 跳转到 engine.py 的 evaluate，在验证集上计算 PSNR / MS-SSIM
                model, dataloader_val, device, cfg, args, save_image=False  # TODO: implement the save image
            )
            
            val_best_psnr = val_stats['val_psnr'][-1] if val_stats['val_psnr'][-1] > val_best_psnr else val_best_psnr  # 更新验证 PSNR 最优值
            val_best_msssim = val_stats['val_msssim'][-1] if val_stats['val_msssim'][-1] > val_best_msssim else val_best_msssim  # 更新验证 MS-SSIM 最优值
            if args.rank in [0, None]:
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                print_str += '\tevaluation: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}'.format(
                    val_stats['val_psnr'][-1].item(), val_best_psnr.item(), val_best_msssim.item())
                print(print_str)
    print("Training complete in: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('E-NeRV training and evaluation script', parents=[get_args_parse()])  # 继承 get_args_parse 中的参数定义
    args = parser.parse_args()

    assert args.cfg_path is not None, 'Need a specific cfg path!'  # 必须提供配置文件路径
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录
    main(args)  # 程序正式入口：跳转到 main()
