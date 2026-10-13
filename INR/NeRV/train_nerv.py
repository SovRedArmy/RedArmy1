# ==========================================================================================
# NeRV: Neural Representations for Videos (NeurIPS 2021) —— 训练/评估入口脚本
# 官方代码: https://github.com/haochen-rye/NeRV    论文: https://arxiv.org/abs/2110.13903
#
# 运行方式(详见 README.md):
#   训练: python train_nerv.py -e 300 --lower-width 96 --num-blocks 1 --dataset bunny ...
#   评估: python train_nerv.py ... --weight checkpoints/nerv_S.pth --eval_only
#
# 代码执行流程(函数调用关系):
#   main() ──> train() ──> (训练循环: CustomDataSet/DataLoader 取帧
#                          → PositionalEncoding 编码帧索引
#                          → Generator 前向 → loss_fn 计算损失 → 反向传播
#                          → adjust_lr 调整学习率 → 定期 evaluate())
#           └─(eval_only 时)──> evaluate()  (可选: quantize_per_tensor 量化 + Huffman 熵编码统计)
# ==========================================================================================
from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_nerv import CustomDataSet, Generator
from utils import *


def main():
    """程序入口: 解析命令行参数 → 拼接实验目录名 exp_id → 启动训练 train()(或分布式 mp.spawn)."""
    parser = argparse.ArgumentParser()

    # dataset parameters
    # ============ 数据集参数 ============
    parser.add_argument('--vid',  default=[None], type=int,  nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame_gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset',)
    parser.add_argument('--test_gap', default=1, type=int, help='evaluation gap')

    # NERV architecture parameters
    # ============ NeRV 网络结构参数(论文 Sec. 3) ============
    # embedding parameters
    # 帧索引位置编码: '底数_层数'(b_L), 默认 1.25_80 → sin/cos 成对共 160 维(论文用 b=1.2541, L=80)
    parser.add_argument('--embed', type=str, default='1.25_80', help='base value/embed length for position encoding')

    # FC + Conv parameters
    # stem_dim_num: MLP 隐层'宽度_层数'; fc_hw_dim: MLP 输出特征图'高_宽_通道';
    # expansion: 第一个卷积 stage 的通道扩张倍数; reduction: 之后每级通道缩减倍数
    parser.add_argument('--stem_dim_num', type=str, default='1024_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_128', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=8, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    # 每个 stage 的上采样倍率列表, 与 fc_hw_dim 的乘积应等于目标分辨率(1080p: 5,3,2,2,2; 720p: 5,2,2,2,2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list')
    parser.add_argument('--num-blocks', type=int, default=1)

    # 论文消融: norm='none'(不加归一化)与 act='gelu'/'swish' 效果最好
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument("--single_res", action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv_type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'deconv', 'bilinear'])

    # General training setups
    # ============ 通用训练参数 ============
    # warmup: 预热 epoch 比例(命令行传比例, 代码中乘 epochs 换算); lr_type: 学习率策略(cosine/step/const)
    # loss_type: 损失类型(论文最终用 Fusion6 = 0.7·L1 + 0.3·(1-SSIM)); lw: 多分辨率中间监督损失的权重
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    # ============ 评估参数 ============
    # eval_only: 只评估不训练; quant_bit/quant_axis: 权重量化位宽与逐通道量化轴(视频压缩用)
    # dump_images: 保存预测帧与 GT 帧; eval_fps: 每帧前向 10 次以测解码速度(FPS)
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # pruning paramaters
    # ============ 剪枝参数(视频压缩流水线第 2 步) ============
    # prune_ratio: 目标保留参数比例; prune_steps: 各次剪枝所在的 epoch 比例(渐进式剪枝+微调)
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters
    # ============ 分布式训练参数 ============
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    # ============ 日志与输出目录参数 ============
    # 所有 checkpoint / tensorboard / 日志文件都写到 outf/exp_id 目录下
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    args = parser.parse_args()
        
    args.warmup = int(args.warmup * args.epochs)   # warmup 从"epoch 比例"换算成"epoch 数"

    print(args)
    torch.set_printoptions(precision=4) 

    if args.debug:
        args.eval_freq = 1                          # debug 模式: 每个 epoch 都评估, 且输出到 debug 目录
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    # ---- 用所有关键超参拼接出实验名 exp_id, 保证不同配置的结果互不覆盖地存到不同目录 ----
    if args.prune_ratio < 1 and not args.eval_only: 
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    extra_str = '_Strd{}_{}Res{}{}'.format( ','.join([str(x) for x in args.strides]),  'Sin' if args.single_res else f'_lw{args.lw}_multi',  
            '_dist' if args.distributed else '', f'_eval' if args.eval_only else '')
    norm_str = '' if args.norm == 'none' else args.norm

    exp_id = f'{args.dataset}/embed{args.embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}__exp{args.expansion}_reduce{args.reduction}_low{args.lower_width}_blk{args.num_blocks}_cycle{args.cycles}' + \
            f'_gap{args.frame_gap}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}' + \
            f'_{args.loss_type}{norm_str}{extra_str}{prune_str}'
    
    exp_id += f'_act{args.act}_{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
    	print('Will overwrite the existing output dir!')
    	shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    # 用实验名的哈希生成分布式通信端口, 避免同时跑多个实验时端口冲突
    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        # 多卡分布式: 为每张 GPU 启动一个 train 进程(local_rank = 0, 1, ...)
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        # 单进程(单卡或 DataParallel): local_rank 传 None
        train(None, args)

def train(local_rank, args):
    """训练主流程: 构建模型/数据/优化器 → 逐 epoch 训练 → 定期评估并保存 checkpoint.

    local_rank: 当前进程使用的 GPU 编号(分布式时为 0,1,...; 单进程时为 None).
    主要步骤:
        1. 固定随机种子, 保证可复现
        2. 构建 PositionalEncoding(帧索引编码) 与 Generator(NeRV 网络)
        3. (可选)收集待剪枝的层, 设置渐进式剪枝
        4. 统计参数量; 包装 DDP/DataParallel; 创建 Adam 优化器
        5. 加载 --weight 指定的权重 / 自动断点续训 model_latest.pth
        6. 构建 train/val DataLoader
        7. 若 eval_only 则直接 evaluate() 后返回
        8. epoch 循环: 位置编码 → 前向 → loss_fn → 反向传播 → adjust_lr, 并记录 PSNR/MS-SSIM
    """
    cudnn.benchmark = True
    # 固定 torch / numpy / random 三个随机源, 保证实验可复现
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    # 记录训练/验证集上的历史最优指标
    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False

    PE = PositionalEncoding(args.embed)    # 帧索引位置编码器(论文 Sec. 3.1)
    args.embed_length = PE.embed_length    # 编码维度(如 160), 作为 Generator 的输入维度
    # 构建 NeRV 网络(Generator): 位置编码向量 → MLP → 逐级上采样卷积 → RGB 帧
    model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)

    ##### prune model params and flops #####
    # ---- 剪枝相关(论文视频压缩流水线第 2 步): prune_ratio < 1 时启用全局非结构化剪枝 ----
    prune_net = args.prune_ratio < 1
    # import pdb; pdb.set_trace; from IPython import embed; embed()
    if prune_net:
        # 收集所有可剪枝的权重: stem(MLP)中的 Linear 与每个 NeRVBlock 里的卷积
        param_list = []
        for k,v in model.named_parameters():
            if 'weight' in k:
                if 'stem' in k:                          # stem(MLP)中的 Linear 层
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                elif 'layers' in k[:6] and 'conv' in k:   # NeRVBlock 里的卷积层
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv)
        param_to_prune = [(ele, 'weight') for ele in param_list]
        # 渐进式剪枝: 共 len(prune_steps) 次, 每次剪掉的比例按几何级数分配,
        # 使最终保留的参数比例恰好为 prune_ratio
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]  # epoch 比例 → epoch 编号
        prune_num = 0                                     # 已完成的剪枝次数
        if args.eval_only:
            # 只评估模式: 先剪一次以生成 weight_mask, 便于正确加载剪枝后的 checkpoint
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,      # 全局按 |w| 大小剪枝
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    # 统计模型参数量(单位: 百万), 并创建 tensorboard writer(只有主进程 local_rank∈{0,None} 写日志)
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    if local_rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

        print(f'{args}\n {model}\n Model Params: {params}M')
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {params}M\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    # ---- 把模型搬到 GPU: 分布式(DDP) / 单机多卡(DataParallel) / 单卡 三选一 ----
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        # DDP: 每张卡一个进程, NCCL 后端同步梯度(注意 batchSize 会被均分到各卡)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        # 单机多卡但未开 distributed: 用 DataParallel(单进程多线程, 通信效率低于 DDP)
        model = torch.nn.DataParallel(model).cuda() #model.cuda() #
    else:
        model = model.cuda()

    # 论文使用 Adam 优化器(beta 默认 0.5)
    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    # resume from args.weight
    # ---- 加载 --weight 指定的预训练权重(用于评估/微调/剪枝实验), 并兼容 'module.' 前缀差异 ----
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt)
        else:
            model.load_state_dict(new_ckt)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    # ---- 自动断点续训: 若输出目录已存在 model_latest.pth 则恢复(权重/epoch/优化器/最优指标) ----
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        # 从 checkpoint 恢复: 起始 epoch、历史最优指标、优化器状态
        args.start_epoch = checkpoint['epoch'] 
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # setup dataloader
    # ---- 构建训练/验证 DataLoader: 数据 = ./data/<dataset>/ 目录下的逐帧 PNG ----
    img_transforms = transforms.ToTensor()          # PIL → [0,1] 归一化 Tensor
    DataSet = CustomDataSet
    train_data_dir = f'./data/{args.dataset.lower()}'
    val_data_dir = f'./data/{args.dataset.lower()}'

    # 训练集: frame_gap 控制取帧间隔; 分布式时用 DistributedSampler 切分数据, 否则 shuffle 打乱帧序
    train_dataset = DataSet(train_data_dir, img_transforms,vid_list=args.vid, frame_gap=args.frame_gap,  )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # 验证集: 用 test_gap 作为取帧间隔(可与训练的 frame_gap 不同), 不做 shuffle
    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap,  )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,  shuffle=False,
         num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)
    data_size = len(train_dataset)

    # ---- 只评估模式: 加载 --weight 后直接调用 evaluate() 并把结果写入 eval.txt ----
    if args.eval_only:
        print('Evaluation ...')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}\n'

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
        print_str += f'PSNR/ms_ssim on validate set for bit {args.quant_bit} with axis {args.quant_axis}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        print(print_str)
        with open('{}/eval.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n\n')        
        return

    # Training
    # ---- 主训练循环: 共 cycles 轮 × epochs 个 epoch(默认 cycles=1) ----
    start = datetime.now()
    total_epochs = args.epochs * args.cycles
    for epoch in range(args.start_epoch, total_epochs):
        model.train()
        ##### prune the network if needed #####
        # 到达剪枝节点时执行第 prune_num 次全局剪枝(剪完继续训练以恢复精度)
        if prune_net and epoch in args.prune_steps:
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparisity_num / 1e6 / total_params}')
        
        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []
        # iterate over dataloader
        # ---- 迭代 DataLoader: 每个批次 = (图像张量 data, 归一化帧索引 norm_idx) ----
        for i, (data,  norm_idx) in enumerate(train_dataloader):
            if i > 10 and args.debug:
                break
            embed_input = PE(norm_idx)      # 帧索引 → 位置编码向量(网络的实际输入)
            if local_rank is not None:
                data = data.cuda(local_rank, non_blocking=True)      # DDP: 搬到指定 GPU
                embed_input = embed_input.cuda(local_rank, non_blocking=True)
            else:
                data,  embed_input = data.cuda(non_blocking=True),   embed_input.cuda(non_blocking=True)
            # forward and backward
            output_list = model(embed_input)   # Generator 前向: 得到各尺度的重建图像
            # 多分辨率监督: 把 GT 下采样到每个输出尺度, 逐尺度计算损失(最后一层权重恒为 1)
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
            loss_sum = sum(loss_list)
            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)   # 更新学习率
            optimizer.zero_grad()
            loss_sum.backward()               # 反向传播
            optimizer.step()                  # 更新权重(即对视频的"编码"过程)

            # compute psnr and msssim
            # 逐批累计各尺度 PSNR / MS-SSIM, 形状 (batchsize, num_stage)
            psnr_list.append(psnr_fn(output_list, target_list))
            msssim_list.append(msssim_fn(output_list, target_list))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                train_msssim = torch.cat(msssim_list, dim=0) #(batchsize, num_stage)
                train_msssim = torch.mean(train_msssim.float(), dim=0) #(num_stage)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                    time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])
            train_msssim = all_reduce([train_msssim.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        # ---- 主进程写 tensorboard 日志, 更新历史最优指标, 打印每 epoch 耗时 ----
        if local_rank in [0, None]:
            h, w = output_list[-1].shape[-2:]
            is_train_best = train_psnr[-1] > train_best_psnr
            train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
            train_best_msssim = train_msssim[-1] if train_msssim[-1] > train_best_msssim else train_best_msssim
            writer.add_scalar(f'Train/PSNR_{h}X{w}_gap{args.frame_gap}', train_psnr[-1].item(), epoch+1)
            writer.add_scalar(f'Train/MSSSIM_{h}X{w}_gap{args.frame_gap}', train_msssim[-1].item(), epoch+1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}_gap{args.frame_gap}', train_best_psnr.item(), epoch+1)
            writer.add_scalar(f'Train/best_MSSSIM_{h}X{w}_gap{args.frame_gap}', train_best_msssim, epoch+1)
            print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(h, train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # 组装 checkpoint 字典(权重 + epoch + 历史最优指标 + 优化器状态)
        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
            'optimizer': optimizer.state_dict(),   
        }    
        # evaluation
        # ---- 每 eval_freq 个 epoch(或最后 10 个 epoch)在验证集上评估一次 ----
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
            val_end_time = datetime.now()
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(local_rank)])
                val_msssim = all_reduce([val_msssim.to(local_rank)])            
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                h, w = output_list[-1].shape[-2:]
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                is_val_best = val_psnr[-1] > val_best_psnr
                val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                val_best_msssim = val_msssim[-1] if val_msssim[-1] > val_best_msssim else val_best_msssim
                writer.add_scalar(f'Val/PSNR_{h}X{w}_gap{args.test_gap}', val_psnr[-1], epoch+1)
                writer.add_scalar(f'Val/MSSSIM_{h}X{w}_gap{args.test_gap}', val_msssim[-1], epoch+1)
                writer.add_scalar(f'Val/best_PSNR_{h}X{w}_gap{args.test_gap}', val_best_psnr, epoch+1)
                writer.add_scalar(f'Val/best_MSSSIM_{h}X{w}_gap{args.test_gap}', val_best_msssim, epoch+1)
                print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(h, val_psnr[-1].item(),
                     val_best_psnr.item(), val_best_msssim.item(), (val_end_time - val_start_time).total_seconds())
                print(print_str)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
                if is_val_best:
                    torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))

        if local_rank in [0, None]:
            # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            # 每轮覆盖保存 model_latest.pth(供断点续训); 训练指标最优时另存 model_train_best.pth
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))


@torch.no_grad()
def evaluate(model, val_dataloader, pe, local_rank, args):
    """在验证集上评估模型: 可选"量化 + Huffman 熵编码"统计码率, 并计算 PSNR / MS-SSIM / FPS.

    流程:
        1. 若 quant_bit != -1: 用 quantize_per_tensor 量化全部权重并加载回模型,
           再对量化编码做 Huffman 编码, 统计平均每权重比特数(即视频压缩的码率, 写入 eval.txt)
        2. 遍历 val_dataloader: 帧索引位置编码 → 前向(可重复 10 次测 FPS) → 逐尺度 PSNR / MS-SSIM
        3. 若 dump_images: 保存重建帧与 GT 帧到 visualize/ 目录
    返回 (val_psnr, val_msssim): 形状 (num_stage,), 最后一个元素对应最终分辨率.
    """
    # Model Quantization
    # ---- 模型量化(视频压缩流水线第 3 步) + Huffman 熵编码(第 4 步): 用于统计压缩后码率 ----
    if args.quant_bit != -1:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        for k,v in cur_ckt.items():
            large_tf = (v.dim() in {2,4} and 'bias' not in k)   # 2D/4D 权重 → 可按通道量化
            # 大张量按 args.quant_axis 逐通道量化, 其余(如 bias)整张量量化(axis=-1)
            quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1)
            valid_quant_v = quant_v[v!=0] # only include non-zero weights
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v            # 用量化重建的权重替换原权重
        cat_param = torch.cat(quant_weitht_list)   # 汇总所有非零权重的量化编码
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))       # 每个编码值出现的频次

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)  # 由数据分布生成 Huffman 码表
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]                 # 每个符号的编码比特长度
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)    # 平均每权重比特数
        # import pdb; pdb.set_trace; from IPython import embed; embed()       
        encoding_efficiency = avg_bits / args.quant_bit   # 相对量化位宽的压缩率(<1 表示熵编码进一步省空间)
        print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
        print(print_str)
        if local_rank in [0, None]:
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')       
        # 把量化后的权重真正加载回模型(之后的前向即"量化模型"的推理效果)
        model.load_state_dict(cur_ckt)

        # import pdb; pdb.set_trace; from IPython import embed; embed()

    psnr_list = []
    msssim_list = []
    if args.dump_images:   # dump_images: 需要保存重建帧(pred)与 GT 帧
        from torchvision.utils import save_image
        visual_dir = f'{args.outf}/visualize'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    time_list = []      # 记录每次前向耗时(用于估算解码 FPS)
    model.eval()
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        if i > 10 and args.debug:
            break
        embed_input = pe(norm_idx)     # 帧索引 → 位置编码向量
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data,  embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

        # compute psnr and msssim
        # eval_fps 模式: 每帧前向 10 次, 更准确地测解码速度
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            # embed_input = embed_input.half()
            # model = model.half()
            start_time = datetime.now()
            output_list = model(embed_input)   # 一次前向 = 解码一帧
            torch.cuda.synchronize()           # 等待 GPU 真正完成, 计时才准确
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        # dump predictions
        if args.dump_images:
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')

        # compute psnr and ms-ssim
        # 与训练一致: GT 下采样到各输出尺度, 逐尺度计算 PSNR / MS-SSIM
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)        
        if i % args.print_freq == 0:
            # 解码速度 FPS = 已前向帧数 / 累计前向时间
            fps = fwd_num * (i+1) * args.batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
    model.train()   # 恢复训练模式(evaluate 结束后交还训练循环)

    return val_psnr, val_msssim


if __name__ == '__main__':
    main()   # 脚本入口: python train_nerv.py ...
