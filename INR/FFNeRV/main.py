"""
FFNeRV 训练 / 评估主程序。

论文: FFNeRV: Flow-Guided Frame-Wise Neural Representations for Videos
      (Joo Chan Lee, Daniel Rho, Jong Hwan Ko, Eunbyung Park; ICML 2023; arXiv:2212.12294)

用法（详见 README.md）:
    表示(训练):  python main.py -e 300 ... --conv-type conv    --t-dim 64 128 256 512 ...
    压缩(训练):  python main.py -e 600 ... --conv-type compact --wbit 8 --t-dim 300 600 ...
    评估/解码:   python main.py ... --eval-only [--weight 权重路径 --dump-images]

整体调用流程:
    main()                      解析参数、创建输出目录
      -> train()                训练主函数（--eval-only 时直接转评估）
           -> Generator.forward()        (model.py) 网络前向
           -> resample() / get_grid()    光流变形（warping）
           -> loss_fn() / adjust_lr()    (utils.py) 损失与学习率
           -> 周期性 evaluate()          验证集评估 + 熵编码效率统计
"""
from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime
import math

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

from model import CustomDataSet, Generator
from utils import *

def main():
    """程序入口：解析命令行参数、准备输出目录，然后启动训练或评估。

    调用链: main() -> train(None, args)（单卡）或 mp.spawn 并在每个 GPU 进程中调用 train(local_rank, args)。
    """
    parser = argparse.ArgumentParser()

    # dataset parameters —— 数据集参数
    parser.add_argument('--vid',  default=[None], type=int,  nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame-gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset',)
    parser.add_argument('--test-gap', default=1, type=int, help='evaluation gap')
    parser.add_argument('--resol', nargs='+', default=[1280, 720], type=int, help='frame resolution')
    parser.add_argument('--offset', default=0, type=int, help='dataset offset')

    # model parameters —— 模型参数
    parser.add_argument('--t-dim', nargs='+', default=[256, 512], type=int, help='temporal resolution of grids')
    parser.add_argument('--fc-hw-dim', type=str, default='9_16_128', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=8, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list')
    parser.add_argument('--num-blocks', type=int, default=1)

    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument("--single-res", action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv-type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'compact' ,'deconv', 'bilinear'])

    # General training setups —— 常规训练设置
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not-resume-epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr-type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr-steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss-type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters —— 评估参数
    parser.add_argument('--eval-only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval-freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--dump-images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval-fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # pruning paramaters —— 剪枝参数
    parser.add_argument('--prune-steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune-ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters —— 分布式训练参数
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init-method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, —— 日志与输出目录
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    # FFNeRV 专属参数:
    #   --agg-ind: 流引导聚合引用的邻近帧相对索引（如 -2 -1 1 2 表示 t-2,t-1,t+1,t+2 四个参考帧）
    #   --wbit:    量化感知训练(QAT)的权重位宽（32=全精度；8=论文压缩版）
    parser.add_argument('--agg-ind', nargs='+', default=[-2,-1,0,1,2], type=int, help='relative indices of neighboring frames to reference')
    parser.add_argument('--wbit', default=32, type=int, help='QAT weight bit width')

    args = parser.parse_args()


    # agg_ind 设为全局变量，供 train()/evaluate() 中的流引导帧聚合使用
    global agg_ind
    agg_ind = args.agg_ind

    args.warmup = int(args.warmup * args.epochs)  # warmup 从"epoch 比例"换算成绝对 epoch 数

    print(args)
    torch.set_printoptions(precision=4)

    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    # 依据是否剪枝、是否仅评估，构造输出目录名 output/<outf>/<dataset>/<train|eval>...
    if args.prune_ratio < 1 and not args.eval_only:
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    extra_str = f'eval' if args.eval_only else 'train'

    exp_id = f'{args.dataset}/{extra_str}{prune_str}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
    	print('Will overwrite the existing output dir!')
    	shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    # 用实验 id 的哈希生成分布式初始化端口，避免多个实验之间端口冲突
    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2)
    args.ngpus_per_node = torch.cuda.device_count()
    # 多 GPU 分布式: 每张卡 spawn 一个 train 进程；否则单进程直接调用 train
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

# used for warping function
def get_grid(flow):
    """把"像素单位的光流"转换成 F.grid_sample 需要的归一化采样网格。

    思路: 目标像素坐标 (x,y) 加上光流 (dx,dy) 得到源像素坐标，
    再除以半宽/半高并减 1，映射到 grid_sample 要求的 [-1,1] 坐标系。
    输入 flow: (B, 2, m, n)；返回 (B, m, n, 2)（grid_sample 的输入格式）。
    """
    m, n = flow.shape[-2:]
    shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)  # 各像素的 x 坐标
    shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)  # 各像素的 y 坐标
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)  # 网格化

    grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)  # (1,2,m,n) 恒等网格（目标坐标）
    workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)  # 半宽/半高

    # 源坐标 = 目标坐标 + 光流，归一化到 [-1,1] 并调成 (B,m,n,2)
    flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

    return flow_grid

# warping function
def resample(feats, flow):
    """光流变形（warping）：按 flow 把参考帧的像素搬到当前帧坐标系下。

    feats: (B, 3, H, W) 参考帧；flow: (B, 2, h, w) 低分辨率光流（1/4 分辨率）。
    步骤: 光流双线性上采样到帧分辨率且数值等比放大（低分辨率下 1 像素的位移
    对应高分辨率下 scale 像素）-> get_grid() 构造采样网格 -> grid_sample 双线性采样。
    """
    scale_factor = float(feats.shape[-1]) / flow.shape[-1]
    flow = torch.nn.functional.interpolate(
        flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    flow = flow * scale_factor  # 光流数值随分辨率等比放大
    flow_grid = get_grid(flow)
    warped_feats = F.grid_sample(feats, flow_grid, mode="bilinear", padding_mode="border")
    return warped_feats

def train(local_rank, args):
    """训练（= 视频"编码"）主函数，也是 --eval-only 时的评估入口。

    流程:
      1. 构建 Generator（可选设置非结构化剪枝）、Adam 优化器、恢复 checkpoint
      2. 构建 train/val DataLoader，并分配全局帧缓存 key_frames
      3. --eval-only: 直接 evaluate() 后返回
      4. 训练循环（每个 epoch）:
         - 首个 epoch 先整体前向一遍填充帧缓存 key_frames
         - 每个 batch: Generator.forward() 得到 [流+一级权重头, 独立帧+二级权重头]
           -> 刷新帧缓存 -> 取邻近参考帧 -> resample() 光流变形
           -> 两级 softmax 加权聚合得到 [聚合帧, 独立帧, 最终帧]
           -> loss_fn 加权损失 -> adjust_lr -> 反向传播更新
         - 周期性 evaluate() 并保存 checkpoint

    参数: local_rank 为当前进程的 GPU 编号（非分布式时为 None）；args 为命令行参数。
    """
    cudnn.benchmark = True
    # 固定随机种子，保证可复现
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    # 记录训练/验证集上的历史最优指标
    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False

    # 构建 FFNeRV 生成器（即视频的隐式表示）
    model = Generator(fc_hw_dim=args.fc_hw_dim, expansion=args.expansion,
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid, aggs=len(args.agg_ind), wbit=args.wbit, t_dim=args.t_dim)

    ##### prune model params and flops #####
    # 非结构化剪枝设置（--prune-ratio < 1 时启用；默认 1.0 即关闭）。
    # 注: 'video_field' 是 NeRV/HNeRV 旧版字段名，本模型中实际为 video_grid（历史遗留）。
    prune_net = args.prune_ratio < 1
    # import pdb; pdb.set_trace; from IPython import embed; embed()
    if prune_net:
        param_list = []
        param_to_prune = []
        for k,v in model.named_parameters():
            print(k,)
            if 'video_field' in k:
                param_list.append(model.video_field[int(k.split('.')[-1])])
                #param_to_prune.append([model.video_field, k.split('.')[-1]])
            if 'weight' in k:
                # 收集各卷积层（conv1/conv2）的 weight，用于 L1 全局非结构化剪枝
                if 'layers' in k[:6] and 'conv1' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv1)
                    param_to_prune.append([model.layers[layer_ind].conv.conv1,'weight'])
                elif 'layers' in k[:6] and 'conv2' in k:
                    if not isinstance(model.layers[layer_ind].conv.conv2, nn.Identity):
                        param_to_prune.append([model.layers[layer_ind].conv.conv2,'weight'])
                        param_list.append(model.layers[layer_ind].conv.conv2)


        #print(param_list)
        #param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))  # 每个剪枝节点的保留比例
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]  # 比例 -> 绝对 epoch
        prune_num = 0
        if args.eval_only:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    # 统计参数量、打印/写日志、创建 tensorboard（仅 0 号进程或单卡执行）
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
    # —— 模型搬到 GPU：分布式用 DDP 包装（每卡一份、梯度同步）；多卡非分布式/单卡直接 .cuda()
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
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
        model.cuda() #model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    # Adam 优化器（beta 为命令行参数，默认 0.5）
    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    # resume from args.weight
    # —— 从 --weight 指定的权重文件加载（兼容旧格式键名: 去掉 'blocks.0.' 与 'module.' 前缀）
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
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))

    # resume from model_latest
    # —— 自动从输出目录的 model_latest.pth 断点续训（恢复权重/优化器/最优指标）
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

            sparsity_num = 0.
            for param in param_list:
                for param in param_list:
                    if isinstance(param, nn.Parameter):
                        sparsity_num += (param==0).sum()
                    else:
                        sparsity_num += (param.weight == 0).sum()
            # 注: 此处引用的 args.start_epoch 在下方才被赋值（原代码如此，仅在剪枝续训路径触发）
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparsity_num / 1e6 / total_params}')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        # 从 checkpoint 恢复起始 epoch、历史最优指标与优化器状态
        args.start_epoch = checkpoint['epoch']
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # setup dataloader
    # —— 训练集按 --frame-gap 抽帧；验证集按 --test-gap + --offset 抽帧（可评估帧插值）
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet
    train_data_dir = f'./datasets/{args.dataset.lower()}'
    val_data_dir = f'./datasets/{args.dataset.lower()}'

    train_dataset = DataSet(train_data_dir, img_transforms,vid_list=args.vid, resol=args.resol, frame_gap=args.frame_gap, )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True,
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, resol=args.resol, frame_gap=args.test_gap, offset=args.offset )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,  shuffle=False,
         num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)

    #used for both train and eval
    # —— 以下全局变量供 train()/evaluate() 的流引导聚合共享:
    #    data_size : 训练帧数（帧缓存长度的基准）
    #    len_agg   : 参考帧数+1（len_agg-1 = 参考帧个数 K = len(agg_ind)）
    #    key_frames: GPU 上的帧缓存，存放各帧最近一次前向得到的"独立帧"（half 省显存）；
    #                下标 data_size 是越界参考帧（首帧之前/末帧之后）指向的全零槽
    global data_size
    data_size = len(train_dataset)
    global len_agg
    len_agg = len(agg_ind)+1
    # frame buffer
    global key_frames
    key_frames = torch.cuda.FloatTensor(data_size+1, 3, args.resol[1], args.resol[0]).fill_(0).half()


    if args.eval_only:
        # 仅评估模式: （剪枝时先移除 mask 统计稀疏度）调用 evaluate() 并把结果写入 eval.txt
        print('Evaluation ...')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparsity_num / 1e6 / total_params}\n'

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        val_psnr, val_msssim = evaluate(model, train_dataloader, val_dataloader, local_rank, args)
        print_str += f'PSNR/ms_ssim on validate set for bit {args.wbit}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        print(print_str)
        with open('{}/eval.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n\n')
        return


    start = datetime.now()
    total_epochs = args.epochs * args.cycles  # cycles>1 时 lr 随 epoch%epochs 周期性重启

    updated = False  # 帧缓存是否已在首个 epoch 填充过
    for epoch in range(args.start_epoch, total_epochs):
        model.train()
        ##### prune the network if needed #####
        # —— 到达 prune_steps 节点时，做一次 L1 全局非结构化剪枝并打印稀疏度，随后继续训练
        if prune_net and epoch in args.prune_steps:
            prune_num += 1
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparsity_num / 1e6 / total_params}')

        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []

        # if frame buffer is empty, fill the buffer before training
        # —— 首个 epoch 先整体前向一遍（不更新参数），用当前网络输出填充帧缓存 key_frames，
        #    使后续流引导聚合有邻近"已解码帧"可引用
        if updated == False:
            for i, (data,  norm_idx) in enumerate(train_dataloader):
                t_input = norm_idx
                if local_rank is not None:
                    data = data.cuda(local_rank, non_blocking=True)
                    t_input = t_input.cuda(local_rank, non_blocking=True)
                else:
                    data,  t_input = data.cuda(non_blocking=True),   t_input.cuda(non_blocking=True)
                # forward and backward
                output_list = model(t_input)
                key_frames[torch.round(norm_idx*data_size).long()] = output_list[1][:,0:3,:,:].detach().clone().half()

        soft = nn.Softmax(dim=1)  # 通道维 softmax（两级聚合权重的归一化）
        # iterate over dataloader
        # —— 训练迭代：前向 -> 流引导两级聚合 -> 加权损失 -> 反向传播
        for i, (data,  norm_idx) in enumerate(train_dataloader):
            if i > 10 and args.debug:
                break
            t_input = norm_idx
            # 数据搬到 GPU（分布式时按 local_rank 指定卡）
            if local_rank is not None:
                data = data.cuda(local_rank, non_blocking=True)
                t_input = t_input.cuda(local_rank, non_blocking=True)
            else:
                data,  t_input = data.cuda(non_blocking=True),  t_input.cuda(non_blocking=True)
            # forward and backward
            # —— 前向得到两级头输出: output_list[0]=(B,3K,H/4,W/4) 流+一级权重,
            #    output_list[1]=(B,5,H,W) 独立帧RGB+二级权重
            output_list = model(t_input)

            idx = torch.round(norm_idx*data_size).long()  # 当前帧在帧缓存中的下标
            # update frame buffer
            # —— 用最新的"独立帧"预测刷新帧缓存（模拟解码端可用的已解码参考帧）
            key_frames[idx] = output_list[1][:,0:3,:,:].detach().clone().half()
            # flow-guided frame aggregation
            # indexing for batched warping
            # —— 计算各参考帧在缓存中的下标 = 当前帧下标 + 相对索引 agg_ind
            agg_idx = idx.unsqueeze(1).repeat(1, len_agg-1) + torch.tensor(agg_ind)
            # 越界（首帧之前/末帧之后）的参考帧统一指向全零槽 data_size
            agg_idx = torch.where(agg_idx >= 0, agg_idx, data_size)
            agg_idx = torch.where(agg_idx < data_size, agg_idx, data_size)
            # 取出 K 个参考帧并展平成 (B*K, 3, H, W)，以便批量变形
            agg_frames = key_frames[agg_idx.long()].to(torch.float32)
            agg_frames = agg_frames.reshape(-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
            # 头 1 的前 2K 通道是 K 个参考帧各自的光流 -> (B*K, 2, H/4, W/4)
            flows = output_list[0][:,:2*(len_agg-1),:,:]
            flows = flows.reshape(-1, 2, flows.shape[-2], flows.shape[-1])
            # warping
            # —— 按各自的光流把参考帧变形到当前帧坐标系 resample()
            agg_frames = resample(agg_frames, flows).unsqueeze(0)
            agg_frames = agg_frames.reshape(-1, len_agg-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
            # first aggregation
            # —— 第一级聚合: 头 1 剩余 K 通道是各参考帧权重，
            #    最近邻上采样 4 倍到全分辨率后在参考帧维做 softmax，逐像素加权求和
            wei1 = output_list[0][:,(len_agg-1)*2:,:,:]
            wei1 = torch.nn.functional.interpolate(wei1, scale_factor=4, mode='nearest')
            wei1 = soft(wei1)
            # the aggregated frame
            agg_frame = torch.sum(agg_frames * wei1.unsqueeze(2), dim=1, keepdim=True)  # 聚合帧
            # second aggregation
            # —— 第二级聚合: 把"独立帧"与"聚合帧"两个候选按头 2 的 2 通道权重图
            #    （softmax）逐像素融合: 静态区域偏向聚合帧（借邻帧补细节），
            #    运动区域偏向独立帧（避免多帧平均导致模糊）
            agg_frames = torch.cat([output_list[1][:,0:3,:,:].unsqueeze(1), agg_frame],dim=1)
            wei2 = soft(output_list[1][:,3:5,:,:]).unsqueeze(2)
            # aggregated frame, independent frame, final frame
            # —— 三个输出（聚合帧/独立帧/最终帧）都参与监督
            output_list = [agg_frame.squeeze(1), output_list[1][:,0:3,:,:], torch.sum(agg_frames * wei2, dim=1)]

            # GT 池化到与各输出相同尺寸（此处三个输出均为全分辨率，实际等于原 GT）
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            # weighted loss function
            # —— 辅助损失（聚合帧/独立帧）权重为 args.lw（README 取 0.1），最终帧权重为 1
            loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
            loss_sum = sum(loss_list)
            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            # compute psnr and msssim
            # —— 记录当前 batch 各输出的 PSNR / MS-SSIM
            psnr_list.append(psnr_fn(output_list, target_list))
            msssim_list.append(msssim_fn(output_list, target_list))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                # 周期性汇总并打印/写日志
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

        updated = True

        # collect numbers from other gpus
        # —— 多卡时对训练指标做 all_reduce 汇总
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])
            train_msssim = all_reduce([train_msssim.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        # —— 写 tensorboard、更新历史最优、打印 epoch 耗时
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

        state_dict = model.state_dict()
        # 组装 checkpoint（含 epoch、权重、历史最优指标与优化器状态，便于断点续训）
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
        # —— 每 eval_freq 个 epoch（以及最后 10 个 epoch）评估一次验证集
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, train_dataloader, val_dataloader, local_rank, args)
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

        # —— 每个 epoch 保存 model_latest.pth（最新）与 model_train_best.pth（训练最优）
        if local_rank in [0, None]:
            # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))

# quantization function used for evaluation
class qfn(torch.autograd.Function):
    """伪量化函数（model.qfn 的本地副本，evaluate() 统计熵编码效率时使用）。

    前向: k-bit 对称量化，量化级数 n = 2^(k-1)-1（8bit 时 n=127），
          out = floor(|x|*n)/n * sign(x)；
    反向: 直通估计器 STE，梯度原样传回。
    """
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

@torch.no_grad()
def evaluate(model, train_dataloader, val_dataloader, local_rank, args):
    """评估函数：统计熵编码效率（可选）、在验证集上逐帧重建并计算指标。

    流程:
      1. 若 --wbit 8 --eval-only: 模拟"QAT 量化 + Huffman 熵编码"的模型压缩效果，
         统计量化权重平均码长与编码效率（相对 wbit 定长）
      2. 在 train_dataloader 上整体前向一遍，填充帧缓存 key_frames
         （流引导聚合需要邻近参考帧的"已解码"结果）
      3. 在 val_dataloader 上逐帧前向 + 与训练相同的两级流引导聚合（只保留最终帧），
         计算 PSNR / MS-SSIM；可选 --eval-fps 测解码速度、--dump-images 导出预测图
    """
    # Saved weights are not applied quantization -> applying qfn
    # —— 保存的权重是全精度: 这里模拟 QAT 量化后做 Huffman 熵编码的压缩效果
    if args.wbit == 8 and args.eval_only:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weight_list = []
        for k,v in cur_ckt.items():
            #applying qfn
            # —— 与 QAT 前向一致: tanh 压缩到 (-1,1) + k-bit 伪量化
            weight = torch.tanh(v)
            quant_v = qfn.apply(weight, args.wbit)
            valid_quant_v = quant_v
            quant_weight_list.append(valid_quant_v.flatten())
        cat_param = torch.cat(quant_weight_list)   # 全部量化权重展平拼接
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))       # 每个量化电平出现的频次
        print(num_freq)

        # generating HuffmanCoding table
        # —— 按量化权重的频率分布构建 Huffman 编码表（频率高的电平码长更短）
        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]                 # 每个符号的码长（比特）
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num] # 总比特数 = 频次 * 码长
        avg_bits = total_bits / len(input_code_list)
        # import pdb; pdb.set_trace; from IPython import embed; embed()
        encoding_efficiency = avg_bits / args.wbit # 相对 wbit 定长存储的压缩率
        print_str = f'Entropy encoding efficiency for bit {args.wbit}: {encoding_efficiency}'
        print(print_str)
        if local_rank in [0, None]:
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
        model.load_state_dict(cur_ckt)             # 恢复全精度权重，不影响后续评估

        # import pdb; pdb.set_trace; from IPython import embed; embed()

    psnr_list = []
    msssim_list = []
    if args.dump_images:
        # 导出预测帧与 GT 帧到输出目录 visualize/ 下
        from torchvision.utils import save_image
        visual_dir = f'{args.outf}/visualize'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    time_list = []
    model.eval()
    soft = nn.Softmax(dim=1)
    global key_frames
    # 先在训练集上前向一遍填充帧缓存（评估的流引导聚合需要邻近参考帧）
    for i, (data,  norm_idx) in enumerate(train_dataloader):
        if i > 10 and args.debug:
            break
        t_input = norm_idx
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            t_input = t_input.cuda(local_rank, non_blocking=True)
        else:
            data,  t_input = data.cuda(non_blocking=True),   t_input.cuda(non_blocking=True)

        # compute psnr and msssim

        output_list = model(t_input)
        # (norm_idx-1e-6) 防止 round 后越过缓存末端
        key_frames[torch.round((norm_idx-1e-6)*data_size).long()] = output_list[1][:,0:3,:,:].detach().clone().half()

    # 验证集逐帧重建评估
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        t_input = norm_idx
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            t_input = t_input.cuda(local_rank, non_blocking=True)
        else:
            data,  t_input = data.cuda(non_blocking=True),  t_input.cuda(non_blocking=True)

        # compute psnr and msssim
        # —— --eval-fps 时每帧前向 10 次以统计解码速度
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            start_time = datetime.now()
            output_list = model(t_input)

            idx = torch.round((norm_idx-1e-6)*data_size).long()  # 当前帧在缓存中的下标
            # indexing for batched warping
            # —— 与训练一致的流引导两级聚合（见 train() 内注释）
            agg_idx = idx.unsqueeze(1).repeat(1, len_agg-1) + torch.tensor(agg_ind)
            agg_idx = torch.where(agg_idx >= 0, agg_idx, data_size)
            agg_idx = torch.where(agg_idx < data_size, agg_idx, data_size)
            agg_frames = key_frames[agg_idx.long()].to(torch.float32)
            agg_frames = agg_frames.reshape(-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])
            flows = output_list[0][:,:2*(len_agg-1),:,:]
            flows = flows.reshape(-1, 2, flows.shape[-2], flows.shape[-1])
            # warping
            agg_frames = resample(agg_frames, flows).unsqueeze(0)
            agg_frames = agg_frames.reshape(-1, len_agg-1, 3, agg_frames.shape[-2], agg_frames.shape[-1])

            # first aggregation
            wei1 = output_list[0][:,(len_agg-1)*2:,:,:]
            wei1 = torch.nn.functional.interpolate(wei1, scale_factor=4, mode='nearest')
            wei1 = soft(wei1)
            # the aggregate frame
            agg_frame = torch.sum(agg_frames * wei1.unsqueeze(2), dim=1, keepdim=True)
            # second aggregation
            agg_frames = torch.cat([output_list[1][:,0:3,:,:].unsqueeze(1), agg_frame],dim=1)
            wei2 = soft(output_list[1][:,3:5,:,:]).unsqueeze(2)
            # 评估只保留最终融合帧
            output_list = [torch.sum(agg_frames * wei2, dim=1)]

            torch.cuda.synchronize()  # 同步 GPU 以准确计时
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        # dump predictions
        # —— 导出预测帧 pred_*.png 与 GT 帧 gt_*.png
        if args.dump_images:
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')

        # compute psnr and ms-ssim
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)
        if i % args.print_freq == 0:
            fps = fwd_num * (i+1) * args.batchSize / sum(time_list)  # 解码速度（帧/秒）
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
    model.train()

    return val_psnr, val_msssim


if __name__ == '__main__':
    main()  # 程序入口: python main.py ...


