# -*- coding: utf-8 -*-
"""杂项工具集：分布式初始化与通信、配置读写、损失函数、学习率调度、评价指标、打印工具。"""
import os
import subprocess
import yaml
import random
import math
import time
import argparse
import numpy as np
import torch.distributed as dist
import torch
import torch.nn.functional as F
from collections import defaultdict
from pytorch_msssim import ms_ssim, ssim   # MS-SSIM / SSIM 的第三方实现，用于损失与指标
from tqdm import tqdm


def setup_for_distributed(is_master):
    '''
    Disable the print for non-master process
    （劫持内建 print：非主进程的打印被静默，避免分布式日志混乱）
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:  # 仅主进程（或强制 force=True）真正输出
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def init_distributed_mode(args):
    """初始化分布式训练：依次识别 torchrun（RANK/WORLD_SIZE 环境变量）与 SLURM 调度环境，
    两者都不存在时回退为单机模式（args.distributed=False, args.rank=None）。
    分布式模式下设置 NCCL 后端并初始化进程组。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  # torchrun / torch.distributed.launch 启动
        args.rank = int(os.environ['RANK'])            # 全局进程编号
        args.world_size = int(os.environ['WORLD_SIZE'])  # 总进程数
        args.gpu = int(os.environ['LOCAL_RANK'])        # 本机 GPU 编号
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:  # SLURM 集群调度启动
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = int(os.environ['SLURM_NODELIST'])
        node_list = str(os.environ['SLURM_NODELIST'])
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(      # 取首节点主机名作为通信地址 MASTER_ADDR
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        port_str = str(args.port)
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', port_str)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')  # 单机模式：直接返回
        args.distributed = False
        args.rank = None
        return 
    
    args.distributed = True

    torch.cuda.set_device(args.gpu)  # 绑定当前进程使用的 GPU
    args.dist_backend = 'nccl'       # GPU 间通信后端
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  # 初始化进程组
    torch.distributed.barrier()      # 所有进程同步，确保初始化完成
    setup_for_distributed(args.rank == 0)  # 仅 rank 0 保留 print 输出


def get_sha():
    """返回当前代码的 git 版本信息（commit sha / 是否有未提交改动 / 分支名），用于实验溯源。"""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'has uncommited changes' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def load_yaml_as_dict(cfg_path):
    """读取 yaml 配置文件并返回字典（全部超参数的唯一来源）"""
    with open(cfg_path) as yaml_file:
        yaml_as_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_as_dict


def dump_cfg_yaml(cfg, output_dir):
    """把配置字典原样写回输出目录 cfg.yaml，便于追溯该次实验设置"""
    save_path = os.path.join(output_dir, 'cfg.yaml')
    fp = open(save_path, 'w')
    fp.write(yaml.dump(cfg))
    fp.close()


def is_dist_avail_and_initialized():
    """分布式是否可用且已初始化"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """获取当前进程的全局编号（未启用分布式时返回 0）"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """获取总进程数（未启用分布式时返回 1）"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """当前是否为主进程（rank==0）"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """仅在主进程执行 torch.save，避免多进程写坏 checkpoint"""
    if is_main_process():
        torch.save(*args, **kwargs)


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    （为 DataLoader 的每个 worker 重新设置 numpy/random 种子，保证数据加载可复现）
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def to_cuda(dict_data, device):
    """把数据字典 {img_id, img_gt} 整体搬到指定设备（异步拷贝）"""
    dict_data = {k: v.to(device, non_blocking=True) for k, v in dict_data.items()}
    return dict_data


def loss_fn(pred, target, loss_type):
    """单个尺度上的损失计算。默认配置 Fusion6 = 0.7*L1 + 0.3*(1-SSIM)，即论文公式 L = 0.7||v-v_hat||_1 + 0.3(1-SSIM)。"""
    target = target.detach()  # 目标不参与梯度回传

    if loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean()       
    elif loss_type == 'L1':
        loss = torch.mean(torch.abs(pred - target))
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':   # 以下 Fusion1~12 为各种 L1/L2 与 SSIM/MS-SSIM 的加权组合消融
        loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = 0.3 * torch.mean(torch.abs(pred - target)) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion4':
        loss = 0.5 * torch.mean(torch.abs(pred - target)) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion6':   # <- 默认配置使用：论文的 L1 + SSIM 组合（alpha=0.7）
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * torch.mean(torch.abs(pred - target))
    elif loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * torch.mean(torch.abs(pred - target))
    elif loss_type == 'Fusion9':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion10':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion11':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion12':
        loss = 0.8 * torch.mean(torch.abs(pred - target)) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    return loss


def loss_compute(output_list, target_list, loss_type='Fusion6'):
    """多尺度损失：对每个尺度的 (output, target) 分别调用 loss_fn，返回损失列表（由调用方求和）"""
    loss_list = [loss_fn(output, target, loss_type) for output, target in zip(output_list, target_list)]
    return loss_list


def adjust_lr(optimizer, cur_epoch, all_epoch, cur_iter, data_size, cfg):
    """按“全局迭代步数”做 warmup + cosine 学习率调度（论文的 warmup-cosine 策略）。

    前 lr_point（默认 0.2）比例的迭代：学习率从 0.1*lr 线性升到 lr（warmup）；
    之后：按余弦曲线从 lr 衰减到 0。
    返回当前学习率。
    """
    all_iter = all_epoch * data_size   # 训练总迭代数
    now_iter = cur_epoch * data_size + cur_iter  # 当前全局迭代数

    if cfg['optim']['lr_schedule'] == 'warmup_cosine':
        if now_iter < all_iter * cfg['optim']['lr_point']:  # warmup 阶段：线性升温
            lr_mult = 0.1 + 0.9 * now_iter / (all_iter * cfg['optim']['lr_point'])
        else:  # cosine 阶段：cos 半周期衰减到 0
            whole = all_iter - all_iter * cfg['optim']['lr_point']
            cur = now_iter - all_iter * cfg['optim']['lr_point']
            lr_mult = 0.5 * (math.cos(math.pi * cur / whole) + 1.0)
    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):  # 直接改写优化器参数组的学习率
        param_group['lr'] = cfg['optim']['lr'] * lr_mult

    return cfg['optim']['lr'] * lr_mult


def psnr_fn(output_list, target_list):
    """按尺度计算 PSNR = -10*log10(MSE)，返回 (batchsize, num_stage)。

    注意：MSE 是对整个 batch 的图像一起求均值后再取对数，
    batchsize=1 时即逐帧 PSNR；num_stage 为输出尺度数（sin_res=true 时为 1）。
    """
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')  # 该尺度上的 MSE
        psnr = -10 * torch.log10(l2_loss)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)  # 广播到 batch 维
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr


def msssim_fn(output_list, target_list):
    """按尺度计算 MS-SSIM，返回 (batchsize, num_stage)。

    低分辨率尺度（高 < 160）无法做多层 MS-SSIM，记为 0。"""
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim


def RoundTensor(x, num=2, group_str=False):
    """把张量按指定小数位四舍五入后转成字符串，便于日志打印。
    group_str=False 时按逗号拼接全部元素；True 时按行分组、行间用 / 分隔。"""
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str


def reduce_dict(input_dict, average=True):
    """分布式场景：把各进程的指标张量 all_reduce 求和（默认再除以进程数取平均）。"""
    world_size = get_world_size()
    if world_size < 2:  # 单进程无需通信
        return input_dict
    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):  # 排序保证各进程堆叠顺序一致
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)  # 跨进程求和
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
