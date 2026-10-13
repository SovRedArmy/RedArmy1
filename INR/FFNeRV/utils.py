"""FFNeRV 工具函数集合（被 main.py 通过 `from utils import *` 引用）。

包含：
- all_gather / all_reduce: 多 GPU 分布式训练的通信辅助函数
- psnr2:                  逐元素 PSNR 图（未被主流程调用）
- loss_fn:                训练损失（L2 / L1 / SSIM 及 Fusion1~12 加权组合）
- psnr_fn:                按输出 stage 计算 PSNR
- msssim_fn:              按输出 stage 计算 MS-SSIM
- RoundTensor:            张量格式化成字符串，用于日志打印
- adjust_lr:              学习率调度（cosine / step / const，含 warmup）
- worker_init_fn:         DataLoader worker 的随机种子设定（保证可复现）
"""
import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

# 跨所有进程收集张量：把每个进程同形状的张量按第 0 维拼接后返回，
# 用于分布式训练时汇总各卡的预测/指标
def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


# 跨所有进程对张量求和；average=True 时再除以进程总数取平均，
# 用于多卡汇总 PSNR/MS-SSIM 等指标
def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def psnr2(img1, img2):
    """逐元素 PSNR：对每个元素计算 -10*log10((a-b)^2) 并截断到 [0,50]。

    注意这不是整图 PSNR 指标（主流程用的是 utils.psnr_fn），未被调用。
    """
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr

def loss_fn(pred, target, args):
    """训练损失函数：按 args.loss_type 计算预测帧与目标帧之间的损失。

    可选类型:
      - 'L2' / 'L1' / 'SSIM': 单一损失
      - 'Fusion1'~'Fusion12': L1/L2 与 (1-SSIM)/(1-MS-SSIM) 的不同加权组合；
        README 推荐的 Fusion6 = 0.7*L1 + 0.3*(1-SSIM)（论文默认损失）
    target 先 detach，保证目标不参与梯度回传。
    """
    target = target.detach()

    if args.loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean()
    elif args.loss_type == 'L1':
        loss = torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif args.loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion2':
        loss = 0.3 * torch.mean(torch.abs(pred - target)) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion4':
        loss = 0.5 * torch.mean(torch.abs(pred - target)) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion6':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'Fusion9':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion10':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion11':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion12':
        loss = 0.8 * torch.mean(torch.abs(pred - target)) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    return loss

def psnr_fn(output_list, target_list):
    """按输出 stage 逐个计算 PSNR（基于整图 MSE: PSNR = -10*log10(MSE)）。

    输入为"输出帧列表"与对应的"目标帧列表"（一一对应），
    返回 (batchsize, num_stage)：每行是同一 batch 在各 stage 的 PSNR。
    """
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr

def msssim_fn(output_list, target_list):
    """按输出 stage 逐个计算 MS-SSIM。

    仅当该 stage 输出的短边 >= 160 时才计算（MS-SSIM 需要多次下采样，
    小图会报错），否则记 0。返回 (batchsize, num_stage)。
    """
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
    """把张量格式化为字符串，便于日志打印。

    group_str=False: 所有元素四舍五入到 num 位后用逗号拼接成一行；
    group_str=True : 按行分组，行内逗号、行间用 '/' 分隔。
    """
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

def adjust_lr(optimizer, cur_epoch, cur_iter, data_size, args):
    """学习率调度：按训练进度直接改写 optimizer 中各组的学习率，并返回当前 lr。

    先把 batch 进度折算为小数 epoch: cur_epoch + cur_iter/data_size，
    再按 args.lr_type 计算乘子 lr_mult:
      - 'cosine': 从 warmup 结束到训练结束，乘子按余弦从 1 退火到 0（默认）
      - 'step'  : 每越过 args.lr_steps 中一个节点，乘 0.1
      - 'const' / 'plateau': 恒为 1
    warmup 阶段（cur_epoch < args.warmup）改为线性升温: 0.1 -> 1 倍。
    """
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    if args.lr_type == 'cosine':
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - args.warmup)/ (args.epochs - args.warmup)) + 1.0)
    elif args.lr_type == 'step':
        lr_mult = 0.1 ** (sum(cur_epoch >= np.array(args.lr_steps)))
    elif args.lr_type == 'const':
        lr_mult = 1
    elif args.lr_type == 'plateau':
        lr_mult = 1
    else:
        raise NotImplementedError

    if cur_epoch < args.warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / args.warmup

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult

# 为每个 DataLoader worker 进程重新播种 numpy/random，保证多进程数据加载的可复现性
def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

