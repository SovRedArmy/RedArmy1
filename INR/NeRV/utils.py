# ==========================================================================================
# NeRV 训练/评估的通用工具函数
#  - PositionalEncoding   : 帧索引的位置编码(论文 Sec. 3.1)
#  - loss_fn              : 损失函数(L1/L2/SSIM 及各种 Fusion 组合)
#  - psnr_fn / msssim_fn  : 图像质量指标 PSNR / MS-SSIM
#  - quantize_per_tensor  : 模型权重量化(视频压缩流水线第 3 步)
#  - adjust_lr            : 学习率调度(warmup + cosine/step/const)
#  - all_gather / all_reduce / worker_init_fn : 分布式训练与 DataLoader 辅助函数
# ==========================================================================================
import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

def quantize_per_tensor(t, bit=8, axis=-1):
    """对权重张量 t 做均匀量化(论文视频压缩流水线第 3 步: 模型量化).

    量化公式: quant_t = round((t - t_min) / scale), 其中 scale = (t_max - t_min) / 2^bit.
    返回:
        quant_t : 量化后的整数编码(用于统计 Huffman 熵编码后的平均比特数)
        new_t   : 反量化(重建)后的权重, 即真正加载回模型参与推理的权重
    参数:
        bit  : 量化位宽(如 8 表示 8bit 量化, 论文显示 8bit 已接近 32bit 精度)
        axis : -1 表示整张量统一量化; 0/1 表示按第 0/1 维逐通道量化
               (卷积权重按输出通道、全连接权重按行逐通道量化, 精度更高)
    说明: 值为 0 的元素(被剪枝的权重)不参与 min/max 统计; scale 加 1e-19 防止除零.
    """
    if axis == -1:
        t_valid = t!=0                        # 剪枝后为 0 的权重不参与量化范围统计
        t_min, t_max =  t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / 2**bit      # 每个量化级对应的实数区间长度
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            
    # import pdb; pdb.set_trace; from IPython import embed; embed()       
    quant_t = ((t - t_min) / (scale + 1e-19)).round()   # 量化: 实数权重 → 整数编码
    new_t = t_min + scale * quant_t                     # 反量化: 整数编码 → 近似实数权重
    return quant_t, new_t

# all_gather: 分布式训练中, 收集所有进程(rank)上的同名张量并沿第 0 维拼接(本仓库主流程未直接调用)
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


# all_reduce: 分布式训练中, 对所有进程的张量求和(可选再除以进程数取平均),
# 用于多 GPU 场景下汇总 PSNR / MS-SSIM 等评估指标
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


class PositionalEncoding(nn.Module):
    """帧索引的位置编码(论文 Sec. 3.1, 类似 NeRF 的 Fourier 特征编码).

    把归一化帧索引 p∈[0,1) 映射为:
        Γ(p) = (sin(b^0·πp), cos(b^0·πp), sin(b^1·πp), cos(b^1·πp), ..., sin(b^{L-1}·πp), cos(b^{L-1}·πp))
    参数格式 pe_embed='b_L', 例如 '1.25_80' 表示底数 b=1.25、层数 L=80 → 输出 2L=160 维.
    'none' 表示不使用位置编码(输出 1 维原始帧索引; 论文消融显示不用 PE 时 PSNR 大幅下降).
    """
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()           # 'none' 或 '底数_层数'
        if self.pe_embed == 'none':
            self.embed_length = 1                  # 不编码: 输出维度为 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]  # 底数 b 与层数 L
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels    # sin/cos 成对出现 → 2L 维

    def forward(self, pos):
        """pos: (B,) 归一化帧索引 → (B, 2L) 位置编码向量."""
        if self.pe_embed == 'none':
            return pos[:,None]                     # 保持形状 (B,1)
        else:
            pe_list = []
            for i in range(self.levels):
                # 第 i 层频率: p · b^i · π, 频率逐层升高以区分相邻帧
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)         # 堆叠成 (B, 2L)


def psnr2(img1, img2):
    """逐像素 PSNR: 对每个像素算 -10·log10(误差平方), 并截断到 [0,50].

    注意与 psnr_fn 不同: 这里不先对整图求 MSE, 返回的是逐像素 PSNR 图(主流程未使用).
    """
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1                            # 图像值域为 [0,1], 故峰值取 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)  # 限制范围, 避免数值爆炸
    return psnr

def loss_fn(pred, target, args):
    """按 args.loss_type 计算重建损失(论文 Sec. 4.3 消融了不同损失组合).

    可选:
        L2 / L1 / SSIM    : 单一损失
        Fusion1 ~ Fusion12: L1/L2 与 SSIM/MS-SSIM 的加权组合,
                            论文最终采用 Fusion6 = 0.7·L1 + 0.3·(1-SSIM)
    pred / target: (B,3,H,W), 值域 [0,1]; target 先 detach(不对 GT 求梯度).
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
    """计算每个输出尺度上的 PSNR(图像值域 [0,1], 峰值取 1).

    output_list / target_list: 各尺度的 (B,3,H,W) 图像列表(多分辨率输出).
    返回 (batchsize, num_stage): 每个样本在每个尺度上的 PSNR, 最后一列对应最终分辨率.
    """
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')  # 整图 MSE
        psnr = -10 * torch.log10(l2_loss)   # PSNR = -10·log10(MSE)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)   # 复制到每个样本
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr

def msssim_fn(output_list, target_list):
    """计算每个输出尺度上的 MS-SSIM(多尺度结构相似度).

    MS-SSIM 需要多次下采样, 空间尺寸太小(高 < 160)时无法计算, 记为 0;
    返回 (batchsize, num_stage) 的 MS-SSIM 矩阵.
    """
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:   # 分辨率太低时 MS-SSIM 无意义, 直接记 0
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim

def RoundTensor(x, num=2, group_str=False):
    """把张量各元素四舍五入到 num 位小数并转成可打印字符串(仅用于日志输出).

    group_str=True 时按行分组: 行内用逗号分隔, 行间用 '/' 分隔.
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
    """按策略调整学习率并写回 optimizer(论文: Adam + warmup + cosine 退火).

    先把当前迭代折算成小数 epoch: cur_epoch += cur_iter / data_size,
    使学习率在一个 epoch 内也平滑变化(而非每个 epoch 才跳一次).
    可选策略 args.lr_type:
        cosine  : warmup 结束后按余弦曲线从 lr 衰减到 0(论文默认, 训练更稳)
        step    : 每跨过 args.lr_steps 中的一个节点, 学习率乘 0.1
        const   : 恒定学习率
        plateau : 占位实现, 效果同 const
    warmup 阶段(cur_epoch < args.warmup): 学习率从 0.1·lr 线性升到 lr, 避免训练初期不稳定.
    """
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)   # 迭代数折算成小数 epoch
    if args.lr_type == 'cosine':
        # 余弦退火: warmup 后学习率乘子从 1 平滑降到 0
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - args.warmup)/ (args.epochs - args.warmup)) + 1.0)
    elif args.lr_type == 'step':
        # 阶梯衰减: 遇到一个 lr_steps 节点就乘一次 0.1
        lr_mult = 0.1 ** (sum(cur_epoch >= np.array(args.lr_steps)))
    elif args.lr_type == 'const':
        lr_mult = 1
    elif args.lr_type == 'plateau':
        lr_mult = 1
    else:
        raise NotImplementedError

    if cur_epoch < args.warmup:
        # 线性 warmup: 0.1·lr → lr
        lr_mult = 0.1 + 0.9 * cur_epoch / args.warmup

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult   # 把新学习率写回每个参数组

    return args.lr * lr_mult

# DataLoader 多进程 worker 的随机种子初始化: 每个 worker 用 base seed 派生自己的种子,
# 保证 num_workers>0 时数据增强/随机操作仍可复现
def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    (为每个数据加载子进程重新设置随机种子, 保证实验可复现)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

class PositionalEncodingTrans(nn.Module):
    """Transformer 风格的"查表式"位置编码(备用实现, 当前训练主流程使用的是 PositionalEncoding).

    预生成 max_len × d_model 的 sin/cos 编码表(register_buffer, 不参与训练),
    前向时按帧索引四舍五入后直接查表取码.
    """
    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维用 cos
        self.register_buffer('pe', pe)                 # 固定表, 不作为可学习参数

    def forward(self, pos):
        # 归一化帧索引 → 整数行号, 直接查表得到 (B, d_model) 的位置编码
        index = torch.round(pos * self.max_len).long()
        p = self.pe[index]
        return p
