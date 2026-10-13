import torch
import torch.nn as nn
import numpy as np
import math

def batched_conv(x, wb, conv_shape, ps_layer):
    """分组卷积前向（与 models/hyponets/layers.py 相同的实现）：
    把每个样本的卷积权重 wb[b] 展开为独立卷积核，用 groups=B 的 conv2d 一次前向计算
    B 个模型，再 PixelShuffle 上采样 —— 即论文 NeRV-Dec 的并行解码算子。"""
    B = wb.size(0)
    ch_out, ch_in, cur_ks, pad = conv_shape
    # import pdb; pdb.set_trace; from IPython import embed; embed()
    conv_weigts = wb[:,:-1].view((B, ch_in, cur_ks, cur_ks, ch_out))
    conv_weigts = conv_weigts.permute(0,-1,1,2,3).flatten(end_dim=1)
    conv_bias = wb[:,-1:].flatten()
    x = torch.nn.functional.conv2d(x, conv_weigts, conv_bias, stride=1, padding=pad, groups=B)
    return ps_layer(x)

def init_wb(out_ch, in_ch, ks):
    """按 nn.Conv2d 的默认方式初始化 (in_ch*ks*ks+1, out_ch) 形状的权重+偏置：
    Kaiming 均匀分布权重、fan_in 决定范围的均匀分布 bias；最后一行为 bias。"""
    weight = torch.empty(in_ch, out_ch, ks, ks)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(out_ch)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    wb_list = [weight.permute(0,2,3,1).flatten(end_dim=-2), bias[None]]
    return torch.cat(wb_list, dim=0).detach()

class HypoConvnets(nn.Module):
    """从零训练的 NeRV 基线模型（论文中 NeRV baseline 的对照实现，也是 NeRVEnc.hyponet 的单视频版本）：
    帧索引 -> 位置编码 -> [conv + PixelShuffle 上采样] x depth 层 -> RGB 帧。
    参数以 (in_ch*ks*ks+1, out_ch) 形式存在 self.params（'wb0','wb1',...）中，
    以便与超网络生成的权重格式保持一致（附录 0.B：theta_1 的卷积初始化非常重要）。"""

    def __init__(self, args):
        super().__init__()
        from math import ceil
        in_dim, out_dim, self.use_pe = 1, 3, True
        hid_dim, strds_list, ks_list, act = args.hid_dim, args.strds, args.ks, args.act
        self.pe_dim, self.pe_sigma, self.out_bias = args.pe_dim, args.pe_sigma, args.out_bias
        last_dim = in_dim * self.pe_dim

        depth = len(strds_list)
        ks_list += ks_list[-1:] * (depth - len(ks_list))
        ch = hid_dim

        self.ps_layers = nn.ModuleList()
        self.params = nn.ParameterDict()
        self.conv_shape_list = []
        for i, cur_strd in enumerate(strds_list):
            cur_dim = ch if i < depth - 1 else out_dim
            cur_ks = ks_list[i]
            ch_out = cur_dim*cur_strd**2
            self.params[f'wb{i}'] = nn.Parameter(init_wb(ch_out, last_dim, cur_ks))
            cur_pad = 0 if (cur_ks == cur_strd and i == 0) else ceil(cur_ks - 1) // 2
            self.conv_shape_list.append((ch_out, last_dim, cur_ks, cur_pad)) 
            self.ps_layers.append(nn.PixelShuffle(cur_strd))
            last_dim = cur_dim

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            NotImplementedError
        self.depth = depth

    def set_params(self, params):
        """用外部传入的（如超网络生成的）权重替换本模型参数。"""
        self.params = params

    def convert_posenc(self, x):
        """NeRV 式帧位置编码：x * exp(linspace(0, log(sigma), D/2)) 后取 [cos, sin] 拼接。"""
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def OutImg(self, x):
        """输出激活：sigmoid / tanh 映射到 [0,1]，或加常数偏置。"""
        if self.out_bias == 'sigmoid':
            return torch.sigmoid(x)
        elif self.out_bias == 'tanh':
            return (torch.tanh(x) * 0.5) + 0.5
        else:
            return x + float(self.out_bias)

    def forward(self, x):
        """解码前向：帧索引时间戳 x (T,) -> 位置编码 -> depth 层分组卷积+PixelShuffle -> (T,3,H,W)。"""
        B, T = 1, x.size(0)
        x = self.convert_posenc(x) 
        x = x.view(T, -1, 1, 1)  # (T, B*PE_DIM, 1, 1)

        for i in range(self.depth):
            x = batched_conv(x, self.params[f'wb{i}'][None], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = x.view((T, B, -1) + x.shape[-2:])
        # import pdb; pdb.set_trace; from IPython import embed; embed()      
        return x.squeeze(1)

def make_coord_grid(shape, range=(-1, 1), device=None):
    """
        Args:
            shape: tuple
            range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
        Returns:
            grid: shape (*shape, )

    构造归一化坐标网格：每个维度生成 [minv, maxv] 上均匀分布的坐标并 meshgrid 堆叠。
    NeRV 基线中用它生成 [-1,1] 的帧时间戳（论文 3.2 节：帧索引先归一到 [-1,1] 再做位置编码）。
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    return grid

from decord import VideoReader
from torchvision.transforms import CenterCrop, Resize, Compose
from torch.utils.data import Dataset

class VideoDataSet(Dataset):
    """（NeRV 基线用）单视频数据集：decord 解码整段视频，均匀采样 frames 帧并
    Resize+CenterCrop 到 crop_size，__getitem__ 返回 {'frame', 'idx'}。"""
    def __init__(self, args):
        vr = VideoReader(args.vid_path)
        frame_num = args.frames
        if frame_num == -1:
            frame_idx = list(range(len(vr)))
        else:
            frame_idx = [int(x*len(vr)/frame_num) for x in range(frame_num)]
        video = torch.from_numpy(vr.get_batch(frame_idx).asnumpy())
        video = video.permute(0,-1,1,2).float() / 255. # # T,H,W,C -> T,C,H,W
        transform = Compose([Resize(size=args.crop_size, antialias=False), CenterCrop(args.crop_size)])
        self.video = transform(video)
        
    def __len__(self):
        """样本数 = 采样后的帧数。"""
        return self.video.size(0)

    def __getitem__(self, idx):
        """返回 {'frame': 第 idx 帧, 'idx': 帧号}。"""
        return {'frame': self.video[idx], 'idx': idx}

def adjust_lr(optimizer, cur_epoch, args):
    """按 lr_type 字符串调整学习率：'hybrid_x_x_x_x_x'（多项式升+降）或
    'cosine_up_pow_minlr'（预热 + 余弦退火），返回当前 lr。"""
    # cur_epoch = (cur_epoch + cur_iter) / args.epochs
    if 'hybrid' in args.lr_type:
        up_ratio, up_pow, down_pow, min_lr, final_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 1 - (1 - final_lr) * ((cur_epoch - up_ratio) / (1. - up_ratio))**down_pow
    elif 'cosine' in args.lr_type:
        up_ratio, up_pow, min_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - up_ratio)/ (1 - up_ratio)) + 1.0)
    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult        

import random
def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    给每个 DataLoader worker 重新设定随机种子，保证可复现。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

import torch.distributed as dist
def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.

    DDP 跨进程张量求和（可选再除以进程数取平均）。
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors
