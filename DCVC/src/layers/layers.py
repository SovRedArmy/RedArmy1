# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 基础算子与网络构件层（论文中 Fig.4 的核心构件 DepthConvBlock 及量化/下界等算子）。
# DCVC-UF 全部主干均由 DepthConvBlock（深度可分离卷积块）堆叠而成，
# 这是其超高速编码/解码的关键之一：计算量远小于普通卷积，且对 GPU 友好。

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Function

from ..utils.transforms import ycbcr2rgb


def bit_estimator_z_prob(x, h, b, a):
    # z 的全分解先验（fully-factorized prior）的累积分布函数 CDF。
    # 由 4 层可学习仿射 + tanb 门控复合而成（等价于 CompressAI 中的 FactorizedPrior，
    # 但这里手工展开以便部署到 C++/CUDA 熵编码器），输出 sigmoid 得到累积概率。
    # this is accumulated prob
    for i in range(4):
        x = x * F.softplus(h[:, :, i:i+1, None]) + b[:, :, i:i+1, None]
        if i != 3:
            x = x + torch.tanh(x) * torch.tanh(a[:, :, i:i+1, None])
    return torch.sigmoid(x)


def bit_estimator_z_fwd(x, h, b, a):
    # 计算符号 x 落在 [x-0.5, x+0.5] 的概率 = CDF(x+0.5) - CDF(x-0.5)，用于估计 z 的比特数
    dtype = x.dtype
    x = x.float()
    h = h.float()
    b = b.float()
    a = a.float()
    lower = bit_estimator_z_prob(x - 0.5, h, b, a)
    upper = bit_estimator_z_prob(x + 0.5, h, b, a)
    prob = upper - lower
    return prob.to(dtype)


def get_mse_yuv_rgb(x, x_hat):
    # 同时计算 YCbCr 域与 RGB 域的 MSE：训练损失混合两种色彩空间的失真，
    # 对应论文“Unified YUV and RGB coding”（统一 YUV/RGB 编码）这一特性
    mse_yuv = torch.sum(F.mse_loss(x, x_hat, reduction='none'), dim=(2, 3))
    org_rgb = ycbcr2rgb(x, clamp=False)
    rec_rgb = ycbcr2rgb(x_hat, clamp=False)
    mse_rgb = torch.sum(F.mse_loss(org_rgb, rec_rgb, reduction='none'), dim=(1, 2, 3))
    return mse_yuv, mse_rgb


def mse_8frames_sum(mse, dist_weights):
    # HT（chunk=8 帧）模型的失真聚合：按帧在 chunk 中的位置加权求和。
    # dist_weights = [1.5, 0.16, 0.4] 分别对应第 8 帧（chunk 末帧）、偶数帧、奇数帧，
    # 与 LD 模型 index_map = [0,1,0,2,0,2,0,2] 的周期权重一一对应，
    # 保证 HT 与 LD 两种 chunk 结构使用同一套帧级权重训练。
    result = (mse[0] + mse[2] + mse[4] + mse[6]) * dist_weights[1] + \
        (mse[1] + mse[3] + mse[5]) * dist_weights[2] + mse[7] * dist_weights[0]
    return result


def mse_weighted_average(mse_yuv, mse_rgb, pixel_num):
    # 失真的加权平均：YUV 侧对 Y/U/V 做 10:1:1 的几何平均（近似 PSNR 加权），
    # 再与 RGB 侧失真按 0.8/0.2 混合。论文中 D = 0.8*D_yuv + 0.2*D_rgb。
    dtype = mse_yuv.dtype
    mse_yuv = mse_yuv.float()
    mse_rgb = mse_rgb.float()
    mse_yuv = mse_yuv / pixel_num
    mse_y, mse_u, mse_v = mse_yuv[:, 0], mse_yuv[:, 1], mse_yuv[:, 2]
    mse_yuv = torch.exp(
        0.0833 * (10 * torch.log(torch.clamp_min(mse_y, 1e-6))
                  + torch.log(torch.clamp_min(mse_u, 1e-6))
                  + torch.log(torch.clamp_min(mse_v, 1e-6)))) * 3
    mse_rgb = mse_rgb / pixel_num
    mse = mse_yuv * 0.8 + mse_rgb * 0.2
    return mse.to(dtype)


class LowerBound(Function):
    # 数值稳定的下界算子 clamp_min(x, bound) 的自定义 autograd 版本：
    # 反向传播时仅对“在界内的元素”或“梯度为负（想把值推小）的元素”放行梯度，
    # 避免 clamp 把梯度截断为 0 导致尺度参数无法训练（熵模型中保护 scale > 0）。
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs)
        ctx.bound = bound
        return torch.clamp_min(inputs, bound)

    @staticmethod
    def backward(ctx, grad):
        inputs = ctx.saved_tensors[0]
        bound = ctx.bound

        pass_through_1 = inputs >= bound
        pass_through_2 = grad < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through * grad, None


class QuantFunc(Function):
    # 量化算子：前向为取整 round()，反向把梯度直通（STE，直通估计器），
    # 从而让“量化”这个不可导操作可以参与端到端训练
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class SubpelConv2x(nn.Module):
    # 2 倍上采样子像素卷积：1x1/3x3 卷积扩充到 4 倍通道后 PixelShuffle 到 2 倍分辨率。
    # 用于解码器中把 1/16 分辨率潜变量恢复到 1/8 分辨率特征
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, force_bias=False):
        super().__init__()
        has_bias = (kernel_size > 1) or force_bias
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding, bias=has_bias),
            nn.PixelShuffle(2),
        )
        self.padding = padding

    def forward(self, x):
        return self.conv(x)


class WSiLU(nn.Module):
    # 加权 SiLU 激活函数：x * sigmoid(4x)，比 GELU/SiLU 更利于量化部署且速度更快
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    # 前馈网络（FFN）中的激活：先做 WSiLU，再把通道按 4 组求和压缩，
    # 等价于用“通道分组相加”代替 1x1 卷积做降维，进一步减少计算量
    def __init__(self):
        super().__init__()
        self.wsilu = WSiLU()

    def forward(self, x):
        x = self.wsilu(x)
        x1 = x[:, 0::4, :, :]
        x2 = x[:, 1::4, :, :]
        x3 = x[:, 2::4, :, :]
        x4 = x[:, 3::4, :, :]
        return x1 + x2 + x3 + x4


class DepthConvBlock(nn.Module):
    # 深度可分离卷积块（论文 Fig.4 中 HT-S 的核心构件）：
    # 结构为 “1x1 降维 → 3x3 深度卷积 → 1x1 升维”（dc 分支）+ 残差，
    # 再接一个 FFN 分支（1x1 升维 4 倍 → WSiLUChunkAdd → 1x1 降维）+ 残差。
    # 参数 dcb2=True 时内部通道数减半（更快的 HT-S 变体）。
    # 整个 DCVC-UF 的编解码器、超先验、熵参数网络均由该块堆叠而成，
    # 是论文实现“超快速度”的基础构件。
    def __init__(self, in_ch, out_ch, *, dcb2=False, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        ch_ratio = 1
        if dcb2:
            assert not shortcut
            ch_ratio = 2
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // ch_ratio, 1),
            WSiLU(),
            nn.Conv2d(out_ch // ch_ratio, out_ch // ch_ratio, 3, padding=1,
                      groups=out_ch // ch_ratio),
            nn.Conv2d(out_ch // ch_ratio, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4 // ch_ratio, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch // ch_ratio, out_ch, 1),
        )

    def forward(self, x):
        # adaptor: 通道数不匹配时的 1x1 适配卷积
        if self.adaptor is not None:
            x = self.adaptor(x)
        # dc: 深度可分离卷积残差分支；ffn: 前馈残差分支
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out


class ResidualBlockUpsample(nn.Module):
    # 上采样残差块：子像素卷积 2 倍上采样 + DepthConvBlock，用于超先验解码器/解码器
    def __init__(self, in_ch, out_ch, dcb2=False, shortcut=True, force_bias=False):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1, force_bias=force_bias)
        self.conv = DepthConvBlock(out_ch, out_ch, dcb2=dcb2, shortcut=shortcut)

        self.shortcut = shortcut

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out


class ResidualBlockWithStride2(nn.Module):
    # 下采样残差块：pixel_unshuffle 2 倍下采样 + 1x1 卷积 + DepthConvBlock，
    # 用于超先验编码器与时间先验编码器
    def __init__(self, in_ch, out_ch, dcb2=False, shortcut=True):
        super().__init__()
        self.down = nn.Conv2d(in_ch * 4, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, dcb2=dcb2, shortcut=shortcut)

        self.shortcut = shortcut

    def forward(self, x):
        out = F.pixel_unshuffle(x, 2)
        out = self.down(out)
        out = self.conv(out)
        return out
