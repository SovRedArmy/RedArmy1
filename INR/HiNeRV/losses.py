"""
损失与指标：所有函数均按帧(frame-wise)计算，输入输出为 [N, C, T, H, W]。
论文默认训练损失为 0.7·L1 + 0.3·MS-SSIM(5×5 窗口)；评估指标为 PSNR 与 MS-SSIM。
"""
from utils import *
import pytorch_msssim


def check_shape(x, y):
    """校验两个张量形状/步长一致且为 5D（[N,C,T,H,W]）。"""
    assert x.shape == y.shape, 'shape of tensors must be the same!'
    assert x.stride() == y.stride(), 'strides of tensors must be the same!'
    assert x.ndim == y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'


def mse(x, y):
    """
    Compute the per-frame MSE loss
    逐帧 MSE：返回 [N, T]，每帧一个值。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.mse_loss(x, y, reduction='none').mean(dim=2)


def l1(x, y):
    """
    Compute the per-frame L1 loss
    逐帧 L1：返回 [N, T]。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.l1_loss(x, y, reduction='none').mean(dim=2)


def psnr(x, y, v_max=1.):
    """
    Compute the per-frame PSNR
    逐帧 PSNR = 10·log10(v_max² / MSE)。
    """
    return 10 * torch.log10((v_max ** 2) / (mse(x, y) + 1e-9))


def ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame SSIM
    逐帧 SSIM（基于 pytorch_msssim），返回 [N, T]。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


def ms_ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame MS-SSIM
    逐帧多尺度 SSIM，返回 [N, T]。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ms_ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


def compute_loss(name, x, y):
    """
    损失分发：按名称调用对应损失函数。
    SSIM 类损失返回 (1 - SSIM)；窗口大小可通过名称后缀 _3x3/_5x5/_7x7 指定
    （论文发现 MS-SSIM 用 5×5 小窗口更好）。
    """
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'ssim':
        return 1. - ssim(x, y)
    elif name == 'ssim_3x3':
        return 1. - ssim(x, y, win_size=3)
    elif name == 'ssim_5x5':
        return 1. - ssim(x, y, win_size=5)
    elif name == 'ssim_7x7':
        return 1. - ssim(x, y, win_size=7)
    elif name == 'ms-ssim':
        return 1. - ms_ssim(x, y)
    elif name == 'ms-ssim_3x3':
        return 1. - ms_ssim(x, y, win_size=3)
    elif name == 'ms-ssim_5x5':
        return 1. - ms_ssim(x, y, win_size=5)
    elif name == 'ms-ssim_7x7':
        return 1. - ms_ssim(x, y, win_size=7)
    else:
        raise ValueError


def compute_metric(name, x, y):
    """指标分发：按名称调用 mse/l1/psnr/ssim/ms-ssim。"""
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'psnr':
        return psnr(x, y)
    elif name == 'ssim':
        return ssim(x, y)
    elif name == 'ms-ssim':
        return ms_ssim(x, y)
    else:
        raise ValueError


def compute_regularization(name, model):
    """正则项分发（当前未实现任何正则）。"""
    raise ValueError