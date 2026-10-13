# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 色彩空间转换工具：BT.709 的 RGB↔YCbCr，以及 YUV420↔YUV444 的上/下采样。
# DCVC-UF 内部统一在 YCbCr444（值域中心化到 0）上编码，
# 通过这些转换同时支持 YUV420 与 RGB 输入（论文“统一 YUV/RGB 编码”）。

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F


YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    'ITU-R_BT.709': (0.2126, 0.7152, 0.0722)  # BT.709 的亮度加权系数
}
Kr, Kg, Kb = YCBCR_WEIGHTS['ITU-R_BT.709']


def rgb2ycbcr(rgb, is_bgr=False):
    # torch 张量版 RGB→YCbCr（BT.709），输出范围 [0,1]
    if is_bgr:
        b, g, r = rgb.chunk(3, -3)
    else:
        r, g, b = rgb.chunk(3, -3)
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    ycbcr = torch.clamp(ycbcr, 0., 1.)
    return ycbcr


def rgb2ycbcr_np(rgb, quant=False):
    # numpy 版 RGB→YCbCr（数据集加载时使用；通道在最后）。
    # quant=True 时量化到 8bit 网格，模拟真实 YUV 输入的量化损失
    '''
    Note: channel_last memory format
    input is hxwx3 RGB float numpy array
    output is ycbcr: hxwx3
    '''
    h, w, c = rgb.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    r, g, b = np.split(rgb, 3, axis=2)
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = np.concatenate((y, cb, cr), axis=2)
    ycbcr = np.clip(ycbcr, 0., 1.)

    if quant:
        ycbcr = np.round(ycbcr * 255.) / 255.

    return ycbcr


def ycbcr2rgb(ycbcr, is_bgr=False, clamp=True):
    # torch 张量版 YCbCr→RGB（BT.709）；clamp=True 时裁剪到 [0,1]
    dtype = ycbcr.dtype
    ycbcr = ycbcr.float()
    y, cb, cr = ycbcr.chunk(3, -3)
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    if is_bgr:
        rgb = torch.cat((b, g, r), dim=-3)
    else:
        rgb = torch.cat((r, g, b), dim=-3)
    if clamp:
        rgb = torch.clamp(rgb, 0., 1.)
    return rgb.to(dtype)


def ycbcr420_to_444_np(y, uv, order=0, separate=False):
    # YUV420 → YCbCr444：UV 分量上采样 2 倍（默认最近邻，与 HEVC 一致），
    # 用于把读取的 YUV420 帧转换为模型输入格式
    '''
    y is 1xhxw Y float numpy array
    uv is 2x(h/2)x(w/2) UV float numpy array
    order: 0 nearest neighbor (default), 1: binear
    return value is 3xhxw YCbCr float numpy array
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    if separate:
        return y, uv
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def yuv_444_to_420(yuv):
    # YCbCr444 → YUV420：UV 分量 2x2 平均池化下采样，用于把重建帧转回 YUV420 输出
    def _downsample(tensor):
        return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    y = yuv[:, :1, :, :]
    uv = yuv[:, 1:, :, :]

    return y, _downsample(uv)
