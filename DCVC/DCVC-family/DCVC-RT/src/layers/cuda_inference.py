# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ============================================================================
# CUDA 定制算子封装 + 纯 PyTorch 回退实现。
# 若成功编译并 import 了 inference_extensions_cuda, 则 CUSTOMIZED_CUDA_INFERENCE=True,
# 所有函数走融合 CUDA kernel; 否则打印告警并回退到等价的 PyTorch 实现(慢但功能一致)。
# 这些算子是论文"降低操作开销"工程化的核心: 把量化、置零、mask 相乘、查表、
# padding、pixel_shuffle 等小操作各自融合为单个 kernel, 减少访存与 kernel 启动次数。
# ============================================================================

import os

import torch
import torch.nn.functional as F


CUSTOMIZED_CUDA_INFERENCE = False
try:
    from inference_extensions_cuda import process_with_mask_cuda, combine_for_reading_2x_cuda, \
        restore_y_2x_cuda, restore_y_4x_cuda, build_index_dec_cuda, \
        round_and_to_int8_cuda, clamp_reciprocal_with_quant_cuda, bias_quant_cuda, \
        add_and_multiply_cuda, bias_pixel_shuffle_8_cuda, replicate_pad_cuda, \
        build_index_enc_cuda, DepthConvProxy, SubpelConv2xProxy  # noqa: F401
    CUSTOMIZED_CUDA_INFERENCE = True
except Exception:  # pylint: disable=W0718
    pass


if not CUSTOMIZED_CUDA_INFERENCE and 'SUPPRESS_CUSTOM_KERNEL_WARNING' not in os.environ:
    print("cannot import cuda implementation for inference, fallback to pytorch.")


def round_and_to_int8(z):
    """把超先验 z 四舍五入并截断到 [-128,127], 返回 (float 的 z_hat, int8 的 z_hat_write)。

    int8 版本用于算术编码(符号本身), float 版本继续送入超先验解码网络。"""
    if CUSTOMIZED_CUDA_INFERENCE and z.is_cuda:
        z_int8 = round_and_to_int8_cuda(z)
        return z, z_int8

    z_hat = torch.clamp(torch.round(z), -128., 127.)
    z_hat_write = z_hat.to(dtype=torch.int8)
    return z_hat, z_hat_write


def clamp_reciprocal_with_quant(q_dec, y, min_val):
    """编码端量化: q_dec 下限截断到 min_val, 取倒数得 q_enc, 并返回 y*q_enc。

    q_dec 是先验网络预测的解码量化步长(乘回幅值用), q_enc=1/q_dec 用于把 y 缩放到
    整数符号域。CUDA 路径不修改输入, 与解码端行为保持一致。"""
    if CUSTOMIZED_CUDA_INFERENCE and q_dec.is_cuda:
        # q_dec is not inplace modified at decoder side
        q_dec = clamp_reciprocal_with_quant_cuda(q_dec, y, min_val)
        return q_dec, y

    q_dec = torch.clamp_min(q_dec, min_val)
    q_enc = torch.reciprocal(q_dec)
    y = y * q_enc
    return q_dec, y


def add_and_multiply(y_hat_0, y_hat_1, q_dec):
    """解码侧合并: (y_hat_0 + y_hat_1) * q_dec, 把两半检查板拼回并乘回量化步长。"""
    if CUSTOMIZED_CUDA_INFERENCE and y_hat_0.is_cuda:
        add_and_multiply_cuda(y_hat_0, y_hat_1, q_dec)
        return y_hat_0

    y_hat = y_hat_0 + y_hat_1
    y_hat = y_hat * q_dec
    return y_hat


def process_with_mask(y, scales, means, mask, force_zero_thres):
    """检查板量化(编码端核心):
    y_res = (y - means*mask) * mask     # 只保留 mask=1 位置的残差
    y_q   = round(y_res)                # 量化为整数符号
    (可选) scales<=thres 的位置 y_q 置零, 跳过编码(force_zero_thres 工程加速)
    y_hat = y_q + means*mask            # 编码端重建值
    返回 (y_res, y_q, y_hat, scales*mask)。"""
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda:
        thres = force_zero_thres if force_zero_thres is not None else -1.
        return process_with_mask_cuda(y, scales, means, mask, thres)

    scales_hat = scales * mask
    means_hat = means * mask

    y_res = (y - means_hat) * mask
    y_q = torch.round(y_res)
    if force_zero_thres is not None:
        cond = scales_hat > force_zero_thres
        y_q = y_q * cond
    y_q = torch.clamp(y_q, -128., 127.)
    y_hat = y_q + means_hat

    return y_res, y_q, y_hat, scales_hat


def combine_for_reading_2x(x, mask, inplace=False):
    """解码端把两路检查板参数"稠密化": (x*mask 的两个通道份)相加成单份,

    供逐符号查 CDF 行号用。inplace 时直接复用 x 的前半通道。"""
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda and x.is_contiguous():
        B, C, H, W = x.shape
        if inplace:
            out = x[:, :C // 2, :, :]
        else:
            out = torch.empty((B, C // 2, H, W), dtype=x.dtype, layout=x.layout, device=x.device)
        combine_for_reading_2x_cuda(out, x, mask)
        return out

    x = x * mask
    x0, x1 = x.chunk(2, 1)
    return x0 + x1


def restore_y_2x(y, means, mask):
    """解码端由稠密符号序列还原两路检查板的稀疏 y_hat: (cat(y,y)+means)*mask。"""
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        out = torch.empty_like(means)
        restore_y_2x_cuda(out, y, means, mask)
        return out

    return (torch.cat((y, y), dim=1) + means) * mask


def restore_y_2x_with_cat_after(y, means, mask, to_cat):
    """restore_y_2x 的融合版: 还原 y_hat_0 的同时把 common_params 拼接在其后,

    直接产出下一轮 SpatialPrior 需要的 cat_params, 省去两次独立 kernel。"""
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        B, C1, H, W = means.shape
        C2 = to_cat.shape[1]
        out = torch.empty((B, C1 + C2, H, W), dtype=means.dtype, layout=means.layout,
                          device=means.device)
        restore_y_2x_cuda(out[:, :C1, :, :], y, means, mask)
        out[:, C1:, :, :] = to_cat
        return out[:, :C1, :, :], out

    out = (torch.cat((y, y), dim=1) + means) * mask
    return out, torch.cat((out, to_cat), dim=1)


def restore_y_4x(y, means, mask):
    """I 帧 4 路版本: (cat(y,y,y,y)+means)*mask 还原四分之一通道的 y_hat。"""
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        out = torch.empty_like(means)
        restore_y_4x_cuda(out, y, means, mask)
        return out

    return (torch.cat((y, y, y, y), dim=1) + means) * mask


def build_index_dec(scales, scale_min, scale_max, log_scale_min, log_step_recip, skip_thres=None):
    """解码端 CDF 行号: index = round((log(scale)-log_min)/log_step), 截断到 [0,127]。

    skip_thres 启用时同时返回 skip_cond(scale>thres), 只有这些位置才需要解码。"""
    if CUSTOMIZED_CUDA_INFERENCE and scales.is_cuda:
        out = torch.empty_like(scales, dtype=torch.uint8)
        skip_cond = None
        if skip_thres is not None:
            skip_cond = torch.empty_like(scales, dtype=torch.bool)
        else:
            skip_thres = -1.

        build_index_dec_cuda(out, skip_cond, scales, scale_min, scale_max, log_scale_min,
                             log_step_recip, skip_thres)
        return out, skip_cond

    skip_cond = None
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
    return indexes, skip_cond


def build_index_enc(symbols, scales, scale_min, scale_max, log_scale_min,
                    log_step_recip, skip_thres=None):
    """编码端打包: int16 = (int8 符号 << 8) | uint8 CDF 行号, 只保留非跳过位置。

    C++ rANS 逐符号读取 int16, 一次解码出"符号值+CDF 行号"两个信息。"""
    if CUSTOMIZED_CUDA_INFERENCE and scales.is_cuda:
        out = torch.empty_like(scales, dtype=torch.int16)
        skip_cond = None
        if skip_thres is not None:
            skip_cond = torch.empty_like(scales, dtype=torch.bool)
        else:
            skip_thres = -1.

        build_index_enc_cuda(out, skip_cond, symbols, scales, scale_min, scale_max, log_scale_min,
                             log_step_recip, skip_thres)

        out = out[skip_cond]
        return out

    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    symbols = symbols.to(dtype=torch.int16)
    out = (symbols << 8) + indexes
    out = out.to(dtype=torch.int16)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
        out = out[skip_cond]
    return out


def replicate_pad(x, pad_b, pad_r):
    """边缘复制 padding(pad_b 行到底部, pad_r 列到右侧), 用于把尺寸补齐到块对齐。"""
    if pad_b == 0 and pad_r == 0:
        return x
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        return replicate_pad_cuda(x, pad_b, pad_r)
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def bias_pixel_shuffle_8(x, bias):
    """融合算子: 加 bias -> PixelShuffle(8) 上采样 -> clamp[0,1], 用于帧重建的最后一步。"""
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        B, C, H, W = x.shape
        assert B == 1
        out = torch.empty((B, 3, H * 8, W * 8), dtype=x.dtype, device=x.device, layout=x.layout)
        bias_pixel_shuffle_8_cuda(out, x, bias, C, H * W, W, True)
        return out

    out = x + bias[None, :, None, None]
    out = F.pixel_shuffle(out, 8)
    out = torch.clamp(out, 0., 1.)
    return out


def bias_quant(x, bias, quant_step):
    """融合算子: 加 bias 再乘量化缩放 quant_step(原地), 用于 1x1 卷积收尾。"""
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        bias_quant_cuda(x, bias, quant_step)
        return x

    out = x + bias[None, :, None, None]
    out = out * quant_step
    return out
