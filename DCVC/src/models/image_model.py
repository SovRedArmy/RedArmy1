# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 帧内（图像）编解码模型 DMCI —— DCVC-UF-Intra（论文 Sec.3.1）。
# 结构：8 倍 pixel_unshuffle 下采样 → 深度卷积编码器 → 1/16 潜变量 y；
# 超先验 z 与四分组棋盘格熵编码（与视频模型共用 forward_prior_4x）；
# 解码侧对称。QP 可查表缩放（q_scale_*，64 级），单模型覆盖宽码率。
# 视频序列的首帧（I 帧）由本模型编码，其重建帧经 pixel_unshuffle 后
# 作为 P 模型的初始参考特征。

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, QuantFunc, ResidualBlockUpsample, ResidualBlockWithStride2
from ..utils.common import loss_func


g_ch_src = 3 * 8 * 8     # pixel_unshuffle(8) 后的通道数（3 通道 × 64）
g_ch_enc_dec = 384       # 编/解码主干的通道宽度
g_ch_y = 256             # 潜变量 y 的通道数
g_ch_z = 128             # 超先验潜变量 z 的通道数


class IntraDecoder(nn.Module):
    # 帧内解码器：1x1 上采样卷积 ×2（1/16 → 1/4）+ 深度卷积块 + pixel_shuffle(8) 还原全分辨率
    def __init__(self):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(g_ch_y, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src)

    def forward(self, x, quant_step):
        # y_hat (1/16) → 1/4 特征 → 按 QP 量化步长缩放 → 1x1 卷积 → pixel_shuffle(8) 全分辨率
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


class IntraEncoder(nn.Module):
    # 帧内编码器：pixel_unshuffle(8) 下采样到 1/8 → 深度卷积块 → QP 缩放 → 步长 2 卷积到 1/16
    def __init__(self):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, g_ch_y, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        # 全分辨率 x → 1/8 特征 → QP 缩放 → 1/16 潜变量 y
        out = F.pixel_unshuffle(x, 8)
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)


class IntraHyperDecoder(nn.Module):
    # 超先验解码器：把 1/64 的 z_hat 上采样回 1/16，生成熵参数（scales/means）
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class IntraHyperEncoder(nn.Module):
    # 超先验编码器：把 1/16 的 y 两次下采样到 1/64 得到 z
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class IntraSpatialPrior(nn.Module):
    # 空间先验网络：输入 [已解码的 y_hat 分组, 压缩后的公共参数]，
    # 输出下一分组专用的 (scales, means)（帧内模型二者都输出）
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            nn.Conv2d(g_ch_y * 2, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class IntraYPriorFusion(nn.Module):
    # 超先验熵参数的融合/精炼网络（帧内模型无时间先验，仅精炼超先验输出）
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            nn.Conv2d(g_ch_y * 2, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class DMCI(CompressionModel):
    # DCVC-UF 的帧内模型（Deep Model for Compression of Intra-frame）：
    # 组装上面各子模块，实现 forward_one_frame（训练）/ compress、decompress（测试）
    def __init__(self):
        super().__init__(z_channel=g_ch_z)

        self.enc = IntraEncoder()          # 编码器：x → y
        self.hyper_enc = IntraHyperEncoder()  # 超先验编码器：y → z

        self.hyper_dec = IntraHyperDecoder()   # 超先验解码器：z_hat → 熵参数
        self.y_prior_fusion = IntraYPriorFusion()  # 熵参数精炼

        self.y_spatial_prior_reduction = nn.Conv2d(g_ch_y * 2, g_ch_y, 1)  # 公共参数通道压缩
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(g_ch_y * 2, g_ch_y * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(g_ch_y * 2, g_ch_y * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(g_ch_y * 2, g_ch_y * 2, force_adaptor=True)
        self.y_spatial_prior = IntraSpatialPrior()  # 渐进空间先验（4 步均值/尺度再估计）

        self.dec = IntraDecoder()  # 解码器：y_hat → x_hat

        # QP 查找表：64 个 QP 各对应一组逐通道缩放向量（编码/解码、1/8 域与 1/16 域分开）
        self.q_scale_enc = nn.Parameter(torch.ones((self.qp_num(), g_ch_enc_dec)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.qp_num(), g_ch_enc_dec)))
        self.q_scale_y_enc = nn.Parameter(torch.ones((self.qp_num(), g_ch_y)))
        self.q_scale_y_dec = nn.Parameter(torch.ones((self.qp_num(), g_ch_y)))
        self._initialize_weights()

    def forward_one_frame(self, x, qp, recon_only=False):
        # 训练前向：x → 编码 → 量化（STE）→ 熵编码参数 → 四分组渐进编码 → 解码重建。
        # recon_only=True 时只返回重建（用作视频训练时的 I 帧参考）
        curr_q_enc = self.index_select_dim0(self.q_scale_enc, qp)    # QP 查表：编码缩放
        curr_q_dec = self.index_select_dim0(self.q_scale_dec, qp)    # QP 查表：解码缩放
        curr_y_q_enc = self.index_select_dim0(self.q_scale_y_enc, qp)  # QP 查表：y 编码缩放
        curr_y_q_dec = self.index_select_dim0(self.q_scale_y_dec, qp)  # QP 查表：y 解码缩放

        y = self.enc(x, curr_q_enc)          # 编码得到 1/16 潜变量 y
        z = self.hyper_enc(y)                # 超先验 z
        z_hat = QuantFunc.apply(z)           # z 取整（STE）

        params = self.hyper_dec(z_hat)       # 超先验解码出公共熵参数
        params = self.y_prior_fusion(params)  # 精炼
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW]      # 裁掉 padding 引入的多余部分
        y_res, y_q, y_hat, scales_hat = self.forward_prior_4x(  # 四分组棋盘格熵编码
            y, curr_y_q_enc, curr_y_q_dec,
            params, self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec)  # 解码重建
        if recon_only:
            return x_hat

        # ---- 训练用比特/失真估计 ----
        y_for_bit = self.add_noise(y_res)    # 加噪代理量化误差
        z_for_bit = self.add_noise(z)
        bits_y = self.get_y_bits(y_for_bit, scales_hat)  # y 的比特数（高斯熵模型）
        bits_z = self.get_z_bits(z_for_bit, qp)          # z 的比特数（全分解先验）

        mse = self.get_mse(x, x_hat)         # YUV+RGB 混合失真
        bits_y = torch.sum(bits_y, dim=(1, 2, 3))
        bits_z = torch.sum(bits_z, dim=(1, 2, 3))
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp = (bits_y + bits_z) / pixel_num

        return {
            'x_hat': x_hat,
            'mse': mse,
            'bpp': bpp,
            'bits_y': bits_y,
            'bits_z': bits_z,
        }

    def compress(self, x, qp, padding_b, padding_r):
        # 测试压缩入口：调用 C++/CUDA 推理扩展 DMCIProxy 完成整条
        # 前向 + 量化 + rANS 熵编码，返回字节码流（首次调用时懒加载并注入权重与 CDF）
        if self.proxy is None:
            try:
                from inference_extensions_cuda import DMCIProxy
            except Exception:
                raise NotImplementedError(
                    'cannot import cuda implementation for inference. '
                    'Please build the inference extensions first.'
                )
            state_dict = self.state_dict()
            state_dict = self.add_cdf_to_state_dict(state_dict)
            self.proxy = DMCIProxy()
            self.proxy.set_param(state_dict, self.gaussian_encoder.skip_thres)
        bit_stream, x_hat, ec_parallel = self.proxy.compress(x, qp, padding_b, padding_r)
        return {
            'bit_stream': bit_stream.tobytes(),
            'x_hat': x_hat,
            'ec_parallel': ec_parallel,
        }

    def decompress(self, bit_stream, sps, qp, ec_part):
        # 测试解压入口：从字节码流经 CUDA 代理解码出重建帧
        # sps 提供真实高宽，ec_part 为熵编码并行子流的分配信息
        x_hat = self.proxy.decompress(
            np.frombuffer(bit_stream, dtype=np.uint8), qp, sps['height'], sps['width'], ec_part)
        return {'x_hat': x_hat}

    def forward(self, x, qp, lambdas=None, get_loss_info=False, recon_only=False):
        # 训练入口：R-D 损失 = lambda * D + R（见 utils.common.loss_func）
        result = self.forward_one_frame(x, qp, recon_only)
        loss = loss_func(result, lambdas)
        info = None
        if get_loss_info:
            _, _, H, W = x.size()
            pixel_num = H * W
            info = self.get_loss_info(result, loss, pixel_num)
        return loss['loss'], info
