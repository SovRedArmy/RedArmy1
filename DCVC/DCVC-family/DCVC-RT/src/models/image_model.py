# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ============================================================================
# DCVC-RT 的 I 帧(帧内)图像压缩模型 DMCI (Deep Model for Compression - Intra)。
# 论文: "Towards Practical Real-Time Neural Video Compression" (CVPR 2025)。
#
# I 帧模型与 P 帧模型(DMC)共享同一套代码基础设施(common_model/entropy_models),
# 但没有时间上下文, 是纯帧内编码。其熵编码采用 4 路检查板分区(compress_prior_4x),
# 比 P 帧的 2 路(compress_prior_2x)压缩率更高、耗时也更长——I 帧一帧只编一次。
# ============================================================================

import torch
from torch import nn
import torch.nn.functional as F


from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8

g_ch_src = 3 * 8 * 8   # 像素域 patch-embed 后通道数: 3 x 8x8 = 192
g_ch_enc_dec = 368     # 编码器/解码器主干通道数


class IntraEncoder(nn.Module):
    """I 帧编码器: 像素 -> 1/8 patch-embed -> 主干卷积 -> stride2 -> y(1/16, N 通道)。"""

    def __init__(self, N):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        """x: 1x3xHxW 当前帧; quant_step: 按 qp 取出的 q_scale_enc 缩放向量。

        pixel_unshuffle(8) 把像素搬到 1/8 网格(192 通道)后走 torch/CUDA 融合路径。"""
        out = F.pixel_unshuffle(x, 8)
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(out, quant_step)

        return self.forward_cuda(out, quant_step)

    def forward_torch(self, out, quant_step):
        """普通路径: 第一块卷积 -> 量化缩放 -> 主干卷积 + stride2 下采样。"""
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)

    def forward_cuda(self, out, quant_step):
        """CUDA 路径: 量化缩放作为 enc_1 的 quant_step 参数在块内融合完成。"""
        out = self.enc_1(out, quant_step=quant_step)
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    """I 帧解码器: y_hat(1/16, N 通道) 上采样回 1/8 -> 主干卷积 -> 1/8 patch 还原为像素。"""

    def __init__(self, N):
        super().__init__()

        # 首块为子像素卷积上采样 2x (1/16 -> 1/8), 其余为 12 个深度卷积块
        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec),
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
        """x: y_hat(1xNxH/16xW/16); 返回 1x3xHxW 的重建帧。"""
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)

        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out

    def forward_cuda(self, x, quant_step):
        """CUDA 路径: 逐块调用以在最后一块融合 quant_step, 减少中间张量读写。"""
        out = self.dec_1[0](x)
        out = self.dec_1[1](out)
        out = self.dec_1[2](out)
        out = self.dec_1[3](out)
        out = self.dec_1[4](out)
        out = self.dec_1[5](out)
        out = self.dec_1[6](out)
        out = self.dec_1[7](out)
        out = self.dec_1[8](out)
        out = self.dec_1[9](out)
        out = self.dec_1[10](out)
        out = self.dec_1[11](out)
        out = self.dec_1[12](out, quant_step=quant_step)
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


class DMCI(CompressionModel):
    """DCVC-RT 的 I 帧模型: 单隐变量 + 超先验 + 4 路检查板熵编码的 learned image codec。

    对应论文中的 intra codec。同一模型用 q_scale_enc/q_scale_dec 两组按 qp 索引的
    缩放向量覆盖 64 个码率点(模块库式码率控制)。"""

    def __init__(self, N=256, z_channel=128):
        super().__init__(z_channel=z_channel)

        self.enc = IntraEncoder(N)

        # 超先验编码器: y(1/16) -> z(1/64)
        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        # 超先验解码器: z(1/64) -> y 的分辨率
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        # 先验融合: 由 z_hat 得到 y 的先验参数(q_enc/q_dec, scales, means)
        # 输出 N*2+2 通道 = q(2) + scales(N) + means(N)
        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        # 4 路检查板上下文模型的配套模块:
        # reduction 把先验参数降维, adaptor_1/2/3 在每一轮解码后融合 y_hat_so_far
        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        # 空间先验: 由 [y_hat_so_far, common_params] 估计下一轮的 (scales, means)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2, 1),
        )

        self.dec = IntraDecoder(N)

        # 模块库式码率控制(vector bank): 编码/解码各一组按 qp 索引的缩放向量
        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))

    def compress(self, x, qp):
        """编码一帧 I 帧。x 为 padding 后的图像, qp 为量化参数。

        返回 {'bit_stream': 码流字节, 'x_hat': 重建帧}。流程:
        1. IntraEncoder -> y; HyperEncoder -> z, 量化为 int8 的 z_hat
        2. hyper_dec + y_prior_fusion -> y 的先验参数
        3. compress_prior_4x: 4 轮检查板熵编码得到 y_hat 与 4 组待编码符号
        4. IntraDecoder 重建 x_hat(GPU) 与 rANS 算术编码(CPU, 独立 stream)并行
        """
        device = x.device
        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        # y 需 pad 到 4 的倍数(超先验两次 stride2 后 z 为 1/64, 要求 y 为 1/16 整数格)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = \
            self.compress_prior_4x(
                y, params, self.y_spatial_prior_reduction,
                self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        # 记录事件: 4 组符号就绪后, CPU 上的 rANS 编码才开始(与 GPU 重建并行)
        cuda_event = torch.cuda.Event()
        cuda_event.record()
        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            cuda_event.wait()
            self.entropy_coder.reset()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.gaussian_encoder.encode_y(y_q_w_2, s_w_2)
            self.gaussian_encoder.encode_y(y_q_w_3, s_w_3)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, sps, qp):
        """解码一帧 I 帧。sps 携带原始宽高与 ec_part 标志, qp 来自码流头。

        返回 {'x_hat': 重建帧}。流程: 算术解码 z -> 超先验先验参数 ->
        decompress_prior_4x 四轮检查板算术解码 y_hat -> IntraDecoder 重建。"""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        # z 在 1/64 分辨率, y 在 1/16 分辨率(均由 sps 宽高直接算出形状)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = self.get_downsampled_shape(sps['height'], sps['width'], 16)
        self.bit_estimator_z.decode_z(z_size, qp)
        z_q = self.bit_estimator_z.get_z(z_size, device, dtype)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = params[:, :, :y_height, :y_width].contiguous()
        y_hat = self.decompress_prior_4x(params, self.y_spatial_prior_reduction,
                                         self.y_spatial_prior_adaptor_1,
                                         self.y_spatial_prior_adaptor_2,
                                         self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)
        return {"x_hat": x_hat}
