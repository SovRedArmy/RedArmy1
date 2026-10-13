# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# DCVC-UF 低延迟（Low-Delay, LD）视频模型 DMC —— 论文中的消融/对照结构：
# chunk 尺寸 N=1（逐帧编码），其余机制（时间记忆传播、2 分组棋盘格熵编码、
# QP 查表缩放）与 HT 模型一致。通道数更小（g_ch_y=128 等），速度极快，
# 用于论文中验证 chunk 式框架收益的对照（Tab.1 的 LD 行）。

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .common_model import CkptModule, CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, QuantFunc, ResidualBlockUpsample, \
    ResidualBlockWithStride2
from ..utils.common import loss_func


g_frame_delay = 1     # LD 模型每次只编 1 帧
g_ch_src_d = 3 * 8 * 8  # 单帧 pixel_unshuffle(8) 后的通道数
g_ch_y = 128          # 潜变量 y 的通道数
g_ch_z = 128          # 超先验 z 的通道数
g_ch_d = 256          # 1/8 特征通道数
g_ch_m = 256          # 记忆 memory 通道数


class Decoder(CkptModule):
    # LD 解码器：y_hat(1/16) 上采样到 1/8，与 ctx（此处直接用 memory）融合出特征
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d + g_ch_m, g_ch_d, dcb2=True),
            DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def internal_forward(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature


class Encoder(CkptModule):
    # LD 编码器：单帧 pixel_unshuffle(8) → 与 memory 融合 → QP 缩放 → 1/16 的 y
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_src_d + g_ch_m, g_ch_d, dcb2=True),
            DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
        )
        self.conv2 = DepthConvBlock(g_ch_d, g_ch_d, dcb2=True)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

    def internal_forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature


class FeatureAdaptorI(CkptModule):
    # 初始记忆适配器：I 帧特征 → 初始 memory M_0
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_src_d, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
        )

        self.proxy = None

    def internal_forward(self, x):
        return self.conv(x)


class FeatureAdaptorM(CkptModule):
    # 记忆更新适配器：M_i = conv(concat(M_{i-1}, F_i))，机制与 HT 模型一致
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_m + g_ch_d, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
        )

    def internal_forward(self, memory, feature):
        return self.conv(torch.cat((memory, feature), dim=1))


class FeatureExtractor(CkptModule):
    # 特征提取器：memory → 编码条件 ctx（LD 中与 memory 同空间，直接拼接使用）
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
        )

        self.proxy = None

    def internal_forward(self, x):
        return self.conv(x)


class HyperDecoder(CkptModule):
    # 超先验解码器：z_hat(1/64) → 1/16 熵参数
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z, dcb2=True, shortcut=False),
            ResidualBlockUpsample(g_ch_z, g_ch_z, dcb2=True, shortcut=False),
            DepthConvBlock(g_ch_z, g_ch_y, dcb2=True),
        )

        self.proxy = None

    def internal_forward(self, x):
        return self.conv(x)


class HyperEncoder(CkptModule):
    # 超先验编码器：y(1/16) → z(1/64)
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z, dcb2=True),
            ResidualBlockWithStride2(g_ch_z, g_ch_z, dcb2=True, shortcut=False),
            ResidualBlockWithStride2(g_ch_z, g_ch_z, dcb2=True, shortcut=False),
        )

    def internal_forward(self, x):
        return self.conv(x)


class PriorFusion(CkptModule):
    # 先验融合：[超先验参数, 时间先验参数(QP 缩放后)] → 公共熵参数（3×g_ch_y 通道：
    # quant_step / scales / means），供 forward_prior_2x 使用
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3, dcb2=True),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3, dcb2=True),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3, dcb2=True),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def internal_forward(self, x1, x2, quant):
        return self.conv(torch.cat((x1, x2 * quant), dim=1))


class ReconHead(CkptModule):
    # 单帧重建头：F_i → 全分辨率重建帧；for_reset=True 时输出 192 通道伪帧特征
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
        )
        self.head = nn.Conv2d(g_ch_d, g_ch_src_d, 1)

    def internal_forward(self, x, for_reset=False):
        out = self.conv(x)
        out = self.head(out)
        if for_reset:
            return out
        return F.pixel_shuffle(out, 8)


class SpatialPrior(CkptModule):
    # 空间先验网络（2 分组棋盘格第二遍使用）：
    # 输入 [公共参数, 第 1 遍 y_hat]，输出第 2 遍专用的 (scales, means)
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 2, dcb2=True),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2, dcb2=True),
            nn.Conv2d(g_ch_y * 2, g_ch_y * 1, 1),
        )

    def internal_forward(self, x1, x2):
        return self.conv(torch.cat((x1, x2), dim=1))


class TemporalPriorEncoder(CkptModule):
    # 时间先验编码器：memory(1/8) → 1/16 时间先验参数
    def __init__(self):
        super().__init__()
        self.conv = ResidualBlockWithStride2(g_ch_m, g_ch_y * 2, dcb2=True, shortcut=False)

    def internal_forward(self, x):
        return self.conv(x)


class DMC(CompressionModel):
    # LD 视频模型主体：与 HT 的 DMC 接口一致（forward/compress/decompress），
    # 熵编码使用 2 分组棋盘格（forward_prior_2x）
    def __init__(self):
        super().__init__(z_channel=g_ch_z)

        self.feature_adaptor_i = FeatureAdaptorI()   # 初始记忆适配
        self.feature_adaptor_m = FeatureAdaptorM()   # 记忆更新
        self.feature_extractor = FeatureExtractor()  # memory → ctx

        self.encoder = Encoder()              # 帧编码器
        self.hyper_encoder = HyperEncoder()   # 超先验编码器

        self.hyper_decoder = HyperDecoder()            # 超先验解码器
        self.temporal_prior_encoder = TemporalPriorEncoder()  # 时间先验编码器
        self.y_prior_fusion = PriorFusion()            # 先验融合
        self.y_spatial_prior = SpatialPrior()          # 空间先验（2 分组）
        self.decoder = Decoder()                       # 帧解码器

        self.recon_head = ReconHead()  # 单帧重建头

        # QP 查找表（64 级）：编码 / 解码 / 时间先验三组逐通道缩放
        self.q_encoder = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self.q_decoder = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self.q_feature = nn.Parameter(torch.ones((self.qp_num(), g_ch_y * 2)))
        self._initialize_weights()

        self.ref_feature = None  # 上一帧特征 F_i（或 I 帧 patchify 特征）
        self.memory = None       # 时间记忆 M_i
        self.ctx = None          # 编码条件

    def apply_feature_adaptor(self):
        # 每帧编码前更新记忆并导出 ctx（与 HT 模型机制相同）
        if self.memory is None:
            self.memory = self.feature_adaptor_i(self.ref_feature)
        else:
            self.memory = self.feature_adaptor_m(self.memory, self.ref_feature)
        self.ctx = self.feature_extractor(self.memory)

    def clear_dpb(self):
        # 清空跨帧状态（新序列 / I 帧后）
        self.ref_feature = None
        self.memory = None
        self.ctx = None

    def get_rd_info(self, result, fa_idx):
        # LD 的失真按帧位置周期加权（与 HT 的 mse_8frames_sum 权重对应）：
        # index_map = [0,1,0,2,0,2,0,2]，dist_weights = [1.5(第7帧), 0.16, 0.4]
        dist_weights = [0.52, 1.33, 0.83]
        dist_weight = dist_weights[fa_idx]
        rd = {
            'bits_y': result['bits_y'],
            'bits_z': result['bits_z'],
            'bpp': result['bpp'],
            'mse': result['mse'] * dist_weight,
        }
        return rd

    def get_recon_and_feature(self, y_hat, ctx, q_decoder):
        # 帧解码：y_hat → F_i → 重建帧
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_head(feature)
        return x_hat, feature

    def res_prior_param_decoder(self, z_hat, memory, q_feature):
        # 熵参数估计：时间先验 + 超先验 → 融合
        temporal_params = self.temporal_prior_encoder(memory)
        hyper_params = self.hyper_decoder(z_hat)
        params = self.y_prior_fusion(hyper_params, temporal_params, q_feature)
        return params

    def set_use_ckpt(self, use_ckpt=True):
        # 批量开/关梯度检查点
        self.feature_adaptor_i.set_use_ckpt(use_ckpt)
        self.feature_adaptor_m.set_use_ckpt(use_ckpt)
        self.feature_extractor.set_use_ckpt(use_ckpt)
        self.temporal_prior_encoder.set_use_ckpt(use_ckpt)
        self.encoder.set_use_ckpt(use_ckpt)
        self.decoder.set_use_ckpt(use_ckpt)
        self.recon_head.set_use_ckpt(use_ckpt)
        self.hyper_encoder.set_use_ckpt(use_ckpt)
        self.hyper_decoder.set_use_ckpt(use_ckpt)
        self.y_prior_fusion.set_use_ckpt(use_ckpt)
        self.y_spatial_prior.set_use_ckpt(use_ckpt)

    def set_ref_feature(self, feature, reset_feature_memory):
        # 保存 F_i 供下一帧使用；reset_feature_memory=True 时把 F_i 映回伪帧特征
        # 并清空记忆（周期性 feature reset，抑制误差漂移）
        self.ref_feature = feature
        if reset_feature_memory:
            feature = self.recon_head(feature, for_reset=True)
            self.clear_dpb()
            self.ref_feature = feature

    def add_ref_feature_from_frame(self, frame, apply_feature_adaptor=True):
        # 由 I 帧重建帧建立初始参考特征；测试时经 CUDA 代理 DMCLDProxy 在 GPU 端完成
        if self.training:
            self.ref_feature = F.pixel_unshuffle(frame, 8)
            return
        if self.proxy is None:
            try:
                from inference_extensions_cuda import DMCLDProxy
            except Exception:
                raise NotImplementedError(
                    'cannot import cuda implementation for inference. '
                    'Please build the inference extensions first.'
                )
            state_dict = self.state_dict()
            state_dict = self.add_cdf_to_state_dict(state_dict)
            self.proxy = DMCLDProxy()
            self.proxy.set_param(state_dict, self.gaussian_encoder.skip_thres)
        return self.proxy.add_ref_feature_from_frame(frame, apply_feature_adaptor)

    def compress(self, x, qp, reset_feature_memory, padding_b, padding_r):
        # 测试压缩入口：CUDA 代理完成整条前向 + rANS 编码
        bit_stream, ec_parallel = self.proxy.compress(
            x, qp, reset_feature_memory, padding_b, padding_r)
        return {
            'bit_stream': bit_stream.tobytes(),
            'ec_parallel': ec_parallel,
        }

    def decompress(self, bit_stream, sps, qp, ec_part, reset_feature_memory):
        # 测试解压入口：CUDA 代理从码流解码出重建帧
        x_hat = self.proxy.decompress((np.frombuffer(bit_stream, dtype=np.uint8)),
                                      qp, sps['height'], sps['width'], ec_part,
                                      reset_feature_memory)

        return {
            'x_hat': x_hat,
        }

    def forward_one_frame(self, x, qp, reset_feature_memory=False):
        # 单帧训练前向：ctx 更新 → 编码 → 超先验 → 2 分组棋盘格熵编码 →
        # 解码重建 → F_i 传播 → 比特/失真估计（与 HT 的 forward_one_frame 同构）
        q_encoder = self.index_select_dim0(self.q_encoder, qp)
        q_decoder = self.index_select_dim0(self.q_decoder, qp)
        q_feature = self.index_select_dim0(self.q_feature, qp)
        self.apply_feature_adaptor()

        y = self.encoder(x, self.ctx, q_encoder)

        z = self.hyper_encoder(y)
        z_hat = QuantFunc.apply(z)

        params = self.res_prior_param_decoder(z_hat, self.memory, q_feature)
        y_res, y_q, y_hat, scales_hat = self.forward_prior_2x(y, params, self.y_spatial_prior)

        x_hat, feature = self.get_recon_and_feature(y_hat, self.ctx, q_decoder)

        self.set_ref_feature(feature, reset_feature_memory)

        y_for_bit = self.add_noise(y_res)
        z_for_bit = self.add_noise(z)
        bits_y = self.get_y_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, qp)

        mse = self.get_mse(x, x_hat)
        bits_y = torch.sum(bits_y, dim=(1, 2, 3))
        bits_z = torch.sum(bits_z, dim=(1, 2, 3))
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp = (bits_y + bits_z) / pixel_num

        return {'bits_y': bits_y,
                'bits_z': bits_z,
                'bpp': bpp,
                'mse': mse,
                'x_hat': x_hat,
                }

    def forward(self, x, qp, lambdas=None, get_loss_info=False, curr_poc=0):
        # 训练入口：x 为单帧 Tensor 或帧列表（逐帧调用 forward_one_frame）。
        # index_map 按帧在 8 帧周期中的位置取失真权重（见 get_rd_info）
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        if not isinstance(x, list):
            fa_idx = index_map[curr_poc % 8]
            result = self.forward_one_frame(x, qp)
            rd = self.get_rd_info(result, fa_idx)
            loss = loss_func(rd, lambdas)
            self.ref_feature = self.ref_feature.detach()
            self.memory = self.memory.detach()
            info = None
            if get_loss_info:
                _, _, H, W = x.size()
                pixel_num = H * W
                info = self.get_loss_info(rd, loss, pixel_num)
            return loss['loss'], info

        frame_nums = len(x)
        losses = []
        for frame_index in range(frame_nums):
            cur_frame = x[frame_index]
            fa_idx = index_map[(frame_index + 1) % 8]
            result = self.forward_one_frame(cur_frame, qp)
            rd = self.get_rd_info(result, fa_idx)
            loss = loss_func(rd, lambdas)
            losses.append(loss['loss'])
        info = None
        if get_loss_info:
            _, _, H, W = x[0].size()
            pixel_num = H * W
            info = self.get_loss_info(rd, loss, pixel_num)
        loss = torch.mean(torch.stack(losses))
        return loss, info
