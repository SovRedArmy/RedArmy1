# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ★ DCVC-UF 高吞吐视频模型 DMC（论文主模型，Sec.3）★
# chunk 式编码框架：一次把 8 帧（g_frame_delay）作为一个 chunk 联合编码为
# 一个紧凑潜变量 y，再由 8 个帧专属解码头并行重建 —— 论文 Fig.3 的实现。
# 两种结构（ModelStructure）：
#   - HTL（htl, 大模型）：标准通道配置；
#   - HTS（hts, 小模型）：更浅的 DepthConvBlock 堆叠 + dcb2 半通道块，更快；
# HTS 的熵编码采用“尺度一步估计”的精简机制（见 forward_prior_4x 与
# SpatialPriorAdaptor 的 is_hts 分支），即论文 Sec.3.3 的单次码流交互设计。
# 时间上下文通过 feature memory（M_i）跨 chunk 传播（temporal chunk context），
# 训练可扩展到 128 帧长序列而无需周期刷新。

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .common_model import CkptModule, CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, QuantFunc, ResidualBlockUpsample, \
    ResidualBlockWithStride2, mse_8frames_sum
from ..utils.common import ModelStructure, loss_func


g_frame_delay = 8            # 一个 chunk 包含的帧数 N=8
g_ch_src_d_intra = 3 * 8 * 8  # 单帧 pixel_unshuffle(8) 后的通道数（192）
g_ch_src_d = g_ch_src_d_intra * g_frame_delay  # 8 帧拼接后的通道数（1536）
g_ch_y = 256                 # 潜变量 y 的通道数
g_ch_z = 128                 # 超先验 z 的通道数
g_ch_d = 512                 # chunk 编/解码器与 1/8 特征 F_i 的通道数
g_ch_m = 512                 # 时间记忆 memory M_i 的通道数
g_ch_recon = 256             # 帧专属解码头的中间通道数


class Decoder(CkptModule):
    # chunk 解码器：把 1/16 的 y_hat 上采样到 1/8，并与时间上下文 ctx 融合，
    # 得到共享特征 F_i（供 8 个帧专属头并行重建，也作为下一 chunk 的传播特征）
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:  # HTS：7 个 dcb2（半宽）块，更快
            self.up = SubpelConv2x(g_ch_y, g_ch_d, 1)
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_d * 2, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            )
        else:
            self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1, force_bias=True)
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_d * 2, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
            )

    def internal_forward(self, x, ctx, quant_step):
        # y_hat(1/16) 上采样 → 与 ctx 拼接做跨帧时空建模 → QP 缩放 → F_i(1/8)
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = feature * quant_step
        return feature


class Encoder(CkptModule):
    # chunk 编码器：8 帧 pixel_unshuffle(8) 到 1/8 → 与时间上下文 ctx 拼接 →
    # 深度卷积跨帧交互 → QP 缩放 → 步长 2 卷积得 1/16 的单紧凑潜变量 y
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_src_d + g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            )
        else:
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_src_d + g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
            )
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

    def internal_forward(self, x, ctx, quant_step):
        # x: [B, 8*3, H, W]（8 帧按通道拼接）→ 1/8 patchify → 与 ctx 融合 → y(1/16)
        feature = F.pixel_unshuffle(x, 8)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = feature * quant_step
        feature = self.down(feature)
        return feature


class FeatureAdaptorI(CkptModule):
    # 初始特征适配器：序列开始时（或记忆重置后）把 I 帧 pixel_unshuffle(8) 的
    # 192 通道特征投影到 memory 空间（g_ch_m=512），得到初始 M_0
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_src_d_intra, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            )
        else:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_src_d_intra, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
            )

    def internal_forward(self, x):
        return self.conv(x)


class FeatureAdaptorM(CkptModule):
    # 记忆更新适配器（论文 Eq.4 时间 chunk 上下文传播）：
    # M_i = conv(concat(M_{i-1}, F_i)) —— 把上一 chunk 的记忆与当前 chunk 的
    # 解码特征 F_i 融合，实现长期时间相关性的高效建模（无需重传历史潜变量）
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m + g_ch_d, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            )
        else:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m + g_ch_d, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
            )

    def internal_forward(self, memory, feature):
        # 融合旧记忆与新特征：M_i = conv(concat(M_{i-1}, F_i))
        return self.conv(torch.cat((memory, feature), dim=1))


class FeatureExtractor(CkptModule):
    # 特征提取器：从 memory M_i 中导出编码条件/时间先验上下文 ctx，
    # 分别送入 chunk 编码器（作为条件）与时间先验编码器（估计熵参数）
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            )
        else:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
            )

    def internal_forward(self, x):
        return self.conv(x)


class HyperDecoder(CkptModule):
    # 超先验解码器：1/64 的 z_hat 两次上采样回 1/16，输出超先验熵参数
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            kwargs = {"shortcut": False}
        else:
            kwargs = {"force_bias": True}
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_y, **kwargs),
            ResidualBlockUpsample(g_ch_y, g_ch_y, **kwargs),
            DepthConvBlock(g_ch_y, g_ch_y),
        )

    def internal_forward(self, x):
        return self.conv(x)


class HyperEncoder(CkptModule):
    # 超先验编码器：1/16 的 y 两次下采样到 1/64 得 z
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            kwargs = {"shortcut": False}
        else:
            kwargs = {}
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_y),
            ResidualBlockWithStride2(g_ch_y, g_ch_y, **kwargs),
            ResidualBlockWithStride2(g_ch_y, g_ch_z, **kwargs),
        )

    def internal_forward(self, x):
        return self.conv(x)


class PriorFusion(CkptModule):
    # 先验融合网络（论文 Fig.3 的 parameter estimation）：
    # 拼接 [超先验参数, 时间先验参数] → 精炼得到 768 通道公共熵参数
    # （HT 模型下 768 = 3×256，对应 quant_step / scales / means 三组各 256 通道）
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def internal_forward(self, x1, x2):
        # x1: 超先验参数；x2: 时间先验参数
        return self.conv(torch.cat((x1, x2), dim=1))


class ReconHead(CkptModule):
    # ★ 帧专属解码头（frame-specific decoders，论文 Sec.3.2 / Fig.3 右侧）★
    # 把共享特征 F_i 送入 8 个并行的独立解码头，各重建 1 帧 —— MoE 式设计，
    # 在保证单帧重建质量的同时维持 chunk 级吞吐（8 帧同时出）。
    # HTS 结构额外共享每 2 帧一个的 conv1，减小参数量。
    # for_reset=True 时把 F_i 映射回 192 通道“伪帧”用于重置记忆（feature reset）
    def __init__(self, is_hts):
        super().__init__()
        self.is_hts = is_hts
        if is_hts:
            self.conv1 = nn.ModuleList()
            for _ in range(g_frame_delay // 2):
                self.conv1.append(
                    nn.Sequential(
                        DepthConvBlock(g_ch_d, g_ch_d),
                    )
                )

            self.conv2 = nn.ModuleList()
            for _ in range(g_frame_delay):
                self.conv2.append(
                    nn.Sequential(
                        DepthConvBlock(g_ch_d, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        nn.Conv2d(g_ch_recon, g_ch_src_d_intra, 1),
                    )
                )
        else:
            self.conv = nn.ModuleList()
            for _ in range(g_frame_delay):
                self.conv.append(
                    nn.Sequential(
                        DepthConvBlock(g_ch_d, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        nn.Conv2d(g_ch_recon, g_ch_src_d_intra, 1),
                    )
                )

    def internal_forward(self, x, for_reset=False):
        # 正常模式：8 个帧专属头各自从 F_i 重建一帧（pixel_shuffle(8) 回全分辨率）
        if self.is_hts:
            if for_reset:
                # 记忆重置路径：用最后一组头把 F_i 映回 192 通道伪帧特征
                out = self.conv1[-1](x)
                return self.conv2[-1](out)

            all_out = []
            for i in range(g_frame_delay):
                if i % 2 == 0:
                    common = self.conv1[i // 2](x)  # 每 2 帧共享一个前置块（HTS）
                out = self.conv2[i](common)          # 帧专属头（MoE 式并行）
                out = F.pixel_shuffle(out, 8)
                all_out.append(out)
            return all_out
        else:
            if for_reset:
                return self.conv[-1](x)

            all_out = []
            for i in range(g_frame_delay):
                out = self.conv[i](x)
                out = F.pixel_shuffle(out, 8)
                all_out.append(out)
            return all_out


class SpatialPrior(CkptModule):
    # 渐进空间先验网络（棋盘格 4 分组间使用）：
    # HTS 输出 256 通道（只出 means）；HTL 输出 512 通道（scales+means 各 256）
    def __init__(self, is_hts):
        super().__init__()
        out_ch = g_ch_y * 1 if is_hts else g_ch_y * 2
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            nn.Conv2d(g_ch_y * 2, out_ch, 1),
        )

    def internal_forward(self, x):
        return self.conv(x)


class SpatialPriorAdaptor(DepthConvBlock):
    # 空间先验适配器：为 4 组渐进编码各配一个入口适配块。
    # is_hts=True 时 forward 接受 (y_hat_so_far, common_params) 两个输入并拼接；
    # is_hts=False（HTL）时输入已在外部拼接，这里只过单输入块
    def __init__(self, in_ch, out_ch, *, dcb2=False, shortcut=False, force_adaptor=True,
                 is_hts=True):
        assert force_adaptor
        super().__init__(in_ch, out_ch, dcb2=dcb2, shortcut=shortcut, force_adaptor=force_adaptor)
        self.is_hts = is_hts

    def forward(self, x1, x2=None):
        if self.is_hts:
            return super().forward(torch.cat((x1, x2), dim=1))
        else:
            return super().forward(x1)


class TemporalPriorEncoder(CkptModule):
    # 时间先验编码器：把 1/8 的 memory M_i 下采样到 1/16 并映射为时间先验参数，
    # 与超先验参数在 PriorFusion 中融合。量化前先按 QP 缩放（x * quant）
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            kwargs = {"shortcut": False}
        else:
            kwargs = {}
        self.conv = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2, **kwargs)

    def internal_forward(self, x, quant):
        return self.conv(x * quant)


class DMC(CompressionModel):
    # DCVC-UF 高吞吐视频模型主体：组装上述所有子模块，
    # forward*() 供训练，compress/decompress() 经 CUDA 代理供测试。
    # 状态（跨 chunk 传播）：ref_feature（上一 chunk 的 F_i / I 帧特征）、
    # memory（长期记忆 M_i）、ctx（当前时间上下文）
    def __init__(self, model_structure=ModelStructure.HTS):
        super().__init__(z_channel=g_ch_z)

        self.is_hts = model_structure == ModelStructure.HTS

        self.feature_adaptor_i = FeatureAdaptorI(self.is_hts)   # 初始记忆适配
        self.feature_adaptor_m = FeatureAdaptorM(self.is_hts)   # 记忆更新
        self.feature_extractor = FeatureExtractor(self.is_hts)  # 记忆 → ctx

        self.encoder = Encoder(self.is_hts)              # chunk 编码器
        self.hyper_encoder = HyperEncoder(self.is_hts)   # 超先验编码器

        self.hyper_decoder = HyperDecoder(self.is_hts)            # 超先验解码器
        self.temporal_prior_encoder = TemporalPriorEncoder(self.is_hts)  # 时间先验编码器
        self.y_prior_fusion = PriorFusion()                        # 先验融合
        self.y_spatial_prior_reduction = nn.Conv2d(g_ch_y * 3, g_ch_y, 1)  # 公共参数压缩
        self.y_spatial_prior_adaptor_1 = SpatialPriorAdaptor(     # 第 1 组适配器
            g_ch_y * 2, g_ch_y * 2, is_hts=self.is_hts)
        self.y_spatial_prior_adaptor_2 = SpatialPriorAdaptor(     # 第 2 组适配器
            g_ch_y * 2, g_ch_y * 2, is_hts=self.is_hts)
        self.y_spatial_prior_adaptor_3 = SpatialPriorAdaptor(     # 第 3 组适配器
            g_ch_y * 2, g_ch_y * 2, is_hts=self.is_hts)
        self.y_spatial_prior = SpatialPrior(self.is_hts)  # 渐进空间先验
        self.decoder = Decoder(self.is_hts)               # chunk 解码器

        self.recon_head = ReconHead(self.is_hts)          # 8 个帧专属解码头

        # QP 查找表（64 级）：编码/解码/特征三处各自可学习的逐通道缩放向量
        self.q_encoder = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self.q_decoder = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self.q_feature = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self._initialize_weights()

        self.ref_feature = None  # 上一 chunk 传播来的特征 F_i（或 I 帧 patchify 特征）
        self.memory = None       # 长期时间记忆 M_i
        self.ctx = None          # 当前时间上下文（编码条件 + 时间先验来源）

    def apply_feature_adaptor(self):
        # 每编一个 chunk 前调用：更新记忆并导出 ctx。
        # 无记忆（序列开始/重置后）用 FeatureAdaptorI 初始化，否则用 FeatureAdaptorM 融合
        if self.memory is None:
            self.memory = self.feature_adaptor_i(self.ref_feature)
        else:
            self.memory = self.feature_adaptor_m(self.memory, self.ref_feature)
        self.ctx = self.feature_extractor(self.memory)

    def clear_dpb(self):
        # 清空解码图片缓冲（新序列开始 / I 帧后重置状态）
        self.ref_feature = None
        self.memory = None
        self.ctx = None

    def get_rd_info(self, res):
        # 把 8 帧各自的 MSE 按帧位置权重聚合成 chunk 级失真（见 mse_8frames_sum）
        dist_weights = [1.5, 0.16, 0.4]
        assert g_frame_delay == 8
        mse = mse_8frames_sum(res['mse'], dist_weights)

        rd = {
            'bits_y': res['bits_y'],
            'bits_z': res['bits_z'],
            'bpp': res['bpp'],
            'mse': mse,
        }
        return rd

    def get_recon_and_feature(self, y_hat, ctx, q_decoder):
        # chunk 解码：y_hat → F_i（共享特征）→ 8 帧重建
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_head(feature)
        return x_hat, feature

    def res_prior_param_decoder(self, z_hat, memory, q_feature):
        # 熵参数估计（论文 Fig.3 中 parameter estimation 的参考实现）：
        # 时间先验（来自 memory）+ 超先验（来自 z_hat）→ 融合为公共熵参数
        temporal_params = self.temporal_prior_encoder(memory, q_feature)
        hyper_params = self.hyper_decoder(z_hat)
        params = self.y_prior_fusion(hyper_params, temporal_params)
        return params

    def set_use_ckpt(self, use_ckpt=True):
        # 批量开/关所有子模块的梯度检查点
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
        # chunk 编码完成后调用：保存 F_i 供下一个 chunk 使用；
        # reset_feature_memory=True 时（周期性 feature reset，默认每 32 chunk），
        # 先用 recon_head(for_reset=True) 把 F_i 映射回“伪帧”特征再清空记忆，
        # 以抑制长序列误差漂移（论文 Sec.3.2 末尾的记忆重置机制）
        self.ref_feature = feature
        if reset_feature_memory:
            feature = self.recon_head(feature, for_reset=True)
            self.clear_dpb()
            self.ref_feature = feature

    def add_ref_feature_from_frame(self, frame, apply_feature_adaptor=True):
        # 由像素帧（通常是 I 帧重建帧）建立初始参考特征：
        # 训练时直接 pixel_unshuffle；测试时交给 CUDA 代理（DMCHTSProxy/DMCHTLProxy）
        # 在 GPU 端完成，避免 CPU-GPU 往返。首次调用时懒加载代理并注入权重+CDF
        if self.training:
            self.ref_feature = F.pixel_unshuffle(frame, 8)
            return
        if self.proxy is None:
            try:
                if self.is_hts:
                    from inference_extensions_cuda import DMCHTSProxy
                    proxy_cls = DMCHTSProxy
                else:
                    from inference_extensions_cuda import DMCHTLProxy
                    proxy_cls = DMCHTLProxy
            except Exception:
                raise NotImplementedError(
                    'cannot import cuda implementation for inference. '
                    'Please build the inference extensions first.'
                )
            state_dict = self.state_dict()
            state_dict = self.add_cdf_to_state_dict(state_dict)
            self.proxy = proxy_cls()
            self.proxy.set_param(state_dict, self.gaussian_encoder.skip_thres)
        return self.proxy.add_ref_feature_from_frame(frame, apply_feature_adaptor)

    def compress(self, x, qp, reset_feature_memory, padding_b, padding_r):
        # 测试压缩入口：整条 chunk 前向 + 量化 + rANS 熵编码全部在 CUDA 代理内完成，
        # 返回字节码流与熵编码并行子流数 ec_parallel
        bit_stream, ec_parallel = self.proxy.compress(
            x, qp, reset_feature_memory, padding_b, padding_r)
        return {
            'bit_stream': bit_stream.tobytes(),
            'ec_parallel': ec_parallel,
        }

    def decompress(self, bit_stream, sps, qp, ec_part, reset_feature_memory):
        # 测试解压入口：CUDA 代理从码流解码 8 帧重建（列表）
        x_hat = self.proxy.decompress((np.frombuffer(bit_stream, dtype=np.uint8)),
                                      qp, sps['height'], sps['width'], ec_part,
                                      reset_feature_memory)
        return {
            'x_hat': x_hat,
        }

    def forward_one_frame(self, x, qp, reset_feature_memory=False):
        # 单个 chunk 的训练前向（论文 Fig.3 完整数据流）：
        # ctx 更新 → chunk 编码 → 超先验 → 熵参数 → 四分组渐进熵编码 →
        # chunk 解码 → 8 帧重建 + F_i 传播 → 比特/失真估计
        q_encoder = self.index_select_dim0(self.q_encoder, qp)  # QP 查表缩放向量
        q_decoder = self.index_select_dim0(self.q_decoder, qp)
        q_feature = self.index_select_dim0(self.q_feature, qp)
        self.apply_feature_adaptor()  # M_i 更新 + ctx 导出

        y = self.encoder(x, self.ctx, q_encoder)  # 8 帧 + ctx → 紧凑潜变量 y(1/16)

        z = self.hyper_encoder(y)       # 超先验 z(1/64)
        z_hat = QuantFunc.apply(z)      # 取整（STE）

        params = self.res_prior_param_decoder(z_hat, self.memory, q_feature)  # 公共熵参数
        y_res, y_q, y_hat, scales_hat = self.forward_prior_4x(  # 四分组熵编码（含渐进均值）
            y, None, None, params, self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            spatial_prior_has_scales=not self.is_hts)  # HTS: 尺度一步估计（仅更新均值）

        x_hat, feature = self.get_recon_and_feature(y_hat, self.ctx, q_decoder)  # 解码 8 帧

        self.set_ref_feature(feature, reset_feature_memory)  # F_i 传播 / 记忆重置

        # ---- 训练用比特/失真估计（加噪代理量化）----
        y_for_bit = self.add_noise(y_res)
        z_for_bit = self.add_noise(z)
        bits_y = self.get_y_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, qp)

        x_split = x.chunk(g_frame_delay, 1)  # 把 8 帧拆开逐帧算 MSE
        mse_list = []
        for x_one, x_hat_one in zip(x_split, x_hat):
            mse = self.get_mse(x_one, x_hat_one)
            mse_list.append(mse)
        bits_y = torch.sum(bits_y, dim=(1, 2, 3))
        bits_z = torch.sum(bits_z, dim=(1, 2, 3))
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp = (bits_y + bits_z) / pixel_num

        return {
            'bits_y': bits_y,
            'bits_z': bits_z,
            'bpp': bpp,
            'mse': mse_list,
            'x_hat': x_hat,
        }

    def forward(self, x, qp, lambdas=None, get_loss_info=False, curr_poc=0):
        # 训练入口。x 可以是单个 chunk（Tensor）或 chunk 列表（多 chunk 级联训练），
        # 逐 chunk 调 forward_one_frame 并累加 R-D 损失；
        # 每个 chunk 后 detach 参考特征以控制反向传播长度（级联训练策略）
        if not isinstance(x, list):
            result = self.forward_one_frame(x, qp)
            rd = self.get_rd_info(result)
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
            result = self.forward_one_frame(cur_frame, qp)
            rd = self.get_rd_info(result)
            loss = loss_func(rd, lambdas)
            losses.append(loss['loss'])
        info = None
        if get_loss_info:
            _, _, H, W = x[0].size()
            pixel_num = H * W
            info = self.get_loss_info(rd, loss, pixel_num)
        loss = torch.mean(torch.stack(losses))
        return loss, info
