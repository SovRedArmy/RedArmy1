# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ============================================================================
# DCVC-RT 的 P 帧条件编码模型（DMC = Deep (video) Model for Compression）。
# 论文: "Towards Practical Real-Time Neural Video Compression" (CVPR 2025, arXiv:2502.20762)
#
# 核心设计（对应论文 Sec.3/Sec.4）:
#   1) 隐式时间建模: 不做显式的运动估计/运动补偿(MEMC)，仅用一个 FeatureExtractor
#      从参考帧特征中提取时间上下文 ctx/ctx_t，与当前帧隐变量在通道维拼接后联合编码。
#   2) 单一低分辨率隐变量: 全部变换在 1/8 分辨率上进行（pixel_unshuffle(x, 8) 直接
#      patch-embed 到 1/8 尺度），避免渐进式下采样带来的多次访存（操作开销）。
#   3) 模块库式码率控制: q_encoder/q_decoder/q_feature/q_recon 四组按 qp 索引的
#      可学习缩放向量 + 64+extra_qp 套超先验(BitEstimator)参数，单模型覆盖宽码率范围。
# ============================================================================

import torch
import torch.nn.functional as F
from torch import nn

from .common_model import CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8, \
    bias_pixel_shuffle_8, bias_quant


# 分层质量的 qp 偏移: 每 8 帧为一个组，组内偶数帧偏移 0、奇数帧按 1/8/0/2/0/2/0/2 轮转
# 施加额外 qp（对应论文中的 qp 偏移 [0,8,0,4,0,4,0,4] 思路，用于分层质量/参考刷新）。
qp_shift = [0, 8, 4]
# 因为 shift_qp 会把 qp 抬高最多 extra_qp，所以所有按 qp 索引的参数库都要多准备 extra_qp 组。
extra_qp = max(qp_shift)

# ---- 通道数全局配置 ----
g_ch_src_d = 3 * 8 * 8   # 像素域 patch-embed 后的通道数: 3 通道 x 8x8 patch = 192
g_ch_recon = 320         # 重建生成网络(ReconGeneration)主干通道数
g_ch_y = 128             # 主隐变量 y 的通道数（1/8 分辨率）
g_ch_z = 128             # 超先验隐变量 z 的通道数（1/64 分辨率）
g_ch_d = 256             # 1/8 分辨率特征/上下文的通道数


class FeatureExtractor(nn.Module):
    """参考帧特征提取器（隐式时间建模的核心模块，对应论文 Fig.4 中的 frame/feature extractor）。

    从上一帧解码得到的 1/8 分辨率特征中提取:
      - ctx:   给 Encoder/Decoder 用的空间时间上下文（通道维与 y 拼接后联合编码）
      - ctx_t: 给 TemporalPriorEncoder 用的上下文，用于生成 y 的高斯先验参数
    被拆成 forward_part1/forward_part2 两部分，是为了在解码端让 z 的算术解码
    与 conv2 的推理并行（论文中的并行解码流水线）。
    """

    def __init__(self):
        super().__init__()
        # conv1: 输出 x1 与量化缩放后的 ctx_t（作为时间先验的输入）
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        # conv2: 由 x1 得到编码/解码共用的上下文 ctx
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x, quant):
        """整体前向: x 为参考帧特征(1/8), quant 为按 qp 取出的 q_feature 缩放向量。

        返回 (ctx, ctx_t)。"""
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        """第一步: 得到 x1 及其缩放后的 ctx_t（时间上下文，仅供先验参数生成用）。"""
        x1 = self.conv1(x)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        """第二步: 由 x1 得到空间时间上下文 ctx（与 y 拼接后送入 Encoder/Decoder）。"""
        ctx = self.conv2(x1)
        return ctx


class Encoder(nn.Module):
    """当前帧的编码器: 把像素 patch-embed 后与时间上下文 ctx 融合，下采样到 1/8 得到 y。"""

    def __init__(self):
        super().__init__()
        # 1x1 卷积将 192 通道的像素 patch 投影到 256 通道
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        # 与 ctx (256 通道) 拼接后融合
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        # stride=2 下采样: 特征从 1/8 网格降到 1/16 网格, 得到待编码隐变量 y(128 通道)。
        # 整个编码过程只有这一个隐变量(单一低分辨率隐表征, 无渐进式下采样),
        # z 再由超先验两次 stride2 降到 1/64。Decoder 用 SubpelConv2x 上采样回 1/8 与 ctx 对齐。
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        # 标记 conv1 是否已被折叠进 conv2[0] 的 adaptor（CUDA 融合路径只做一次）
        self.fuse_conv1_flag = False

    def forward(self, x, ctx, quant_step):
        """x: 当前帧(1x3xHxW), ctx: 时间上下文(1x256xH/8xW/8), quant_step: q_encoder 缩放向量。

        先 pixel_unshuffle(8) 把像素搬到 1/8 网格（192 通道），再走 torch 或 CUDA 融合路径。"""
        feature = F.pixel_unshuffle(x, 8)
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(feature, ctx, quant_step)
        return self.forward_cuda(feature, ctx, quant_step)

    def forward_torch(self, feature, ctx, quant_step):
        """普通 PyTorch 路径: conv1 投影 -> 与 ctx 拼接融合 -> 量化缩放 -> stride2 下采样。"""
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature

    def forward_cuda(self, feature, ctx, quant_step):
        """CUDA 融合路径: 首次调用时用矩阵乘把 1x1 的 conv1 折叠进 conv2[0].adaptor,
        消除一次 kernel launch 与中间张量的显存读写（降低论文所说的"操作开销"）。"""
        if not self.fuse_conv1_flag:
            # W_fused = W_adaptor[:, :g_ch_d] @ W_conv1 （两个 1x1 卷积可以合并）
            fuse_weight1 = torch.matmul(
                self.conv2[0].adaptor.weight[:, :g_ch_d, 0, 0],
                self.conv1.weight[:, :, 0, 0]
            )[:, :, None, None]
            fuse_weight2 = self.conv2[0].adaptor.weight[:, g_ch_d:]
            # 偏置同理合并: b_adaptor += W_adaptor @ b_conv1
            self.conv2[0].adaptor.bias.data = self.conv2[0].adaptor.bias + \
                torch.matmul(self.conv2[0].adaptor.weight[:, :g_ch_d, 0, 0],
                             self.conv1.bias[:, None])[:, 0]
            self.conv2[0].adaptor.weight.data = torch.cat([fuse_weight1, fuse_weight2], dim=1)
            self.fuse_conv1_flag = True

        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature, quant_step=quant_step)
        feature = self.down(feature)
        return feature


class Decoder(nn.Module):
    """隐变量解码器: y_hat (1/16) 上采样回 1/8，与时间上下文 ctx 融合得到解码侧特征 feature。

    该 feature 同时是 (a) ReconGeneration 的输入、(b) 下一帧 FeatureExtractor 的参考特征。"""

    def __init__(self):
        super().__init__()
        # 子像素卷积上采样 2x: y(128ch, 1/16) -> feature(256ch, 1/8)
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step,):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, ctx, quant_step)

        return self.forward_cuda(x, ctx, quant_step)

    def forward_torch(self, x, ctx, quant_step):
        """普通路径: 上采样 -> 与 ctx 拼接融合 -> 1x1 卷积 + 量化缩放。"""
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature

    def forward_cuda(self, x, ctx, quant_step):
        """CUDA 融合路径: 上采样时顺带拼接 ctx, 最后的 1x1 卷积 + bias + 量化缩放
        由 bias_quant 单个 kernel 完成。"""
        feature = self.up(x, to_cat=ctx, cat_at_front=False)
        feature = self.conv1(feature)
        feature = F.conv2d(feature, self.conv2.weight)
        feature = bias_quant(feature, self.conv2.bias, quant_step)
        return feature


class ReconGeneration(nn.Module):
    """重建生成网络: 解码特征(1/8) -> 像素域重建帧。

    结构上就是 Encoder 的对称侧: 主干卷积 + 1x1 head 输出 192 通道 + pixel_shuffle(8)
    还原到全分辨率，最后 clamp 到 [0,1]。CUDA 路径把 bias/pixel_shuffle/clamp 融合成
    bias_pixel_shuffle_8 一个 kernel。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x, quant_step):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)

        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out

    def forward_cuda(self, x, quant_step):
        out = self.conv[0](x)
        out = self.conv[1](out)
        out = self.conv[2](out)
        out = self.conv[3](out, quant_step=quant_step)
        out = F.conv2d(out, self.head.weight)
        return bias_pixel_shuffle_8(out, self.head.bias)


class HyperEncoder(nn.Module):
    """超先验编码器: y(1/16,128ch) -> z(1/64,128ch), z 描述 y 逐通道的分布(尺度/均值)。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    """超先验解码器: z_hat(1/64) 上采样回 y 的分辨率，输出分层先验参数（与时间先验融合前）。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    """先验融合: 把分层先验(hyper)与时间先验(temporal)两路参数在通道维拼接后融合，

    输出 3*g_ch_y 通道 = (q_dec, scales, means) 三组高斯先验参数（两步熵编码的第一步）。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    """空间先验(检查板上下文模型): 输入 [y_hat(已解码部分), common_params] 拼接,

    输出 (scales, means)（各 g_ch_y 通道），用于两步熵编码中第二步的分布估计。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class RefFrame():
    """参考帧容器（解码画面缓冲 DPB 中的元素）。

    feature: 上一帧 Decoder 输出的 1/8 特征（P 帧编码/解码的常规参考，速度最快）
    frame:   上一帧的重建像素（仅当需要 "ada_i" 重置时才从 feature 重建出来）
    poc:     图像顺序号"""

    def __init__(self):
        self.frame = None
        self.feature = None
        self.poc = None


class DMC(CompressionModel):
    """DCVC-RT 的 P 帧视频压缩模型（条件编码框架）。"""

    def __init__(self):
        super().__init__(z_channel=g_ch_z, extra_qp=extra_qp)
        self.qp_shift = qp_shift

        # feature_adaptor_i: 从重建像素重新生成参考特征（重置时用, 对应 I 帧式适配）
        # feature_adaptor_p: 常规路径, 对上一帧 Decoder 输出的特征做 1x1 适配
        self.feature_adaptor_i = DepthConvBlock(g_ch_src_d, g_ch_d)
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.feature_extractor = FeatureExtractor()

        self.encoder = Encoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        # 时间先验编码器: 由 ctx_t 直接产出 2*g_ch_y 通道的时间先验参数(stride2 到 y 分辨率)
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior = SpatialPrior()
        self.decoder = Decoder()
        self.recon_generation_net = ReconGeneration()

        # ---- 模块库式码率控制（vector bank）: 4 组按 qp 索引的每通道缩放向量 ----
        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_recon, 1, 1)))

        # 解码画面缓冲（本实现 max_dpb_size=1, 低延迟单参考帧）
        self.dpb = []
        self.max_dpb_size = 1
        self.curr_poc = 0

    def reset_ref_feature(self):
        """丢弃 DPB 中的参考特征，强制下一帧从重建像素(frame)重新提取特征。

        对应码流中 use_ada_i 标志: 周期性把参考从"特征域"重置回"像素域",
        防止参考特征上累积的量化误差漂移。"""
        if len(self.dpb) > 0:
            self.dpb[0].feature = None

    def add_ref_frame(self, feature=None, frame=None, increase_poc=True):
        """把新的参考（特征或像素）插入 DPB 头部，超出容量则弹出最旧的一帧。"""
        ref_frame = RefFrame()
        ref_frame.poc = self.curr_poc
        ref_frame.frame = frame
        ref_frame.feature = feature
        if len(self.dpb) >= self.max_dpb_size:
            self.dpb.pop(-1)
        self.dpb.insert(0, ref_frame)
        if increase_poc:
            self.curr_poc += 1

    def clear_dpb(self):
        """清空 DPB（遇到新的 I 帧时调用）。"""
        self.dpb.clear()

    def set_curr_poc(self, poc):
        """设置当前 POC（编解码两侧保持一致）。"""
        self.curr_poc = poc

    def apply_feature_adaptor(self):
        """根据 DPB 中参考的状态选择适配器:
        - 常规(有 feature): feature_adaptor_p 处理上一帧 Decoder 输出的特征;
        - 重置(只有 frame): feature_adaptor_i 从重建像素 patch-embed 重新生成特征。"""
        if self.dpb[0].feature is None:
            return self.feature_adaptor_i(F.pixel_unshuffle(self.dpb[0].frame, 8))
        return self.feature_adaptor_p(self.dpb[0].feature)

    def res_prior_param_decoder(self, z_hat, ctx_t):
        """由 z_hat(分层先验) 与 ctx_t(时间先验) 融合出 y 的先验参数 params。

        因为输入经过 pad_for_y, 这里把超先验解码结果裁剪到时间先验的尺寸再拼接。"""
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        """解码侧重建: y_hat -> feature(1/8, 存入 DPB 作下一帧参考) -> x_hat(像素)。"""
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def prepare_feature_adaptor_i(self, last_qp):
        """周期性重置(ada_i): 若 DPB 中只有特征没有像素，则先用 ReconGeneration
        从特征重建出像素帧，再 reset_ref_feature 丢弃特征。之后 apply_feature_adaptor
        将走 feature_adaptor_i 路径（像素域重激活，抑制漂移）。"""
        if self.dpb[0].frame is None:
            q_recon = self.q_recon[last_qp:last_qp+1, :, :, :]
            self.dpb[0].frame = self.recon_generation_net(self.dpb[0].feature, q_recon).clamp_(0, 1)
            self.reset_ref_feature()

    def compress(self, x, qp):
        """编码一帧 P 帧。x 为 padding 后的当前帧, qp 为本帧量化参数。

        返回 {'bit_stream': 熵编码后的字节流}。流程:
        1. 按 qp 取出 4 组缩放向量
        2. 参考特征 -> FeatureExtractor -> (ctx, ctx_t)
        3. Encoder(x, ctx) -> y
        4. HyperEncoder(y) -> z, 量化为 int8 的 z_hat
        5. z_hat+ctx_t -> 先验参数 -> 两轮检查板熵编码得到 y_hat 与待编码符号
        6. 算术编码在独立 CUDA stream 上与 Decoder 推理并行
        7. 把解码特征加入 DPB 作为下一帧参考（编码端无需真正重建像素, 更快）
        """
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        device = x.device
        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]

        feature = self.apply_feature_adaptor()
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        y = self.encoder(x, ctx, q_encoder)

        # y 需 pad 到 64 的倍数, 因为超先验两次 stride2 下采样后 z 是 1/64 分辨率
        hyper_inp = self.pad_for_y(y)

        z = self.hyper_encoder(hyper_inp)
        z_hat, z_hat_write = round_and_to_int8(z)
        # 用 CUDA event 让 z 就绪后即可开始 z 的算术编码（与后续 GPU 推理重叠）
        cuda_event_z_ready = torch.cuda.Event()
        cuda_event_z_ready.record()
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat = \
            self.compress_prior_2x(y, params, self.y_spatial_prior)

        cuda_event_y_ready = torch.cuda.Event()
        cuda_event_y_ready.record()
        feature = self.decoder(y_hat, ctx, q_decoder)

        # 算术编码跑在低优先级 stream: 先编 z(等 z event), 再编 y 的两轮(等 y event),
        # 与主 stream 上的 Decoder/ReconGeneration 推理并行（论文: 12% 编码加速）
        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            self.entropy_coder.reset()
            cuda_event_z_ready.wait()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            cuda_event_y_ready.wait()
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        self.add_ref_frame(feature, None)
        return {
            'bit_stream': bit_stream,
        }

    def decompress(self, bit_stream, sps, qp):
        """解码一帧 P 帧。bit_stream 为熵编码字节流, sps 携带分辨率/ec_part/use_ada_i。

        返回 {'x_hat': 重建帧}。流程（含论文中的并行流水线）:
        1. 先算术解码 z（与 FeatureExtractor.forward_part1 的 GPU 推理并行）
        2. z_hat + ctx_t -> 先验参数 -> 解码第一轮 y（与 forward_part2 并行）
        3. 解码第二轮 y 得到 y_hat
        4. Decoder -> feature, ReconGeneration -> x_hat, 二者入 DPB
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon = self.q_recon[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        # z 位于 1/64 分辨率, 由 sps 的宽高直接算出形状
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        self.bit_estimator_z.decode_z(z_size, qp)

        # part1(ctx_t) 与 z 的算术解码并行执行
        feature = self.apply_feature_adaptor()
        c1, ctx_t = self.feature_extractor.forward_part1(feature, q_feature)

        z_hat = self.bit_estimator_z.get_z(z_size, device, dtype)
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        infos = self.decompress_prior_2x_part1(params)

        ctx = self.feature_extractor.forward_part2(c1)

        # 第二轮 y 的算术解码在独立 stream 上, 与 ctx 的 forward_part2 并行
        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            y_hat = self.decompress_prior_2x_part2(params, self.y_spatial_prior, infos)
            cuda_event = torch.cuda.Event()
            cuda_event.record()

        cuda_event.wait()
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

        self.add_ref_frame(feature, x_hat)
        return {
            'x_hat': x_hat,
        }

    def shift_qp(self, qp, fa_idx):
        """按帧在 8 帧组内的索引 fa_idx 给 qp 施加偏移(分层质量): qp + qp_shift[fa_idx]。"""
        return qp + self.qp_shift[fa_idx]
