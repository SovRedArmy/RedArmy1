# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 压缩模型基类：实现 DCVC-UF 三种模型（DMCI 帧内 / DMC-HTL / DMC-HTS / DMC-LD）
# 共享的全部机制：QP 查表量化、棋盘格（四叉树式）分组熵编码、比特/失真估计等。
# 其中 forward_prior_4x 是论文“单一隐变量 + 四分组尺度一步估计 + 均值四步渐进估计”
# 熵编码机制（Sec.3.3）的参考实现。

import math
import torch
import torch.utils.checkpoint

from torch import nn

from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from ..layers.layers import LowerBound, QuantFunc, mse_weighted_average, get_mse_yuv_rgb
from ..utils.common import generate_str


class CkptModule(nn.Module):
    # 支持梯度检查点（gradient checkpointing）的模块包装类：
    # 训练长序列（如 128 帧）时显存不足，通过重计算换取显存节省。
    # forward() 会根据 use_ckpt 标志决定是否用 checkpoint 包裹 internal_forward()
    def __init__(self):
        super().__init__()
        self.use_ckpt = False

    def internal_forward(self, *args, **kwargs):
        # 子类实现的真正前向逻辑（forward 的检查点入口）
        raise NotImplementedError

    def set_use_ckpt(self, use_ckpt=True):
        # 开/关梯度检查点
        self.use_ckpt = use_ckpt

    def forward(self, *args, **kwargs):
        if self.use_ckpt:
            return torch.utils.checkpoint.checkpoint(self.internal_forward, *args, **kwargs,
                                                     preserve_rng_state=False, use_reentrant=False)
        return self.internal_forward(*args, **kwargs)


class CompressionModel(nn.Module):
    # 所有压缩模型（帧内 DMCI 与帧间 DMC）的公共基类。
    # 负责：z 比特估计器、y 高斯熵编码器、棋盘格掩码缓存、QP 向量查表、
    # 量化/加噪代理、YUV+RGB 混合失真、单次交互熵编码（forward_prior_2x/4x）等。
    def __init__(self, z_channel):
        super().__init__()

        self.z_channel = z_channel  # 超先验潜变量 z 的通道数
        self.entropy_coder = None   # C++ rANS 熵编码器（update() 后可用）
        self.proxy = None           # CUDA 推理代理（compress/decompress 时懒加载）
        self.bit_estimator_z = BitEstimator(self.qp_num(), z_channel)  # z 的全分解先验
        self.gaussian_encoder = GaussianEncoder()  # y 的高斯熵模型

        self.masks = {}  # 棋盘格掩码缓存，避免重复构造

        self.prob_to_bits_factor = -1.0 / math.log(2.0)  # -1/log(2)：概率转比特数

        self.recon_range = 0.5  # 像素值域中心（输入已减去 0.5）

    def _initialize_weights(self):
        # 卷积层权重 xavier 初始化、偏置清零
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

    def _reset_cuda_proxies(self):
        # 权重被重新加载后，清空各子模块缓存的 CUDA 代理参数，避免使用过期权重
        for module in self.modules():
            if hasattr(module, 'proxy'):
                try:
                    module.proxy = None
                except Exception:
                    pass

    def add_cdf_to_state_dict(self, state_dict):
        # 把熵编码所需的 CDF 表（quantized_cdf / cdf_length）一并放入 state_dict，
        # 传给 C++/CUDA 推理代理使用
        keys = ['quantized_cdf', 'cdf_length']
        state_dict.update({'gaussian_encoder.'+key: torch.from_numpy(val)
                           for key, val in zip(keys, self.gaussian_encoder.get_cdf_info())})
        state_dict.update({'bit_estimator_z.'+key: torch.from_numpy(val)
                           for key, val in zip(keys, self.bit_estimator_z.get_cdf_info())})
        return state_dict

    @staticmethod
    def add_noise(x):
        # 训练时的加性均匀噪声（-0.5, 0.5），作为量化误差的代理，使比特估计可导
        noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5)
        return x + noise.detach()

    @staticmethod
    def get_loss_info(rd, loss, pixel_num):
        # 组装用于日志打印的字符串信息（每 batch 各样本的 bpp/mse/loss）
        info = {
            'bpp_y': generate_str(rd['bits_y'] / pixel_num),
            'bpp_z': generate_str(rd['bits_z'] / pixel_num),
            'mse': generate_str(rd['mse']),
            'losses': generate_str(loss['losses']),
        }
        return info

    def get_mse(self, x, x_hat):
        # 计算 YUV 与 RGB 混合失真（0.8 * YUV + 0.2 * RGB），见 layers.mse_weighted_average
        mse_yuv, mse_rgb = get_mse_yuv_rgb(x, x_hat)

        _, _, H, W = x.size()
        pixel_num = H * W
        mse = mse_weighted_average(mse_yuv, mse_rgb, pixel_num)
        return mse

    @staticmethod
    def get_one_mask(micro_mask, H, W):
        # 由 2x2 微观棋盘格模板平铺出整幅 HxW 的布尔掩码
        mask = torch.tensor(micro_mask, dtype=torch.bool)
        mask = mask.repeat((H + 1) // 2, (W + 1) // 2)
        mask = mask[None, None, :H, :W]
        return mask

    @staticmethod
    def get_padding_size(height, width, p=64):
        # 计算把 (height, width) 补齐到 p 的倍数所需的右侧/下方 padding 大小
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    def index_select_dim0(self, x, index):
        # 按 QP 序号从可学习查找表中取向量，并整形为 [B, C, 1, 1] 以便逐通道缩放
        if x is None:
            return x
        out = torch.index_select(x, 0, index)[:, :, None, None]
        return out

    def probs_to_bits(self, probs):
        # 概率转比特数：-log2(p)，并做下界 0 与数值稳定保护
        dtype = probs.dtype
        probs = probs.float()
        bits = torch.log(LowerBound.apply(probs, 1e-6)) * self.prob_to_bits_factor
        bits = LowerBound.apply(bits, 0)
        return bits.to(dtype)

    @staticmethod
    def process_with_mask(y, scales, means, mask):
        # 用棋盘格掩码处理某一分组的量化：
        #   y_res = (y - mu) * mask（残差，仅保留本分组位置）
        #   y_q   = round(y_res)（整数符号）
        #   y_hat = y_q + mu（反量化重建值）
        #   scales_hat = sigma * mask（本分组位置的尺度，用于估比特）
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = QuantFunc.apply(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def qp_num():
        # QP 级数固定为 64：单模型覆盖宽码率范围、支持码率控制（论文 Sec.3.4）
        return 64

    def separate_prior_image(self, params):
        # 帧内模型（DMCI）：公共参数直接拆成 (scales, means) 两组
        scales, means = params.chunk(2, 1)
        return scales, means

    def separate_prior_video(self, params):
        # 视频模型（DMC）：公共参数拆成 (量化步长, scales, means) 三组；
        # quant_step 经下界 0.5 保护后，编码侧乘 1/step（放大以细量化）、
        # 解码侧乘 step（还原），实现 QP 控制的隐变量缩放
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = LowerBound.apply(quant_step, 0.5)
        q_enc = 1. / quant_step
        q_dec = quant_step
        return q_enc, q_dec, scales, means

    def set_entropy_coder_parallel(self, entropy_coder_parallel):
        # 设置 rANS 并行子流数量（论文中的并行熵编码，降低解码串行开销）
        self.entropy_coder.set_entropy_coder_parallel(entropy_coder_parallel)

    def update(self, skip_thres):
        # 测试前的准备：构造 C++ rANS 熵编码器，并为 y（高斯）与 z（全分解）
        # 两个熵模型生成量化 CDF 表。skip_thres 为跳过小编码符号的阈值（提速）
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, skip_thres=skip_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    @torch.no_grad()
    def get_mask_2x(self, B, C, H, W, device):
        # 构造 2 分组棋盘格掩码（LD 模型使用）：
        # 通道维分成两半，分别配 (1,0/0,1) 与 (0,1/1,0) 互补模板，
        # 两组掩码互补、并集为全 1
        curr_mask_str = f'{B}_{C}_{H}_{W}_2x'
        if curr_mask_str not in self.masks:
            assert C % 2 == 0
            m = torch.ones((B, C // 2, H, W), dtype=torch.bool)
            m0 = self.get_one_mask(((1, 0), (0, 1)), H, W)
            m1 = self.get_one_mask(((0, 1), (1, 0)), H, W)

            mask_0 = torch.cat((m * m0, m * m1), dim=1)
            mask_0 = mask_0.to(device=device)
            mask_1 = torch.cat((m * m1, m * m0), dim=1)
            mask_1 = mask_1.to(device=device)

            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @torch.no_grad()
    def get_mask_4x(self, B, C, H, W, device):
        # 构造 4 分组棋盘格掩码（HT/帧内模型使用，对应论文的 quadtree-like 四分组）：
        # 通道维分成四份，每份配 2x2 棋盘格中一个位置（左上/右上/左下/右下），
        # 四组掩码按旋转排列，并集为全 1 —— 保证每个隐变量元素只被编码一次
        curr_mask_str = f'{B}_{C}_{H}_{W}_4x'
        if curr_mask_str not in self.masks:
            assert C % 4 == 0
            m = torch.ones((B, C // 4, H, W), dtype=torch.bool)
            m0 = self.get_one_mask(((1, 0), (0, 0)), H, W)
            m1 = self.get_one_mask(((0, 1), (0, 0)), H, W)
            m2 = self.get_one_mask(((0, 0), (1, 0)), H, W)
            m3 = self.get_one_mask(((0, 0), (0, 1)), H, W)

            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_0 = mask_0.to(device=device)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_1 = mask_1.to(device=device)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_2 = mask_2.to(device=device)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            mask_3 = mask_3.to(device=device)

            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def get_y_bits(self, y, sigma):
        # 用高斯熵模型估计 y 的比特数（训练时用加噪残差近似量化）
        probs = self.gaussian_encoder.get_prob_train(y, sigma)
        return self.probs_to_bits(probs)

    def get_z_bits(self, z, index):
        # 用全分解先验估计 z 的比特数（index 为 QP，先验参数随 QP 变化）
        probs = self.bit_estimator_z.get_prob(z, index)
        return self.probs_to_bits(probs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # 重载权重后清空 CUDA 代理缓存（代理内保存了旧权重的拷贝）
        res = super().load_state_dict(state_dict, *args, **kwargs)
        # CUDA proxy params are cached via .set_param(self.state_dict()) on first use.
        # When weights are reloaded, proxies must be reset to avoid using stale params.
        self._reset_cuda_proxies()
        return res

    def forward_prior_2x(self, y, common_params, y_spatial_prior):
        # 2 分组棋盘格熵编码（LD 模型使用，论文 Sec.3.3 的两遍版本）：
        # 第 1 遍用公共参数的 (scales, means) 编码棋盘格 A；
        # 第 2 遍用 y_spatial_prior(已解码部分, 公共参数) 重新估计 (scales, means) 编码棋盘格 B；
        # y_hat 乘 q_dec 还原量化缩放。此处为训练前向（量化用 STE），
        # 实际码流编解码在 CUDA 代理中按同样流程调用 rANS。
        q_enc, q_dec, scales, means = self.separate_prior_video(common_params)
        y = y * q_enc
        device = common_params.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        means = y_spatial_prior(y_hat_0, common_params)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

        y_res = None if y_res_0 is None else y_res_0 + y_res_1
        y_q = None if y_q_0 is None else y_q_0 + y_q_1
        scales_hat = s_hat_0 + s_hat_1
        return y_res, y_q, y_hat, scales_hat

    def forward_prior_4x(self, y, q_enc, q_dec,
                         common_params, y_spatial_prior_reduction,
                         y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                         y_spatial_prior_adaptor_3, y_spatial_prior,
                         spatial_prior_has_scales=False):
        # ★ 论文核心之一：四分组（quadtree-like）单次交互熵编码（Sec.3.3，Fig.6）★
        # 潜变量 y 按通道被划分成 4 个互补的棋盘格分组（mask_0..3），逐组编码：
        #   - 第 0 组：直接用公共参数（超先验+时间先验融合）给出的 (scales, means)；
        #   - 第 1~3 组：把“已重建的 y_hat_so_far”与公共参数送入
        #     y_spatial_prior_adaptor_i + y_spatial_prior，渐进地重新估计参数。
        # 关键设计（spatial_prior_has_scales 标志）：
        #   - HTS 模型：spatial_prior 只输出 means（scales 一直沿用公共参数一次估计的值）
        #     => 尺度只需在码流交互前估计一次（single interaction），
        #        均值的 4 步渐进估计不依赖码流交互，大幅降低解码开销；
        #   - HTL/LD/帧内 模型：spatial_prior 同时输出 (scales, means)。
        # 训练时用 STE 量化；真实编码时每组编解码后立刻用于下一组的参数估计。
        if q_enc is None:
            # 视频模型路径：从公共参数中解出量化步长与初始 (scales, means)
            q_enc, q_dec, scales, means = self.separate_prior_video(common_params)
            y = y * q_enc
        else:
            # 帧内模型路径：量化步长由 QP 查表给出（q_scale_y_*），公共参数只含 (scales, means)
            spatial_prior_has_scales = True
            scales, means = self.separate_prior_image(common_params)
            y = y * q_enc

        common_params = y_spatial_prior_reduction(common_params)  # 1x1 卷积压缩公共参数通道
        device = common_params.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, device)

        # ---- 第 0 组：使用一次性估计的 (scales, means) ----
        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        # ---- 第 1 组：基于第 0 组重建值渐进估计参数 ----
        y_hat_so_far = y_hat_0
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        else:
            # HTS：只更新均值，尺度沿用一次估计的结果（单次码流交互的关键）
            means = y_spatial_prior(y_spatial_prior_adaptor_1(y_hat_so_far, common_params))
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        # ---- 第 2 组 ----
        y_hat_so_far = y_hat_so_far + y_hat_1
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_2(y_hat_so_far, common_params))
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        # ---- 第 3 组 ----
        y_hat_so_far = y_hat_so_far + y_hat_2
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_3(y_hat_so_far, common_params))
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        # 四组结果叠加（掩码互补，故相加即完整 y_hat），并按 q_dec 还原量化缩放
        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        y_res = None if y_res_0 is None else (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = None if y_q_0 is None else (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        return y_res, y_q, y_hat, scales_hat
