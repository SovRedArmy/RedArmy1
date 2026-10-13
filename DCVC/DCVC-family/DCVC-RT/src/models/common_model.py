# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ============================================================================
# 压缩模型公共基类: 管理 rANS 熵编码器、CUDA stream、mask 缓存,
# 并实现 P 帧的两路检查板熵编码(compress/decompress_prior_2x)
# 与 I 帧的四路检查板熵编码(compress/decompress_prior_4x)。
#
# "两步分布估计"(two-step distribution estimation, 论文引用 [26]):
#   第一轮: 用 (超先验 z_hat + 时间先验 ctx_t) 融合出的参数估计分布, 编码一半系数;
#   第二轮: 用 (已解码的一半 y_hat + 先验参数) 经空间先验网络重新估计分布, 编码另一半。
# 检查板分区通过通道减半 + 空间 2x2/棋盘 pattern 的 mask 实现:
# 每一轮只有一个分区位置上的符号真正进入算术编码, 其余位置乘 mask 置零。
# ============================================================================

import torch
from torch import nn

from ..layers.cuda_inference import combine_for_reading_2x, \
    restore_y_2x, restore_y_2x_with_cat_after, add_and_multiply, \
    replicate_pad, restore_y_4x, clamp_reciprocal_with_quant
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder


class CompressionModel(nn.Module):
    """DMC(P 帧)与 DMCI(I 帧)的公共基类。"""

    def __init__(self, z_channel, extra_qp=0):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        # z 的全零均值 factorized prior: 参数按 (qp_num+extra_qp) 组存放,
        # 解码时按 qp 选择其中一组 —— 论文中的"超先验模块库"。
        self.bit_estimator_z = BitEstimator(64 + extra_qp, z_channel)
        self.gaussian_encoder = GaussianEncoder()

        # 检查板 mask 缓存(按 尺寸+类型 为 key), 避免每帧重建
        self.masks = {}
        # CUDA stream 缓存(按 设备+优先级+序号 为 key), 用于 CPU 熵编码与 GPU 推理并行
        self.cuda_streams = {}

    def get_cuda_stream(self, device, idx=0, priority=0):
        """按 key 取或创建一条 CUDA stream(priority=-1 为低优先级, 供算术编码用)。"""
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]

    @staticmethod
    def get_qp_num():
        """码率点总数: qp 取值 0~63, 共 64 档。"""
        return 64

    @staticmethod
    def get_padding_size(height, width, p=64):
        """计算把 (height, width) 向上补齐到 p 的倍数所需的右侧/底部 padding 量。"""
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        """按 p 倍下采样后的形状: 先补齐到 p 的倍数再除以 p(解码端没有 y 的形状,
        需要从 sps 的原始宽高直接推算 z/y 的形状)。"""
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def update(self, force_zero_thres=None):
        """推理前的一次性初始化: 创建 rANS 熵编码器, 并为 z 的 factorized prior
        和 y 的高斯先验构建量化 CDF 表、注册到 C++ 编解码器。

        force_zero_thres: 工程加速选项(论文未描述)。scales 低于该阈值的符号
        直接置零、不进入算术编码, 用极小的质量损失换取更少的 CPU 编码量。"""
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, force_zero_thres=force_zero_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        """大分辨率(>720p)时启用双熵编码器(ec_part=1): C++ 侧用两条线程并行做
        算术编码/解码, 分摊 CPU 熵编码耗时。该标志会写进 SPS 以便解码端对齐。"""
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)

    def pad_for_y(self, y):
        """把 y 用边缘复制方式 pad 到 4 的倍数: 超先验两次 stride2 后 z 需要 1/64 整数格。"""
        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicate_pad(y, padding_b, padding_r)
        return y_pad

    def separate_prior(self, params, is_video=False):
        """把先验参数按通道拆分。

        I 帧(非 video): 前 2 通道经 sigmoid 映射为编码/解码量化步长, 其余为 (scales, means)。
        P 帧(video): 三等分 (量化步长, scales, means), 量化步长下限 clamp 到 0.5。"""
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp_min(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_encoding(params, y):
        """编码端拆分 P 帧先验参数: 同时把 y 乘上 q_enc(缩放到整数域便于量化),

        返回 (y 已缩放, q_dec, scales, means)。"""
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant(q_dec, y, 0.5)
        return y, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_decoding(params):
        """解码端拆分 P 帧先验参数: 只需 (量化步长, scales, means)。"""
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = torch.clamp_min(quant_step, 0.5)
        return quant_step, scales, means

    def process_with_mask(self, y, scales, means, mask):
        """量化加窗: 残差 (y - means*mask)*mask 四舍五入为整数符号 y_q, 返回

        (y_res, y_q, y_hat = y_q + means*mask, scales*mask), 供高斯熵编码使用。"""
        return self.gaussian_encoder.process_with_mask(y, scales, means, mask)

    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        """由 2x2 微观 pattern 平铺出 (1,1,H,W) 的 mask 张量。"""
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask

    def get_mask_4x(self, batch, channel, height, width, dtype, device):
        """I 帧用的 4 路 mask: 通道四等分, 各份配以 2x2 棋盘的四个互补位置。

        mask_0/1/2/3 对应四个分区; 同一空间位置在每一轮只有一个通道份参与编码。"""
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_4x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
                m2 = self.get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
                m3 = self.get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def get_mask_2x(self, batch, channel, height, width, dtype, device):
        """P 帧用的 2 路 mask: 通道二等分, 两份配以 2x2 对角互补的棋盘位置。"""
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_2x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 1)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (1, 0)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @staticmethod
    def single_part_for_writing_4x(x):
        """4 路压缩: 把 4 个通道份相加成 1 份稠密张量, 只保留 mask=1 位置的有效值。

        算术编码器按行主序扫描稠密符号序列, 0 位置(被跳过/置零)不产生比特。"""
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    @staticmethod
    def single_part_for_writing_2x(x):
        """2 路压缩: 同上, 把 2 个通道份相加成稠密序列供算术编码。"""
        x0, x1 = x.chunk(2, 1)
        return x0 + x1

    def compress_prior_2x(self, y, common_params, y_spatial_prior):
        """P 帧两路检查板熵编码(编码端)。

        第 1 轮: 用 (z_hat+ctx_t 融合出的) 先验参数 + mask_0 量化编码一半系数;
        第 2 轮: 把 [y_hat_0, 先验参数] 送入空间先验网络重新估计 (scales, means),
        用 mask_1 编码另一半。
        返回 4 个稠密化的待编码张量(y 的两轮符号与尺度)与重建 y_hat。"""
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        # y_hat = (y_hat_0 + y_hat_1) * q_dec: 拼回两半并乘回量化步长恢复幅值
        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)

        y_q_w_0 = self.single_part_for_writing_2x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_2x(y_q_1)
        s_w_0 = self.single_part_for_writing_2x(s_hat_0)
        s_w_1 = self.single_part_for_writing_2x(s_hat_1)
        return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat

    def decompress_prior_2x(self, common_params, y_spatial_prior):
        """P 帧两路检查板熵解码(同步版本, 实际解码走 part1/part2 拆分版本)。"""
        infos = self.decompress_prior_2x_part1(common_params)
        y_hat = self.decompress_prior_2x_part2(common_params, y_spatial_prior, infos)
        return y_hat

    def decompress_prior_2x_part1(self, common_params):
        """解码第 1 步(可与 FeatureExtractor.forward_part2 的 GPU 推理并行):

        拆先验参数 -> 稠密化 scales 并查表得 CDF 序号 -> 触发第 1 轮算术解码。
        中间状态打包进 infos 字典供 part2 使用。"""
        q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        scales_r = combine_for_reading_2x(scales, mask_0, inplace=False)
        indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_r)
        self.gaussian_encoder.decode_y(indexes)
        infos = {
            "q_dec": q_dec,
            "mask_0": mask_0,
            "mask_1": mask_1,
            "means": means,
            "scales_r": scales_r,
            "skip_cond": skip_cond,
            "indexes": indexes,
        }
        return infos

    def decompress_prior_2x_part2(self, common_params, y_spatial_prior, infos):
        """解码第 2 步: 取回第 1 轮解码符号 -> 还原 y_hat_0 -> 空间先验估计第 2 轮

        分布 -> 解码另一半 -> 拼回 y_hat = (y_hat_0 + y_hat_1) * q_dec。"""
        dtype = common_params.dtype
        device = common_params.device
        y_q_r = self.gaussian_encoder.get_y(infos["scales_r"].shape,
                                            infos["scales_r"].numel(),
                                            dtype, device,
                                            infos["skip_cond"], infos["indexes"])
        y_hat_0, cat_params = restore_y_2x_with_cat_after(y_q_r, infos["means"], infos["mask_0"],
                                                          common_params)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        scales_r = combine_for_reading_2x(scales, infos["mask_1"], inplace=True)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_1 = restore_y_2x(y_q_r, means, infos["mask_1"])

        y_hat = add_and_multiply(y_hat_0, y_hat_1, infos["q_dec"])
        return y_hat

    def compress_prior_4x(self, y, common_params, y_spatial_prior_reduction,
                          y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                          y_spatial_prior_adaptor_3, y_spatial_prior):
        """I 帧四路检查板熵编码(编码端)。与 2x 版类似, 但分 4 轮:

        每轮解码/编码一个通道份(对应 2x2 棋盘的一个位置), 且下一轮的分布
        以"目前已累积的 y_hat_so_far"为条件重新估计。"""
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        _, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        _, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        y_q_w_0 = self.single_part_for_writing_4x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_4x(y_q_1)
        y_q_w_2 = self.single_part_for_writing_4x(y_q_2)
        y_q_w_3 = self.single_part_for_writing_4x(y_q_3)
        s_w_0 = self.single_part_for_writing_4x(s_hat_0)
        s_w_1 = self.single_part_for_writing_4x(s_hat_1)
        s_w_2 = self.single_part_for_writing_4x(s_hat_2)
        s_w_3 = self.single_part_for_writing_4x(s_hat_3)
        return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat

    def decompress_prior_4x(self, common_params, y_spatial_prior_reduction,
                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                            y_spatial_prior_adaptor_3, y_spatial_prior):
        """I 帧四路检查板熵解码(解码端): 逐轮"估计分布 -> 算术解码一个通道份 ->

        累积到 y_hat_so_far 作为下一轮的条件", 4 轮后乘回量化步长得到 y_hat。"""
        _, quant_step, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        scales_r = self.single_part_for_writing_4x(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat
