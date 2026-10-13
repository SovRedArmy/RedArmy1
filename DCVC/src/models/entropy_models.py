# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 熵模型：y（高斯先验 + 棋盘格分组）与 z（全分解先验）的概率建模与 CDF 构造。
# 对应论文 Sec.3.3 的熵编码机制；实际 rANS 编解码由 C++ 扩展
# MLCodec_extensions_cpp（RansEncoder/RansDecoder）完成，本文件负责
# 在 update() 阶段把连续概率离散化为量化 CDF 表并注入编码器。

import math
import torch

from torch import nn

from ..layers.layers import bit_estimator_z_fwd, bit_estimator_z_prob, LowerBound


MAX_ENTROPY_CODING_VALUE = 8  # 熵编码符号的取值半径：符号范围 [-8, 8]，共 17 个符号


class AEHelper():
    # 熵编码辅助类：缓存量化后的 CDF 表（quantized_cdf 与 cdf_length），
    # 供 C++ rANS 编码器查表使用
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self._quantized_cdf = None
        self._cdf_length = None

    def get_cdf_info(self):
        # 返回缓存的 (量化 CDF, 每条 CDF 的长度)
        return self._quantized_cdf, self._cdf_length

    def set_cdf_info(self, quantized_cdf, cdf_length):
        # 保存 CDF 信息（numpy 形式）
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()


class EntropyCoder():
    # 对 C++ rANS 编码器（MLCodec_extensions_cpp.RansEncoder/RansDecoder）的轻量封装，
    # 提供概率直方图 → 量化 CDF 的转换以及 CDF/并行度的设置
    def __init__(self):
        super().__init__()

        from MLCodec_extensions_cpp import RansEncoder, RansDecoder
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()

    @staticmethod
    def pmf_to_quantized_cdf(pmf):
        # 调用 C++ 实现把概率质量函数 pmf 转为 16bit 精度的量化累积分布函数
        from MLCodec_extensions_cpp import pmf_to_quantized_cdf as _pmf_to_cdf
        cdf = _pmf_to_cdf(pmf.tolist())
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def reorder_prob(prob):
        # 把“按值大小升序排列（关于 0 对称）”的 pmf 重排为“0, 1, -1, 2, -2, ...”的顺序：
        # 小符号排在前面后，编码大概率符号时 CDF 查表更快，从而加速熵编码
        # prob is with length l, with the last one tail_prob
        # the first (l-1) probs are sysmmetric about 0, in increasing order
        # we want to make the first a few probs in 0, 1, -1, 2, -2, ... order
        length = prob.size(0)
        prob1 = prob.clone()
        center = (length - 1) // 2
        prob1[0] = prob[center]
        for i in range(1, center + 1):
            prob1[2 * i - 1] = prob[center + i]
            prob1[2 * i - 0] = prob[center - i]
        return prob1

    def set_cdf(self, cdf, cdf_length, index):
        # 把第 index 类 CDF 表注入 C++ 编码器与解码器（0=z 的表，1=y 的表）
        self.encoder.set_cdf(cdf, cdf_length, index)
        self.decoder.set_cdf(cdf, cdf_length, index)

    def set_entropy_coder_parallel(self, entropy_coder_parallel):
        # 设置熵编码并行子流数（论文的并行 rANS，缓解串行熵编码瓶颈）
        self.encoder.set_entropy_coder_parallel(entropy_coder_parallel)
        self.decoder.set_entropy_coder_parallel(entropy_coder_parallel)

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        # 把一批 pmf（每条长度可变 pmf_length，不足 max_length）加尾部质量 tail_mass
        # （覆盖 |符号|>sym_range 的溢出）后重排并量化，得到用于 rANS 的 CDF 矩阵
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            prob1 = EntropyCoder.reorder_prob(prob)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob1)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf


class BitEstimator(AEHelper, nn.Module):
    # z（超先验潜变量）的熵模型：逐通道、逐 QP 的全分解先验。
    # 参数 h/b/a 形状为 [qp_num, channel, layer_num]，即每个 QP、每个通道
    # 各有一组独立的先验参数 —— 这与论文“64 QP 单模型”的设计一致。
    def __init__(self, qp_num, channel):
        super().__init__()
        self.layer_num = 4  # 全分解先验的复合层数
        self.h = nn.Parameter(
            torch.nn.init.normal_(torch.empty([qp_num, channel, self.layer_num]), 0, 0.01))
        self.b = nn.Parameter(
            torch.nn.init.normal_(torch.empty([qp_num, channel, self.layer_num]), 0, 0.01))
        self.a = nn.Parameter(
            torch.nn.init.normal_(torch.empty([qp_num, channel, self.layer_num - 1]), 0, 0.01))
        self.qp_num = qp_num
        self.channel = channel

    def get_hba(self, index, dtype=None):
        # 按批内每个样本的 QP 取出各自的 (h, b, a) 先验参数
        if isinstance(index, int):
            h = self.h[index:index+1, :, :]
            b = self.b[index:index+1, :, :]
            a = self.a[index:index+1, :, :]
        else:
            h = torch.index_select(self.h, 0, index)
            b = torch.index_select(self.b, 0, index)
            a = torch.index_select(self.a, 0, index)
        if dtype is not None:
            return h.to(dtype), b.to(dtype), a.to(dtype)
        return h, b, a

    def forward(self, x, index):
        # 返回符号 x 的概率（用于 update() 中搜索有效符号范围）
        h, b, a = self.get_hba(index)
        return bit_estimator_z_prob(x, h, b, a)

    def get_prob(self, x, index):
        # 返回符号 x 落在单位区间 [x-0.5, x+0.5] 的概率，用于训练时估计 z 的比特数
        h, b, a = self.get_hba(index, dtype=x.dtype)
        prob = bit_estimator_z_fwd(x, h, b, a)
        return prob

    @torch.inference_mode()
    def update(self, entropy_coder):
        # 测试前调用：为全部 (qp, channel) 组合离散化出量化 CDF 表并注入 rANS 编码器。
        # 先自适应收缩符号范围 sym_range（概率覆盖 0.001~0.999），减少 CDF 长度以加速编码
        self.entropy_coder = entropy_coder

        device = next(self.parameters()).device
        zeros = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
        index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

        sym_range = zeros + MAX_ENTROPY_CODING_VALUE
        for i in range(MAX_ENTROPY_CODING_VALUE, 1, -1):
            neg_probs = self.forward(zeros - i, index)
            pos_probs = self.forward(zeros + i, index)
            sym_range = torch.where(torch.logical_and(neg_probs < 0.001, pos_probs > 0.999),
                                    i, sym_range)

        sym_range = sym_range.int()
        pmf_length = sym_range * 2 + 1

        max_length = MAX_ENTROPY_CODING_VALUE * 2 + 1
        samples = torch.arange(max_length, device=device)
        samples = samples[None, None, None, :] - sym_range

        lower = self.forward(samples - 0.5, index)
        upper = self.forward(samples + 0.5, index)
        pmf = upper - lower

        pmf = pmf[:, :, 0, :]
        upper = self.forward(sym_range.float(), index)
        tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

        pmf = pmf.reshape([-1, max_length])
        tail_mass = tail_mass.reshape([-1, 1])
        pmf_length = pmf_length.reshape([-1])
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        cdf_length = pmf_length + 2
        self.set_cdf_info(quantized_cdf, cdf_length)
        self.entropy_coder.set_cdf(*self.get_cdf_info(), 0)


class GaussianEncoder(AEHelper, nn.Module):
    # y（主潜变量）的熵模型：零均值拉普拉斯式高斯分布，尺度 sigma 被离散到
    # scale_table（128 级、对数均匀）。这样 CDF 表只需按 128 个尺度预生成一次，
    # 推理时按最近尺度查表，避免在线计算 CDF —— 论文超快熵解码的前提之一。
    def __init__(self):
        super().__init__()
        self.scale_min = 0.11   # 尺度下界（也是训练时概率计算的下界保护）
        self.scale_max = 16.0   # 尺度上界
        self.scale_level = 128  # 尺度离散级数
        self.scale_table = self.get_scale_table()

        self.skip_thres = 0  # 符号概率超过该阈值时可被“跳过”（免编码），用于提速

    @staticmethod
    def get_prob_train(values, scales):
        # 训练时计算符号 values 在 N(0, scales) 下落入 [v-0.5, v+0.5] 的概率
        # （用 erfc 表达 CDF，数值稳定），用于比特估计
        def _cdf2(inputs):
            const = float(-(2 ** -0.5))
            return torch.erfc(const * inputs)

        dtype = values.dtype
        values = values.float()
        scales = scales.float()
        scales = LowerBound.apply(scales, 0.11)
        values = torch.abs(values)
        upper = _cdf2((0.5 - values) / scales)
        lower = _cdf2((-0.5 - values) / scales)
        prob = upper - lower
        prob = torch.clamp_min(0.5 * prob, 1e-9)
        return prob.to(dtype)

    def get_scale_table(self):
        # 生成 [0.11, 16] 上对数均匀的 128 级尺度表
        return torch.exp(torch.linspace(math.log(self.scale_min),
                                        math.log(self.scale_max),
                                        self.scale_level))

    def update(self, entropy_coder, skip_thres):
        # 测试前调用：为尺度表中每个 sigma 离散化出 N(0, sigma) 的量化 CDF 表，
        # 注入 rANS 编码器（index=1 号 CDF）；skip_thres 为跳过小编码符号的阈值
        self.entropy_coder = entropy_coder
        self.skip_thres = skip_thres

        zeros = torch.zeros_like(self.scale_table)
        sym_range = zeros + MAX_ENTROPY_CODING_VALUE
        scales = self.scale_table
        cdf_distribution = torch.distributions.normal.Normal(0., scales)
        for i in range(MAX_ENTROPY_CODING_VALUE, 1, -1):
            samples = zeros + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            sym_range = torch.where(probs > 0.999, i, sym_range)

        sym_range = sym_range.int()
        pmf_length = 2 * sym_range + 1
        max_length = 2 * MAX_ENTROPY_CODING_VALUE + 1

        samples = torch.arange(max_length, device=sym_range.device) - sym_range[:, None]
        samples = samples.float()

        scales = self.scale_table[:, None]
        cdf_distribution = torch.distributions.normal.Normal(0., scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2)
        self.entropy_coder.set_cdf(*self.get_cdf_info(), 1)
