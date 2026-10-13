# ============================================================================
# 熵模型与熵编码的 Python 封装:
#   - EntropyCoder: 对 C++ rANS 算术编解码扩展(MLCodec_extensions_cpp)的薄封装
#   - Bitparm/BitEstimator: z 的(按 qp 分组的)factorized prior 及其 CDF 表构建
#   - GaussianEncoder: y 的拉普拉斯式高斯先验(尺度查表量化), 编解码符号/CDF 的映射
# 编码符号用 int16 打包: 高 8 位是 int8 的量化符号, 低 8 位是 uint8 的尺度档位索引,
# 使 C++ 端一次遍历即可同时取得符号值和对应 CDF。
# ============================================================================

import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from ..layers.cuda_inference import build_index_dec, build_index_enc, process_with_mask


class EntropyCoder():
    """rANS 算术编解码器封装: 持有一对 C++ 的 RansEncoder/RansDecoder。

    y(高斯先验)与 z(factorized 先验)共用同一个底层编码器, 按 cdf_group_index
    区分两组 CDF 表; add_cdf 要求编解码两侧注册顺序一致并返回相同组号。"""

    def __init__(self):
        super().__init__()

        from MLCodec_extensions_cpp import RansEncoder, RansDecoder
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        """把概率质量函数量化成 16bit 定点 CDF(C++ 实现), 供 rANS 使用。"""
        from MLCodec_extensions_cpp import pmf_to_quantized_cdf as _pmf_to_cdf
        cdf = _pmf_to_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        """为一批 pmf 统一构建 CDF 表: 每行长度为 max_length+2,

        首尾额外两格是分布两端的尾部累积概率(tail mass)。"""
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        """编码一帧前重置编码器缓冲。"""
        self.encoder.reset()

    def add_cdf(self, cdf, cdf_length, offset):
        """向编解码器注册一组 CDF 表, 返回组号(两侧必须一致, 用 assert 保证)。"""
        enc_cdf_idx = self.encoder.add_cdf(cdf, cdf_length, offset)
        dec_cdf_idx = self.decoder.add_cdf(cdf, cdf_length, offset)
        assert enc_cdf_idx == dec_cdf_idx
        return enc_cdf_idx

    def encode_y(self, symbols, cdf_group_index):
        """编码 y 的符号。symbols: int16, 高 8 位为待编码 int8 符号, 低 8 位为 CDF 行号。"""
        # symbols: int16, high 8 bits: int8 symbol to be encoded; low 8 bits: uint8 index to use
        assert symbols.dtype == torch.int16
        self.encoder.encode_y(symbols.cpu().numpy(), cdf_group_index)

    def encode_z(self, symbols, cdf_group_index, start_offset, per_channel_size):
        """编码 z 的符号: 直接按 int8 逐通道编码(每个通道一行 CDF, 每行覆盖整幅图)。"""
        self.encoder.encode_z(symbols.to(torch.int8).cpu().numpy(),
                              cdf_group_index, start_offset, per_channel_size)

    def flush(self):
        """flush 输出缓冲, 完成当前帧的码流。"""
        self.encoder.flush()

    def get_encoded_stream(self):
        """取回编码好的码流(bytes)。"""
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        """解码前载入码流。"""
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_y(self, indexes, cdf_group_index):
        """按每符号的 CDF 行号批量触发 y 的算术解码(结果稍后用 get_decoded_tensor 取)。"""
        self.decoder.decode_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)

    def decode_and_get_y(self, indexes, cdf_group_index, device, dtype):
        """decode_y + get_decoded_tensor 的合并版, 直接返回 GPU 张量。"""
        rv = self.decoder.decode_and_get_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)
        rv = torch.as_tensor(rv)
        return rv.to(device).to(dtype)

    def decode_z(self, total_size, cdf_group_index, start_offset, per_channel_size):
        """触发 z 的算术解码: total_size 为 z 的元素总数, per_channel_size 为单通道元素数。"""
        self.decoder.decode_z(total_size, cdf_group_index, start_offset, per_channel_size)

    def get_decoded_tensor(self, device, dtype, non_blocking=False):
        """取回解码结果并搬到 device(non_blocking=True 配合 pinned memory 异步搬运)。"""
        rv = self.decoder.get_decoded_tensor()
        rv = torch.as_tensor(rv)
        return rv.to(device, non_blocking=non_blocking).to(dtype)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        """启用/关闭双熵编码器: 大分辨率时用两条 C++ 线程并行做算术编解码。"""
        self.encoder.set_use_two_encoders(use_two_entropy_coders)
        self.decoder.set_use_two_decoders(use_two_entropy_coders)


class Bitparm(nn.Module):
    """z 的 factorized prior 单层变换(按 qp 分组的 1x1 卷积):

    x -> x*softplus(h) + b (+ tanh(x)*tanh(a) 非线性, 末层无)。"""

    def __init__(self, qp_num, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        else:
            self.a = None

    def forward(self, x, index):
        """index: 本帧的 qp; 用 index_select 从参数库里取出该 qp 对应的 h/b/a。"""
        h = torch.index_select(self.h, 0, index)
        b = torch.index_select(self.b, 0, index)
        x = x * F.softplus(h) + b
        if self.final:
            return x

        a = torch.index_select(self.a, 0, index)
        return x + torch.tanh(x) * torch.tanh(a)


class AEHelper():
    """编解码辅助基类: 暂存构建好的量化 CDF 表(cdf/长度/偏移), 供注册给 C++ 编解码器。"""

    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self.cdf_group_index = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class BitEstimator(AEHelper, nn.Module):
    """z 的熵模型: 4 层 Bitparm 组成的 factorized prior(每个 qp 一套参数 —— 模块库)。

    update() 在推理前离线构建所有 qp、所有通道的 CDF 表并注册给 rANS。"""

    def __init__(self, qp_num, channel):
        super().__init__()
        self.f1 = Bitparm(qp_num, channel)
        self.f2 = Bitparm(qp_num, channel)
        self.f3 = Bitparm(qp_num, channel)
        self.f4 = Bitparm(qp_num, channel, True)
        self.qp_num = qp_num
        self.channel = channel

    def forward(self, x, index):
        """返回 CDF 值 sigmoid(logits)。"""
        return self.get_cdf(x, index)

    def get_logits_cdf(self, x, index):
        """4 层变换得到 logits。"""
        x = self.f1(x, index)
        x = self.f2(x, index)
        x = self.f3(x, index)
        x = self.f4(x, index)
        return x

    def get_cdf(self, x, index):
        return torch.sigmoid(self.get_logits_cdf(x, index))

    def update(self, entropy_coder):
        """离线枚举符号值采样 CDF: 先二分确定每个 (qp, 通道) 的有效取值范围

        [minima, maxima], 再逐符号差分得 pmf, 量化成 CDF 表注册给 C++ 编解码器。"""
        self.entropy_coder = entropy_coder

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
            index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

            minima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) - i
                probs = self.forward(samples, index)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) + i
                probs = self.forward(samples, index)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, None, None, :] + pmf_start

            half = float(0.5)

            lower = self.forward(samples - half, index)
            upper = self.forward(samples + half, index)
            pmf = upper - lower

            pmf = pmf[:, :, 0, :]
            upper = self.forward(maxima.to(torch.float32), index)
            tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

            pmf = pmf.reshape([-1, max_length])
            tail_mass = tail_mass.reshape([-1, 1])
            pmf_length = pmf_length.reshape([-1])
            offset = offset.reshape([-1])
            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)
            self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def build_indexes(self, size, qp):
        """构造 z 编/解码时每个元素对应的 CDF 行号: 行号 = 通道号 + qp*通道数。"""
        B, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1) + qp * self.channel
        return indexes.repeat(B, 1, H, W)

    def encode_z(self, x, qp):
        """编码 z: 每个通道用一行 CDF, 每行连续覆盖 H*W 个符号。"""
        _, _, H, W = x.size()
        return self.entropy_coder.encode_z(x.reshape(-1), self.cdf_group_index, qp * self.channel,
                                           H * W)

    def decode_z(self, size, qp):
        """解码 z(只触发, 结果由 get_z 取回): size 为 (z_h, z_w)。"""
        self.entropy_coder.decode_z(self.channel * size[0] * size[1], self.cdf_group_index,
                                    qp * self.channel, size[0] * size[1])

    def get_z(self, size, device, dtype):
        """取回解码出的 z 并整形成 (1, C, z_h, z_w)。"""
        output_size = (1, self.channel, size[0], size[1])
        val = self.entropy_coder.get_decoded_tensor(device, dtype, non_blocking=True)
        return val.reshape(output_size)


class GaussianEncoder(AEHelper):
    """y 的高斯先验熵编码器: 尺度 sigma 在对数域均匀量化为 128 档(scale_level),

    每档一张以 0 为中心、[-pmf_center, pmf_center] 为支撑集的 CDF 表;
    编码时按符号值+尺度档位查表, 解码时只按尺度档位(符号由解码器输出)。
    尺度低于 force_zero_thres 的符号被强制置零跳过编码(工程加速选项)。"""

    def __init__(self):
        super().__init__()
        self.scale_min = 0.11
        self.scale_max = 16.0
        self.scale_level = 128  # <= 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        # 对数域均匀量化的换算常数
        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)
        self.log_step_recip = 1. / self.log_scale_step

        self.force_zero_thres = None
        # 解码端索引/掩码缓存(按元素个数缓存, 避免重复构造)
        self.decode_index_cache = {}
        self.decode_zeros_cache = {}

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        """在对数域 linspace 后 exp 回来, 得到 128 个尺度档位。"""
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    def update(self, entropy_coder, force_zero_thres=None):
        """离线构建 128 张标准正态 CDF 表并注册给 C++ 编解码器。

        先确定每档的支撑半径 pmf_center(截断到尾部概率 <1e-4), 再差分采样 pmf。"""
        self.entropy_coder = entropy_coder
        self.force_zero_thres = force_zero_thres

        pmf_center = torch.zeros_like(self.scale_table) + 8
        scales = torch.zeros_like(pmf_center) + self.scale_table
        cdf_distribution = torch.distributions.normal.Normal(0., scales)
        for i in range(8, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        cdf_distribution = torch.distributions.normal.Normal(0., scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)
        self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def process_with_mask(self, y, scales, means, mask):
        """量化加窗(委托 CUDA/PyTorch 实现): (y-means)*mask 取整为符号 y_q。"""
        return process_with_mask(y, scales, means, mask, self.force_zero_thres)

    def build_indexes_decoder(self, scales):
        """解码端: 由 scales 求每个元素的 CDF 行号(uint8)。

        force_zero_thres 启用时返回 skip_cond(尺度大于阈值的才解码),
        并只把需要解码的行号压缩进一维序列, 减少传给 C++ 的数据量。"""
        scales = scales.reshape(-1)
        indexes, skip_cond = build_index_dec(scales, self.scale_min, self.scale_max,
                                             self.log_scale_min, self.log_step_recip,
                                             self.force_zero_thres)
        if self.force_zero_thres is not None:
            indexes = indexes[skip_cond]
        return indexes, skip_cond

    def build_indexes_encoder(self, symbols, scales):
        """编码端: 把 int8 符号(高 8 位)与尺度行号(低 8 位)打包成 int16 序列,

        同样跳过被 force_zero_thres 置零的元素。"""
        symbols = symbols.reshape(-1)
        scales = scales.reshape(-1)
        symbols = build_index_enc(symbols, scales, self.scale_min, self.scale_max,
                                  self.log_scale_min, self.log_step_recip, self.force_zero_thres)
        return symbols

    def encode_y(self, x, scales):
        """编码一轮 y: x 为稠密符号张量, scales 为对应尺度(同形状)。"""
        symbols = self.build_indexes_encoder(x, scales)
        return self.entropy_coder.encode_y(symbols, self.cdf_group_index)

    def get_decode_index_cache(self, num, device):
        """缓存 [0, num) 的连续整数序列, 用于解码后把压缩序列散回全图位置。"""
        if num not in self.decode_index_cache:
            c = torch.arange(0, num, dtype=torch.int32, device=device)
            self.decode_index_cache[num] = c

        return self.decode_index_cache[num]

    def get_decode_zeros_cache(self, num, device):
        """缓存全零索引序列(clone 后使用), 配合 masked_scatter_ 恢复稀疏解码结果。"""
        if num not in self.decode_zeros_cache:
            c = torch.zeros(num, dtype=torch.int32, device=device)
            self.decode_zeros_cache[num] = c

        return self.decode_zeros_cache[num].clone()

    def decode_and_get_y(self, scales, dtype, device):
        """一轮"查行号 -> 解码 -> 取回"的合并接口, 直接返回 y_q。"""
        indexes, skip_cond = self.build_indexes_decoder(scales)
        self.decode_y(indexes)
        return self.get_y(scales.shape, scales.numel(), dtype, device, skip_cond, indexes)

    def decode_y(self, indexes):
        """触发一轮解码(结果稍后取回, 便于与其他步骤重叠)。"""
        self.entropy_coder.decode_y(indexes, self.cdf_group_index)

    def get_y(self, shape, numel, dtype, device, skip_cond, indexes):
        """取回一轮解码结果并整形成 shape。

        force_zero_thres 启用时: 解码只覆盖非跳过位置, 这里用 masked_scatter_
        把符号散回全位置、跳过位置置零。"""
        if len(indexes) == 0:
            return torch.zeros(shape, dtype=dtype, device=device)
        if skip_cond is not None:
            curr_index = self.get_decode_index_cache(numel, device)
            back_index = self.get_decode_zeros_cache(numel, device)
            back_index.masked_scatter_(skip_cond, curr_index)
        val = self.entropy_coder.get_decoded_tensor(device, dtype, non_blocking=True)
        if skip_cond is not None:
            y = torch.index_select(val, 0, back_index) * skip_cond
            return y.reshape(shape)
        return val.reshape(shape)
