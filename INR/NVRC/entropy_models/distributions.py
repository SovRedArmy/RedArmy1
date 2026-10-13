
"""
条件高斯分布 + rANS 算术编解码（基于 CompressAI 封装）：
    GaussianConditional 把 (symbols, means, scales) 映射为似然（训练）或
    通过 build_indexes -> quantized_cdf 表进行 rANS 编解码（实际压缩）。
    为支持自回归模型的逐步扫描，提供 encode_buffer_* / decode_buffer_* 两阶段接口：
    先缓存所有符号，最后一次 encode，解码端则从流中逐步 decode。
"""
from .utils import *


class InnerGaussianConditional(compressai.entropy_models.GaussianConditional):
    """（CompressAI 原生条件高斯模型的直接复用）"""
    pass


class GaussianConditional(nn.Module):
    """
    - See https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/entropy_models/entropy_models.py
    （对 CompressAI GaussianConditional 的缓冲式封装，支持自回归逐步编码/解码）
    """
    def __init__(self, scale_table=None, *args, **kwargs):
        nn.Module.__init__(self)
        self.inner_module = InnerGaussianConditional(scale_table, *args, **kwargs)
        self.encoder = compressai.ans.BufferedRansEncoder()
        self.decoder = compressai.ans.RansDecoder()
        self.quantized_cdf_list = None
        self.cdf_length_list = None
        self.offset_list = None
        self.symbols_list = None
        self.indexes_list = None
        self.inner_module.update()

    def update(self):
        """（由 scale_table 构建 CDF 表，供 rANS 编解码查表）"""
        self.inner_module.update()
        self.quantized_cdf_list = self.inner_module.quantized_cdf.tolist()
        self.cdf_length_list = self.inner_module.cdf_length.tolist()
        self.offset_list = self.inner_module.offset.tolist()

    def forward(self, x, means, scales):
        """（训练时返回逐元素似然 p(x | means, scales)）"""
        p = self.inner_module._likelihood(x, means=means, scales=scales)
        if self.inner_module.use_likelihood_bound:
            p = self.inner_module.likelihood_lower_bound(p)
        return p

    def encode_buffer_init(self):
        """（清空编码符号缓冲）"""
        self.symbols_list = []
        self.indexes_list = []

    def encode_buffer_append(self, inputs, means, scales):
        """（把一步扫描得到的 (符号, 均值, 尺度) 转成 (symbols, indexes) 并入缓冲）"""
        assert isinstance(inputs, torch.Tensor), 'Inputs should be a torch.Tensor'
        assert means is None or isinstance(means, float) or (isinstance(means, torch.Tensor) and inputs.shape == means.shape), \
            f'Means should be a float or a tensor with the same shape as inputs'
        assert isinstance(scales, float) or (isinstance(scales, torch.Tensor) and inputs.shape == scales.shape), \
            f'Scales should be a float or a tensor with the same shape as inputs'
        indexes = self.inner_module.build_indexes(scales)
        symbols = self.inner_module.quantize(inputs, 'symbols', means)
        self.symbols_list.append(symbols.view(-1).cpu())
        self.indexes_list.append(indexes.view(-1).cpu())

    def encode_with_buffer(self):
        """（把缓冲的全部符号用 rANS 一次性编码，返回字节流）"""
        symbols = torch.concat(self.symbols_list).tolist()
        indexes = torch.concat(self.indexes_list).tolist()
        self.symbols_list = []
        self.indexes_list = []
        self.encoder.encode_with_indexes(symbols, indexes, self.quantized_cdf_list,
                                         self.cdf_length_list, self.offset_list)
        return self.encoder.flush()

    def decode_buffer_init(self, bs):
        """（设置 rANS 解码流）"""
        self.decoder.set_stream(bs)

    def decode_with_buffer(self, inputs_ref, means, scales):
        """（按 indexes 从流中解出符号并加回均值，inputs_ref 仅提供形状/设备模板）"""
        assert isinstance(inputs_ref, torch.Tensor), 'Inputs should be a torch.Tensor'
        assert means is None or isinstance(means, float) or (isinstance(means, torch.Tensor) and inputs_ref.shape == means.shape), \
            f'Means should be a float or a tensor with the same shape as inputs_ref'
        assert isinstance(scales, float) or (isinstance(scales, torch.Tensor) and inputs_ref.shape == scales.shape), \
            f'Scales should be a float or a tensor with the same shape as inputs_ref'
        indexes = self.inner_module.build_indexes(scales)
        symbols = self.decoder.decode_stream(indexes.view(-1).tolist(), self.quantized_cdf_list,
                                             self.cdf_length_list, self.offset_list)
        return self.inner_module.dequantize(inputs_ref.new_tensor(symbols).view_as(inputs_ref), means)