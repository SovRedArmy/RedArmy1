"""
量化器（训练期可微量化，推理/编码期取硬舍入 round()）：
    soft-round-noise（阶段 1）：软舍入 + Kumaraswamy 噪声 + 条件均值反变换，梯度平滑；
    quant-noise（阶段 2）：QuantNoise——以退火概率对部分元素取硬舍入、其余保持连续值。
"""
from .utils import *


def _add_kumaraswamy_noise(inputs, kum):
    """（加 Kumaraswamy 分布噪声：模拟均匀噪声但可在软舍入后正确去噪）"""
    a = kum
    b = (1. / a) * (2 ** a * (a - 1) + 1)
    return inputs + ((1 - torch.rand_like(inputs) ** (1 / b)) ** (1 / a) - 0.5)


def _soft_round(x, temperature):
    """（软舍入：温度越小越接近 round()，但处处可微）"""
    m = x.floor() + 0.5
    z = 2 * torch.tanh(0.5 / temperature)
    r = torch.tanh((x - m) / temperature) / z
    return m + r


def _soft_round_inverse(x, temperature):
    """（软舍入的逆函数：用于条件均值计算）"""
    m = x.floor() + 0.5
    z = 2 * torch.tanh(0.5 / temperature)
    r = torch.atanh((x - m) * z) * temperature
    return m + r


def _soft_round_conditional_mean(x, temperature):
    """（软舍入+噪声下的条件期望 E[x | soft_round(x)+noise]，作为无偏量化估计）"""
    return _soft_round_inverse(x - 0.5, temperature) + 0.5


class QuantizerBase(FuncModule):
    """（量化器基类：forward(inputs, params) 返回量化结果）"""
    def __init__(self):
        super().__init__()

    def forward(self, inputs, params):
        raise NotImplementedError


class NoneQuantizer(QuantizerBase):
    """（不量化：训练时恒等，评估时 round）"""
    def forward(self, inputs, params):
        if self.training:
            return inputs
        else:
            return inputs.round()


class UniformNoiseQuantizer(QuantizerBase):
    """（均匀噪声量化：经典 Ballé 式 additive noise）"""
    def forward(self, inputs, params):
        if self.training:
            return inputs + (torch.rand_like(inputs) - 0.5)
        else:
            return inputs.round()


class STEQuantizer(QuantizerBase):
    """（直通估计器 STE 量化）"""
    def forward(self, inputs, params):
        if self.training:
            return compressai.ops.quantize_ste(inputs)
        else:
            return inputs.round()


class SoftRoundingQuantizer(QuantizerBase):
    """
    See https://github.com/google-deepmind/c3_neural_compression/blob/main/model/latents.py
    （软舍入量化器（论文阶段 1）：soft_round -> Kumaraswamy 噪声 -> 条件均值，
      温度 temp 与噪声参数 kum 由 set_epoch 线性退火。）
    """
    def __init__(self, use_noise):
        super().__init__()
        self.use_noise = use_noise
        self.register_buffer('temp', torch.tensor(1., dtype=torch.float32), persistent=False)
        self.register_buffer('kum', torch.tensor(2., dtype=torch.float32), persistent=False)

    def extra_repr(self):
        s = 'use_noise={use_noise}'
        return s.format(**self.__dict__)

    def set_params(self, temp, kum):
        """（更新退火超参）"""
        self.temp.data.fill_(temp)
        self.kum.data.fill_(kum)

    def forward(self, inputs, params):
        if self.training:
            return _soft_round_conditional_mean(
                _add_kumaraswamy_noise(_soft_round(inputs, self.temp), self.kum), self.temp
            )
        else:
            return inputs.round()


class QuantNoiseQuantizer(QuantizerBase):
    """
    （QuantNoise 量化器（论文阶段 2）：以 noise_ratio 的概率把元素替换为硬舍入值，
      其余保持连续——模拟部分权重被量化时的网络行为，随训练退火到全量化。）
    """
    def __init__(self, use_ste):
        super().__init__()
        self.use_ste = use_ste
        self.register_buffer('noise_ratio', torch.tensor(0., dtype=torch.float32), persistent=False)

    def extra_repr(self):
        s = 'use_ste={use_ste}'
        return s.format(**self.__dict__)

    def set_params(self, noise_ratio):
        """（更新退火超参）"""
        self.noise_ratio.data.fill_(noise_ratio)

    def forward(self, inputs, params):
        if self.training:
            mask = (torch.rand_like(inputs) > self.noise_ratio).to(inputs.dtype)
            if self.use_ste:
                return mask * inputs + (1. - mask) * compressai.ops.quantize_ste(inputs)
            else:
                return mask * inputs + (1. - mask) * inputs.round()
        else:
            return inputs.round()


def get_quantizer(quantizer, **kwargs):
    """（量化器工厂：按配置名实例化）"""
    if quantizer == 'none':
        return NoneQuantizer()
    elif quantizer == 'uniform-noise':
        return UniformNoiseQuantizer()
    elif quantizer == 'ste':
        return STEQuantizer()
    elif quantizer == 'soft-round-noise':
        return SoftRoundingQuantizer(use_noise=True)
    elif quantizer == 'quant-noise':
        return QuantNoiseQuantizer(use_ste=False)
    elif quantizer == 'quant-noise-ste':
        return QuantNoiseQuantizer(use_ste=True)
    else:
        raise ValueError


def update_quantizer(quantizer, **kwargs):
    """（按类型把退火超参写入量化器；每个 epoch 调用一次）"""
    if type(quantizer) == NoneQuantizer:
        pass
    elif type(quantizer) == UniformNoiseQuantizer:
        pass
    elif type(quantizer) == STEQuantizer:
        pass
    elif type(quantizer) == SoftRoundingQuantizer:
        quantizer.set_params(kwargs['soft_round_temp'], kwargs['kum_noise'])
    elif type(quantizer) == QuantNoiseQuantizer:
        quantizer.set_params(kwargs['quant_noise'])
    else:
        pass