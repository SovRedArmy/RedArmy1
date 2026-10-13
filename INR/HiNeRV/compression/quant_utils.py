"""
量化（论文的 Quant-Noise 量化感知训练 + 逐通道对称量化）：
- _quantize_ste：逐通道对称量化 x_q = round(x/scale)，可选 STE 直通梯度；
- QuantNoise：训练时按 noise_ratio 概率把权重替换为量化-反量化值（默认 0.9），
  论文发现不用 STE 的版本在低比特下更好，可支持 6bit 量化；
- quant_model：把整个 state_dict 量化成整数并落盘配置。
"""
from .utils import *
from models.layers import FeatureGrid


# 可被量化的层类型（比剪枝多了 FeatureGrid 网格）
_quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, FeatureGrid)


def _is_quant_target(model, name, module):
    """判断模块是否为量化目标：位于 bitstream_prefix 内、不在 no_quant_prefix 内、类型匹配。"""
    return name.startswith(model.bitstream_prefix) and not name.startswith(model.no_quant_prefix) and isinstance(module, _quant_target_cls)


def _ste(x):
    """
    Straight-through estimator.
    直通估计器：前向 round，反向恒等（梯度直通）。
    """
    return (x.round() - x).detach() + x


def _quantize_ste(x, n, axis=None):
    """
    Per-channel & symmetric quantization with STE.
    逐通道（沿 axis）对称量化到 n bit：
      scale = 2·max|x| / (2^n - 1)，x_q = round(x/scale) 并截断到对称范围。
    """
    quant_range = 2. ** n - 1.
    x_max = abs(x).max(dim=axis, keepdim=True)[0] if axis is not None else abs(x).max()
    x_scale = 2 * x_max / quant_range + 1e-6
    x_q = _ste(x / x_scale).clamp(-2**(n - 1), 2**(n - 1) - 1)
    return x_q, x_scale


class QuantNoise(nn.Module):
    """
    Quant-Noise with optional STE.
    量化噪声参数化（挂在模块 weight 上）：
    训练时以 (1-noise_ratio) 的概率把权重换成"量化再反量化"的版本，
    让权重在训练中就适应量化误差；评估时恒等返回。ste 控制梯度是否直通。
    """
    def __init__(self, bitwidth, noise_ratio, ste, axis):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))  # 量化比特数
        self.register_buffer('noise_ratio', torch.tensor(noise_ratio, dtype=torch.float32))  # 噪声比例（保留原值的概率）
        self.ste = ste
        self.axis = axis  # 逐通道量化的轴（None 表示逐张量）

    def extra_repr(self):
        s = 'ste={ste}, axis={axis}'
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:
            # 训练：量化-反量化 + 按掩码混合（大部分权重被替换为量化版本）
            x_q, x_scale = _quantize_ste(x, self.bitwidth, self.axis)
            x_q = x_q if self.ste else x_q.detach()
            x_qr = x_q.to(x.dtype) * x_scale
            mask = (torch.rand_like(x) > self.noise_ratio).to(x.dtype)
            return x * mask + x_qr * (1. - mask)
        else:
            # 评估：不做量化（真正的量化在 compress_bitstream 中显式进行）
            return x


def init_quantization(args, logger, model):
    """
    Initialize quantization for the model.
    为所有量化目标层的 weight 注册 QuantNoise 参数化
    （初始 bitwidth=8、noise_ratio=0，即不影响主训练阶段）。
    """
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if _is_quant_target(model, k, v):
                if args.debug:
                    logger.info(f'     Set quatization: {k}.weight')
                quant_layer = QuantNoise(bitwidth=8, noise_ratio=0., ste=False, axis=compute_best_quant_axis(v.weight))
                quant_layer.to(list(v.parameters())[0].device)
                torch.nn.utils.parametrize.register_parametrization(v, 'weight', quant_layer)


def set_quantization(args, logger, model, quant_level, quant_noise, quant_ste):
    """
    Set quantization for the model.
    更新已有 QuantNoise 参数化的配置（进入 QAT 阶段时调用）：
    设置目标比特数、噪声比例与是否使用 STE。
    """
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, QuantNoise):
                if args.debug:
                    logger.info(f'     Update quatization: {k}.weight')
                v.bitwidth.copy_(quant_level)
                v.noise_ratio.copy_(quant_noise)
                v.ste = quant_ste


def _quant_tensor(args, logger, x, quant_level):
    """
    Quantize a tensor.
    量化单个张量：返回整数量化值 x_q、反量化值 x_qr，以及含量化轴与 scale 的元信息 meta。
    """
    axis = compute_best_quant_axis(x)
    with torch.no_grad():
        x_q, x_scale = _quantize_ste(x, quant_level, axis)
        x_q = x_q.to(torch.int32)
        x_qr = x_q.to(x.dtype) * x_scale

    if args.debug:
        logger.info(f'     Shape: {x.shape}')
        logger.info(f'     quant axis: {axis}')

    # Return the quantisied tensors (both int/float) and config
    meta = {
        'axis': axis,
        'scale': x_scale.half()  # scale 以半精度存入码流
    }

    return x_q, x_qr, meta


def quant_model(args, logger, model, quant_level):
    """
    Quantize the full model.
    量化整个模型：
    - quant_level=32 视为不量化；
    - 否则对 state_dict 中维度>1 的浮点张量做逐张量量化（bias 等一维张量保留），
      把反量化后的权重写回模型（保证后续评估反映真实量化损失），
      返回整数 state_dict 与量化配置（含每张量的 scale）。
    """
    model = unwrap_model(model)

    # Get the quantisation targets
    excluded_keys = set()

    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix):
            excluded_keys.add(k)

    # Quantize
    qr_state_dict = copy.deepcopy(model.state_dict())
    q_state_dict = {}
    q_config = {'quant_level': quant_level}
    if quant_level == 32:
        pass
    elif quant_level <= 16:
        for k, v in model.state_dict().items():
            if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                q_state_dict[k], qr_state_dict[k], q_config[k] = \
                    _quant_tensor(args, logger, v, quant_level)
    else:
        raise ValueError

    # Load quantisied state dict
    # 把反量化权重载回模型（模型现在处于"量化后"状态）
    model.load_state_dict(qr_state_dict)

    # Ruturn configs
    return q_state_dict, q_config