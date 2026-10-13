"""
量化模块（MyHiNeRV 版）：HiNeRV 的 Quant-Noise QAT 基础上，新增 DCVC-RT 风格的
"残差预测编码"量化——P 模型的权重相对参考模型（已解码的上一 clip 模型）的残差
进行量化与熵编码（类比 DCVC 用已重建信息作为熵编码上下文/先验均值）：
- ResidualQuantNoise：QAT 时对 (θ − θ_ref) 注入量化噪声，训练权重适应残差量化误差
  （DCVC-RT 的 STE 失真路径 + 噪声率估计路径思想，这里训练期统一用噪声注入）；
- quant_model（residual 模式）：θ_res = θ − θ_ref 逐通道对称量化，可选 dead-zone
  （|q| < dz 的残差直接置 0 并跳过熵编码——类比 DCVC-RT 的 force_zero_thres）；
- snap_scales_to_table：把逐通道量化 scale 吸附到对数均匀档位表（DCVC-RT 的
  128 档 scale_table 思想），码流只存 uint8 档位索引而非 half 浮点。
I 模型（无参考）仍走 HiNeRV 原有的 intra 路径（QuantNoise + quant_model）。
"""
from .utils import *
from models.layers import FeatureGrid


# 可被量化的层类型（与 HiNeRV 一致：比剪枝多了 FeatureGrid 网格）
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


def _soft_round(x, temperature):
    """NVRC 软舍入：温度越小越接近 round()，但处处可微（标量常量 tensor 化，兼容 python float temp）。"""
    t = torch.as_tensor(temperature, dtype=x.dtype, device=x.device)
    m = x.floor() + 0.5
    z = 2 * torch.tanh(0.5 / t)
    r = torch.tanh((x - m) / t) / z
    return m + r


def _soft_round_inverse(x, temperature):
    """软舍入的逆函数（条件均值计算用；标量常量 tensor 化）。"""
    t = torch.as_tensor(temperature, dtype=x.dtype, device=x.device)
    m = x.floor() + 0.5
    z = 2 * torch.tanh(0.5 / t)
    r = torch.atanh((x - m) * z) * t
    return m + r


def _soft_round_conditional_mean(x, temperature):
    """软舍入+噪声下的条件期望 E[x | soft_round(x)+noise]——量化误差的无偏可微估计。"""
    return _soft_round_inverse(x - 0.5, temperature) + 0.5


def _add_kumaraswamy_noise(inputs, kum=1.0):
    """Kumaraswamy 噪声（NVRC）：软舍入后可正确去噪的加性噪声模拟。"""
    a = kum
    b = (1. / a) * (2 ** a * (a - 1) + 1)
    return inputs + ((1 - torch.rand_like(inputs) ** (1 / b)) ** (1 / a) - 0.5)


def apply_col_prune(model, col_pruner):
    """D1: 将列级掩码应用到所有 QuantEntropy 层（列掩码乘进 original 权重）。
    键名统一使用 state_dict 的 canonical 格式（不含 .parametrizations 中间层）。"""
    from compression.utils import unwrap_model as _unwrap
    m = _unwrap(model)
    if col_pruner is None or not col_pruner.col_masks:
        return
    with torch.no_grad():
        for name, mod in m.named_modules():
            if not hasattr(mod, 'parametrizations'):
                continue
            for wname, plist in mod.parametrizations.items():
                if wname != 'weight':
                    continue
                # canonical 键：模块路径.weight（与 state_dict 一致）
                key = name + '.' + wname if name else wname
                if key not in col_pruner.col_masks:
                    continue
                w = plist.original
                if w.ndim >= 2:
                    mask = col_pruner.col_masks[key]
                    shape = [w.shape[0]] + [1] * (w.ndim - 1)
                    w.mul_(mask.view(shape).to(w.dtype).to(w.device))


def set_soft_round_anneal(model, temp):
    """C4：全局设置所有 QuantEntropy 的软量化温度（train 循环每 epoch 按进度退火调用）。"""
    from compression.utils import unwrap_model as _unwrap
    m = _unwrap(model)
    with torch.no_grad():
        for k, v in m.named_modules():
            if isinstance(v, QuantEntropy):
                v.set_soft_round(temp)


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
    Quant-Noise with optional STE（HiNeRV 原版，I 模型 intra 路径）。
    训练时以 (1-noise_ratio) 的概率把权重换成"量化再反量化"的版本；
    评估时恒等返回（真正的量化在 compress_bitstream 中显式进行）。
    """
    def __init__(self, bitwidth, noise_ratio, ste, axis):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))
        self.register_buffer('noise_ratio', torch.tensor(noise_ratio, dtype=torch.float32))
        self.ste = ste
        self.axis = axis

    def extra_repr(self):
        s = 'ste={ste}, axis={axis}'
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:
            x_q, x_scale = _quantize_ste(x, self.bitwidth, self.axis)
            x_q = x_q if self.ste else x_q.detach()
            x_qr = x_q.to(x.dtype) * x_scale
            mask = (torch.rand_like(x) > self.noise_ratio).to(x.dtype)
            return x * mask + x_qr * (1. - mask)
        else:
            return x


class ResidualQuantNoise(nn.Module):
    """
    残差量化噪声参数化（P 模型 QAT，DCVC-RT 借鉴）：
    训练时对残差 r = θ − θ_ref 注入量化-反量化噪声，输出 θ_ref + r_noisy，
    使权重在训练中适应"相对参考模型的残差量化误差"。
    ref 为缓冲区（persistent=False，不进 state_dict/码流），由 set_reference 注入。
    评估时恒等返回（真正的残差量化在 compress_bitstream 中显式进行）。
    """
    def __init__(self, bitwidth, noise_ratio, ste, axis, deadzone=1):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))
        self.register_buffer('noise_ratio', torch.tensor(noise_ratio, dtype=torch.float32))
        self.register_buffer('deadzone', torch.tensor(deadzone, dtype=torch.float32))
        self.ste = ste
        self.axis = axis
        self.register_buffer('ref', torch.zeros(0), persistent=False)

    def extra_repr(self):
        s = 'ste={ste}, axis={axis}, deadzone={dz}, has_ref={hr}'
        return s.format(ste=self.ste, axis=self.axis, dz=int(self.deadzone.item()), hr=int(self.ref.numel() > 0))

    def set_reference(self, ref: torch.Tensor):
        """注入参考权重（自动迁移到本模块所在设备——参考常来自 CPU 侧的推导/解码状态）。"""
        ref = ref.detach().to(device=self.ref.device, dtype=torch.float32)
        if self.ref.shape != ref.shape:
            self.ref = ref.clone()
        else:
            self.ref.copy_(ref)

    def forward(self, x):
        if not self.training or self.ref.numel() == 0:
            return x
        ref = self.ref.to(device=x.device, dtype=x.dtype)
        r = x - ref
        r_q, r_scale = _quantize_ste(r, self.bitwidth, self.axis)
        # dead-zone：|q| < dz 的残差直接置零（类比 DCVC-RT force_zero_thres，产生更稀疏的残差）
        r_q = torch.where(r_q.abs() < self.deadzone.to(r_q.dtype), torch.zeros_like(r_q), r_q)
        r_q = r_q if self.ste else r_q.detach()
        r_qr = r_q.to(x.dtype) * r_scale
        mask = (torch.rand_like(x) > self.noise_ratio).to(x.dtype)
        r_out = r * mask + r_qr * (1. - mask)
        return ref + r_out


def gaussian_cdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """高斯 CDF（UAR-NVC QEC 的熵模型）。"""
    return 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))


class QuantEntropy(nn.Module):
    """
    UAR-NVC QEC 风格的率失真感知量化参数化：
    - 每层一个可学习量化步长 q_step（nn.Parameter），训练时以 STE 量化
      x_q = round(x / q_step)，并用零均值高斯估计每个整数的概率
      p_k = Φ((k+0.5)/scale) − Φ((k−0.5)/scale)（scale = std(x_q)，在线估计存 buffer）；
    - self.prob 供训练循环把 bpp 估计（Σ −log2 p / 总像素）加进损失（RD 训练）；
    - residual 模式（P 模型）先减去参考权重 ref（w_pre），量化后再加回；
    - use_hyper=True（quant_mode='hyper'）时自身不再估计熵（率由 WeightHyperprior
      的逐元素 μ/σ 模型估计，见 hyperprior.py），前向只负责 STE 量化（失真路径）；
    - start=False 时恒等（QAT 阶段由 start_quantization 启用）。
    """
    def __init__(self, residual=False, use_hyper=False):
        super().__init__()
        self.q_step = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(0.1, dtype=torch.float32))
        self.register_buffer('start', torch.tensor(False))
        self.register_buffer('soft_temp', torch.tensor(0., dtype=torch.float32))  # >0 启用 NVRC 式软量化（C4）
        self.residual = residual
        self.use_hyper = use_hyper
        self.register_buffer('ref', torch.zeros(0), persistent=False)
        self.prob = None
        self._last_input = None  # 最近一次 forward 的输入（get_residual 用）

    def extra_repr(self):
        return (f'q_step={self.q_step.item():.2e}, scale={self.scale.item():.4f}, '
                f'residual={self.residual}, use_hyper={self.use_hyper}, has_ref={int(self.ref.numel() > 0)}')

    def set_soft_round(self, temp):
        """C4：设置软量化温度（0=关闭回硬 STE；由 set_soft_round_anneal 全局退火）。"""
        self.soft_temp.fill_(float(temp))

    def set_reference(self, ref: torch.Tensor):
        ref = ref.detach().to(device=self.ref.device, dtype=torch.float32)
        if self.ref.shape != ref.shape:
            self.ref = ref.clone()
        else:
            self.ref.copy_(ref)

    def get_residual(self):
        """当前（残差）权重：x − ref（residual 且有参考时）或 x（intra）；未启动时返回 None。"""
        if not bool(self.start) or self._last_input is None:
            return None
        if self.residual and self.ref.numel() > 0:
            return self._last_input - self.ref.to(self._last_input.dtype)
        return self._last_input

    def forward(self, x):
        self._last_input = x
        if not self.training or not bool(self.start):
            return x
        ref = self.ref.to(device=x.device, dtype=x.dtype) if (self.residual and self.ref.numel() > 0) else None
        r = x - ref if ref is not None else x
        q_step = torch.clamp_min(self.q_step, 1e-3)
        z = r / q_step
        if float(self.soft_temp) > 0:
            # C4（NVRC soft-round-noise）：无偏可微的量化误差估计；
            # 率路径同样用软值——概率对权重的梯度得以流通（硬 round 是分段常数、零梯度）
            q = _soft_round_conditional_mean(
                _add_kumaraswamy_noise(_soft_round(z, float(self.soft_temp)), 1.0),
                float(self.soft_temp))
        else:
            q = _ste(z)
        q = q.clamp(-30000, 30000)  # 限幅防止符号表过大
        if not self.use_hyper:
            scale = torch.clamp(torch.sqrt(torch.var(q) + 1e-5), min=2e-3, max=150.)
            self.scale.copy_(scale.detach())
            prob = torch.clamp_min(gaussian_cdf(q + 0.5, 0., scale) - gaussian_cdf(q - 0.5, 0., scale), min=2 ** -32)
            self.prob = prob  # 训练循环读取做率损失
        qr = q.to(x.dtype) * q_step
        if ref is not None:
            qr = qr + ref
        return qr


def start_quantization(model):
    """启动 QEC 量化（QAT 阶段调用）：QuantEntropy.start=True。"""
    model = unwrap_model(model)
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, QuantEntropy):
                v.start.fill_(True)


def get_mean_qstep(model):
    """
    所有 QuantEntropy / WeightHyperprior 可学习量化步长的均值（标量 float）。
    供 QAT 训练日志观察 q_step 动态：率项推大、失真顶回的平衡过程。
    """
    model = unwrap_model(model)
    vals = []
    for k, v in model.named_modules():
        if isinstance(v, QuantEntropy) and bool(v.start):
            vals.append(float(torch.clamp_min(v.q_step.detach(), 1e-3).item()))
    hyper = getattr(model, 'hyper', None)
    if hyper is not None:
        for key in hyper.q_steps:
            vals.append(float(torch.clamp_min(hyper.q_steps[key].detach(), 1e-3).item()))
    return float(np.mean(vals)) if vals else None


def get_rate_bpp(model, num_pixels: float):
    """
    QEC/Hyper 的码率估计（bpp）：
    - quant_mode='hyper'（model.hyper 存在）：WeightHyperprior 的逐元素 μ/σ 模型；
    - 否则（qec）：所有 QuantEntropy 层的 Σ −log2 p / 总像素。
    仅在训练且已启动量化时有效；返回标量张量（无则 None）。
    """
    model = unwrap_model(model)
    if getattr(model, 'hyper', None) is not None:
        from .hyperprior import compute_hyper_rate_bpp
        return compute_hyper_rate_bpp(model, num_pixels)
    bits = None
    for k, v in model.named_modules():
        if isinstance(v, QuantEntropy) and v.prob is not None:
            b = -torch.log2(v.prob).sum()
            bits = b if bits is None else bits + b
    if bits is None:
        return None
    return bits / float(num_pixels)


def _iter_quant_parametrizations(model):
    """
    遍历 (owner_path, weight_name, parametrization_module)。
    parametrization 的模块路径形如 '<owner>.parametrizations.weight.0'，
    据此还原出宿主模块路径与权重名。
    """
    for k, v in model.named_modules():
        if isinstance(v, (QuantNoise, ResidualQuantNoise, QuantEntropy)) and '.parametrizations.' in k:
            owner_path, rest = k.split('.parametrizations.')
            weight_name = rest.split('.')[0]
            yield owner_path, weight_name, v


def _canon_key(k):
    """
    把 state_dict 键规范化为"宿主.权重"形式（去掉参数化包装）：
    '<owner>.parametrizations.weight.original' → '<owner>.weight'。
    参数化内部的其它键（bitwidth/noise_ratio/mask 等 buffer）原样返回（不参与匹配）。
    参考模型与当前模型的参数化状态可能不同，比较前统一规范化。
    """
    if '.parametrizations.' in k and k.endswith('.original'):
        owner, rest = k.split('.parametrizations.')
        return f'{owner}.{rest.split(".")[0]}'
    return k


def set_reference(model, ref_state_dict):
    """
    为所有残差量化参数化（ResidualQuantNoise / residual QuantEntropy）注入参考权重
    （P 模型进入 QAT 前调用）。
    ref_state_dict: 参考模型（已解码重建或推导）的 state_dict（键名允许带/不带参数化包装）。
    形状不匹配的权重（如 clip 长度不同的 CRSPE/时间网格）自动跳过，
    这些权重回退到 intra 量化路径。
    """
    model = unwrap_model(model)
    ref_canon = {_canon_key(k): v for k, v in ref_state_dict.items()}
    with torch.no_grad():
        n, skipped = 0, 0
        for owner_path, weight_name, p in _iter_quant_parametrizations(model):
            if not isinstance(p, (ResidualQuantNoise, QuantEntropy)):
                continue
            key = f'{owner_path}.{weight_name}'
            ref_v = ref_canon.get(key)
            owner = model.get_submodule(owner_path)
            w_shape = tuple(owner.parametrizations.weight.original.shape)
            if ref_v is None or tuple(ref_v.shape) != w_shape:
                skipped += 1
                continue
            p.set_reference(ref_v)
            n += 1
    return n


def init_quantization(args, logger, model):
    """
    为所有量化目标层的 weight 注册量化参数化。
    - args.quant_mode='noise'（默认，HiNeRV 路径）：I 模型注册 QuantNoise，
      P 模型（args.quant_residual=True）注册 ResidualQuantNoise（残差 QAT）；
    - args.quant_mode='qec'（UAR-NVC QEC 路径）：注册 QuantEntropy
      （每层可学习 q_step + 高斯熵率估计，训练时 RD 损失，见 get_rate_bpp）。
    """
    model = unwrap_model(model)
    quant_mode = getattr(args, 'quant_mode', 'noise')
    residual = getattr(args, 'quant_residual', False)
    logger.info(f'Init quantization: mode={quant_mode} residual={residual}')

    with torch.no_grad():
        for k, v in model.named_modules():
            if _is_quant_target(model, k, v):
                if args.debug:
                    logger.info(f'     Set quatization: {k}.weight')
                if quant_mode == 'qec':
                    quant_layer = QuantEntropy(residual=residual)
                elif quant_mode == 'hyper':
                    quant_layer = QuantEntropy(residual=residual, use_hyper=True)
                elif residual:
                    quant_layer = ResidualQuantNoise(bitwidth=8, noise_ratio=0., ste=False,
                                                     axis=compute_best_quant_axis(v.weight),
                                                     deadzone=getattr(args, 'quant_residual_deadzone', 1))
                else:
                    quant_layer = QuantNoise(bitwidth=8, noise_ratio=0., ste=False, axis=compute_best_quant_axis(v.weight))
                quant_layer.to(list(v.parameters())[0].device)
                torch.nn.utils.parametrize.register_parametrization(v, 'weight', quant_layer)


def set_quantization(args, logger, model, quant_level, quant_noise, quant_ste):
    """
    更新量化参数化配置（进入 QAT 阶段时调用）：设置目标比特数、噪声比例与 STE。
    """
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, (QuantNoise, ResidualQuantNoise)):
                if args.debug:
                    logger.info(f'     Update quatization: {k}.weight')
                v.bitwidth.copy_(quant_level)
                v.noise_ratio.copy_(quant_noise)
                v.ste = quant_ste
                if isinstance(v, ResidualQuantNoise):
                    v.deadzone.copy_(float(getattr(args, 'quant_residual_deadzone', 1)))


def _quant_tensor(args, logger, x, quant_level, ref=None, deadzone=1):
    """
    量化单个张量：intra（ref=None）或 residual 模式。
    返回整数量化值 x_q、反量化值 x_qr（residual 模式含 ref 加回）与元信息 meta。
    """
    axis = compute_best_quant_axis(x)
    r = x if ref is None else x - ref
    with torch.no_grad():
        x_q, x_scale = _quantize_ste(r, quant_level, axis)
        if deadzone > 1:
            x_q = torch.where(x_q.abs() < deadzone, torch.zeros_like(x_q), x_q)
        x_q = x_q.to(torch.int32)
        if ref is None:
            x_qr = x_q.to(x.dtype) * x_scale
        else:
            x_qr = x_q.to(x.dtype) * x_scale + ref

    if args.debug:
        logger.info(f'     Shape: {x.shape}')
        logger.info(f'     quant axis: {axis}    residual: {ref is not None}')

    meta = {
        'axis': axis,
        'scale': x_scale.half(),
        'residual': ref is not None,
    }
    return x_q, x_qr, meta


def snap_scales_to_table(q_config, q_state_dict, bits):
    """
    DCVC-RT 风格 scale 档位表：把所有张量的逐通道 scale 吸附到对数均匀档位表
    （2^bits 档），返回新 config（scale 以 uint8 档位索引 + 表参数 lo/hi 存储）
    以及按吸附后 scale 重新量化的整数 state_dict。
    bits<=0 时原样返回。
    """
    if bits is None or bits <= 0:
        return q_config, q_state_dict

    scales = [v['scale'].float() for k, v in q_config.items() if k != 'quant_level' and isinstance(v, dict) and 'scale' in v]
    if len(scales) == 0:
        return q_config, q_state_dict
    lo = max(min(s.min().item() for s in scales), 1e-8)
    hi = max(s.max().item() for s in scales)
    if hi <= lo:
        hi = lo * 2
    n_levels = 2 ** bits
    log_lo, log_hi = math.log(lo), math.log(hi)

    def snap(s):
        t = ((torch.log(s.clamp_min(1e-12)) - log_lo) / (log_hi - log_lo) * (n_levels - 1)).round().clamp(0, n_levels - 1)
        return t.to(torch.uint8), torch.exp(log_lo + t.float() / (n_levels - 1) * (log_hi - log_lo))

    new_config = {'quant_level': q_config['quant_level'], 'scale_table': {'lo': lo, 'hi': hi, 'bits': bits}}
    # 吸附只改 scale 存储，不影响已量化的整数（scale 吸附误差 ≤ 半档，
    # 在 QAT 噪声训练下可接受；解码端按表恢复 scale 反量化）。
    for k, v in q_config.items():
        if k == 'quant_level' or not isinstance(v, dict) or 'scale' not in v:
            continue
        idx, s_snapped = snap(v['scale'].float())
        vc = dict(v)
        vc['scale'] = s_snapped.half()
        vc['scale_index'] = idx
        new_config[k] = vc
    return new_config, q_state_dict


def _quant_tensor_qec(args, logger, x, q_step, ref=None):
    """
    QEC 量化单个张量（UAR-NVC 路径）：q = round((x − ref)/q_step)，scale = std(q)。
    q_step/scale 均来自训练中学习/估计的值（熵编码用高斯模型）。
    """
    with torch.no_grad():
        q_step = max(float(q_step), 1e-3)
        r = x if ref is None else x - ref
        x_q = (r / q_step).round().clamp(-30000, 30000)
        x_q = x_q.to(torch.int32)
        scale = float(torch.clamp(torch.sqrt(torch.var(x_q.float()) + 1e-5), min=2e-3, max=150.))
        if ref is None:
            x_qr = x_q.to(x.dtype) * q_step
        else:
            x_qr = x_q.to(x.dtype) * q_step + ref
    meta = {
        'qec': True,
        'q_step': q_step,
        'scale': scale,
        'residual': ref is not None,
    }
    return x_q, x_qr, meta


def _quant_tensor_hyper(args, logger, x, hyper, canon_key_str, ref=None):
    """
    hyper 模式量化单个张量（DCVC-RT 风格迷你超先验）：
      r = x − ref → z = g_a(r) → z_hat = round(z/q_z) → (μ, σ) = g_s(z_hat)
      q = round(r/q_step)，s = q − round(μ)
    返回 (s_int, z_hat_int, x_qr, meta)。μ/σ 在熵编码时按（half 固化的）
    超网络参数在 CPU 上重算，保证编解码一致。
    """
    with torch.no_grad():
        # 调用方（quant_model）已对超网络做 half 固化，保证编码端 μ/σ 与解码端一致。
        # synthesize 固定在 CPU 上计算（编解码一致性），因此整个流程都在 CPU 进行。
        r = (x if ref is None else x - ref).float().cpu()
        z = hyper.analyze(canon_key_str, r)
        z_hat = hyper.quantize_z(canon_key_str, z).round()
        q_step = max(float(torch.clamp_min(hyper.q_steps[hyper._safe_key(canon_key_str)], 1e-3).item()), 1e-3)
        q_z = max(float(torch.clamp_min(hyper.q_z[hyper._safe_key(canon_key_str)], 1e-3).item()), 1e-3)
        mu, _sig = hyper.synthesize(canon_key_str, z_hat, tuple(x.shape))
        q = (r / q_step).round()
        s = (q - mu.round()).to(torch.int32)
        x_qr = (s.float() + mu.round()) * q_step
        if ref is not None:
            x_qr = x_qr + ref.float().cpu()
        sigma_z = float(torch.clamp(torch.sqrt(torch.var(z_hat.float()) + 1e-5), min=2e-3, max=150.))
        meta = {
            'hyper': True,
            'q_step': q_step,
            'q_z': q_z,
            'sigma_z': sigma_z,
            'residual': ref is not None,
        }
    return s, z_hat.to(torch.int32), x_qr, meta


def quant_model(args, logger, model, quant_level, reference_state_dict=None):
    """
    量化整个模型（compress_bitstream 调用）：
    - reference_state_dict=None：I 模型 intra 路径（HiNeRV 原版 / QEC）；
    - 否则 P 模型 residual 路径：量化 θ − θ_ref（DCVC-RT 时间上下文思想的模型空间版），
      dead-zone 置零的小残差在熵编码时被稀疏跳过（noise 模式）。
    quant_mode='noise'：逐通道对称量化 + 可选 scale 档位表；
    quant_mode='qec'：每层可学习 q_step（UAR-NVC QEC）。
    返回 (整数 state_dict, 量化配置)。
    """
    model = unwrap_model(model)
    quant_mode = getattr(args, 'quant_mode', 'noise')
    residual = reference_state_dict is not None
    deadzone = float(getattr(args, 'quant_residual_deadzone', 1)) if residual else 1.
    scale_bits = int(getattr(args, 'quant_scale_bits', 0))

    # Get the quantisation targets
    excluded_keys = set()
    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix) or k.startswith(getattr(model, 'no_bitstream_prefix', ())):
            excluded_keys.add(k)

    # QEC：从参数化模块读取每层学习到的 q_step（规范化键名索引）
    qec_q_steps = {}
    if quant_mode == 'qec':
        for owner_path, weight_name, p in _iter_quant_parametrizations(model):
            if isinstance(p, QuantEntropy):
                qec_q_steps[f'{owner_path}.{weight_name}'] = float(torch.clamp_min(p.q_step.detach(), 1e-3).item())

    # 参考权重统一按规范化键查找（覆盖参数化权重、FeatureGrid、CRSPE 等全部量化张量）
    ref_canon = {_canon_key(k): v for k, v in reference_state_dict.items()} if residual else None

    qr_state_dict = copy.deepcopy(model.state_dict())
    q_state_dict = {}
    q_config = {'quant_level': quant_level, 'residual': residual, 'quant_mode': quant_mode}
    hyper = getattr(model, 'hyper', None)
    if quant_mode == 'hyper':
        assert hyper is not None, 'quant_mode=hyper requires attach_hyperprior before training'
        hyper._frozen = True
        hyper.freeze_transmitted()
    if quant_level == 32:
        pass
    elif quant_level <= 16:
        for k, v in model.state_dict().items():
            if k not in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                ref = None
                if residual:
                    ref_v = ref_canon.get(_canon_key(k))
                    if ref_v is not None and tuple(ref_v.shape) == tuple(v.shape):
                        ref = ref_v.float().to(v.device)  # 参考常来自 CPU 侧状态，对齐到权重设备
                if quant_mode == 'hyper' and _canon_key(k) in [f'{o}.{w}' for o, w, _ in _iter_quant_parametrizations(model)]:
                    s_int, z_int, qr, meta = _quant_tensor_hyper(args, logger, v, hyper, _canon_key(k), ref=ref)
                    q_state_dict[k] = s_int
                    q_state_dict[k + '__zhat'] = z_int
                    q_config[k] = meta
                    qr_state_dict[k] = qr
                elif quant_mode == 'qec' and _canon_key(k) in qec_q_steps:
                    q_state_dict[k], qr_state_dict[k], q_config[k] = \
                        _quant_tensor_qec(args, logger, v, qec_q_steps[_canon_key(k)], ref=ref)
                else:
                    q_state_dict[k], qr_state_dict[k], q_config[k] = \
                        _quant_tensor(args, logger, v, quant_level, ref=ref, deadzone=deadzone)
    else:
        raise ValueError

    # 可选：scale 吸附到对数档位表（DCVC-RT scale table；仅 noise 模式有意义）
    if quant_mode != 'qec':
        q_config, q_state_dict = snap_scales_to_table(q_config, q_state_dict, scale_bits)

    # Load quantisied state dict（模型进入"量化后"状态，后续评估反映真实量化损失）
    model.load_state_dict(qr_state_dict)

    return q_state_dict, q_config
