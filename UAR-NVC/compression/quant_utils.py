"""
Quantization
"""
from .utils import *
from model.hinerv_models.layers import FeatureGrid



_quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, FeatureGrid)


def _is_quant_target(model, name, module):
    return name.startswith(model.bitstream_prefix) and not name.startswith(model.no_quant_prefix) and isinstance(module, _quant_target_cls)


def _ste(x):
    """
    Straight-through estimator.
    """
    return (x.round() - x).detach() + x


def _quantize_ste(x, n, axis=None):
    """
    Per-channel & symmetric quantization with STE.
    """
    quant_range = 2. ** n - 1.
    x_max = abs(x).max(dim=axis, keepdim=True)[0] if axis is not None else abs(x).max()
    x_scale = 2 * x_max / quant_range + 1e-6
    x_q = _ste(x / x_scale).clamp(-2**(n - 1), 2**(n - 1) - 1)
    return x_q, x_scale

def _quantize_ste_pq(x, n, axis=None):
    """
    Per-channel & unsymmetric quantization with STE.
    """
    x_min = x.min(dim=axis, keepdim=True)[0] if axis is not None else x.min()
    x_max = x.max(dim=axis, keepdim=True)[0] if axis is not None else x.max()
    x_scale = (x_max - x_min) / (2. ** n - 1.)
    x_q = _ste((x - x_min) / x_scale).clamp(0, 2**n - 1)
    return x_q, x_min, x_scale

# Calculate entropy of input data (default: Gaussian distribution)
def get_data_prob(data, mu_given=None, scale_given=None, dim=None):
    if np.prod(data.shape) <= 1:
        return torch.ones(data.shape, device=data.device)
    mu      = torch.mean(data, dim=dim, keepdim=True
                            ) if mu_given       == None else mu_given
    scale   = torch.clamp(torch.sqrt(torch.var(data, dim=dim, keepdim=True) + 1e-5
                            ) if scale_given    == None else scale_given, min=2e-3, max=150)
    prob    = torch.clamp_min(
        gaussian_cdf(data + 0.5, mu, scale) - gaussian_cdf(data - 0.5, mu, scale),
        min=2 ** -32    # No value can cost more than 16 bits.
    )
    return prob, mu, scale

def gaussian_cdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute the cumulative density function (CDF) of the Gaussian distribution.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        mu (Tensor): Mean.
        sigma (Tensor): Standard deviation.

    Returns:
        Tensor: CDF(x, mu, sigma)
    """
    return 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))


class QuantNoise(nn.Module):
    """
    Quant-Noise with optional STE.
    """
    def __init__(self, bitwidth, noise_ratio, ste, axis):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))
        self.register_buffer('noise_ratio', torch.tensor(noise_ratio, dtype=torch.float32))
        self.ste = ste
        self.axis = axis
        self.start = False

    def extra_repr(self):
        s = 'ste={ste}, axis={axis}'
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training and self.start:
            x_q, x_scale = _quantize_ste(x, self.bitwidth, self.axis)
            x_q = x_q if self.ste else x_q.detach()
            x_qr = x_q.to(x.dtype) * x_scale
            mask = (torch.rand_like(x) > self.noise_ratio).to(x.dtype)
            return x * mask + x_qr * (1. - mask)
        else:
            return x
        

class QuantEntropy(nn.Module):
    """
    Quant-Noise with optional STE.
    version 2: add entropy to training loss
    """
    def __init__(self):
        super().__init__()
        self.q_step = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(0.1, dtype=torch.float32))
        self.prob   = None
        self.prob   = torch.zeros_like(self.q_step)
        self.start  = False
        
        self.model_type = 'I'
        self.w_pre = None

    def extra_repr(self):
        s = 'Quant Aware Training'
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.model_type == 'P' and self.w_pre is None:
            self.w_pre = x.clone().detach()
            
        if self.start:
        # q_step = torch.clamp_min(self.q_step, 1e-3)
            q_step = self.q_step
            q_step = torch.clamp_min(q_step, 1e-3)
                
            if self.model_type == 'P':
                x = x - self.w_pre
                
            x_q = _ste(x / q_step)
            
            self.prob, mu, scale = get_data_prob(x_q, mu_given=0)
            self.scale.data = torch.squeeze(scale)
            
            x_qr = x_q.to(x.dtype) * q_step
            
            if self.model_type == 'P':
                x_qr = x_qr + self.w_pre
                
            return x_qr
        else:
            return x


def init_quantization(args, logger, model):
    """
    Initialize quantization for the model.
    """
    q_mode = args.compression_cfg.q_mode
    assert q_mode in ['PQ', 'QAT', 'QEC'], f'Invalid quantization mode: {q_mode}'
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if _is_quant_target(model, k, v):
                if args.debug:
                    logger.info(f'     Set quatization: {k}.weight')
                if q_mode=='QAT':
                    quant_layer = QuantNoise(bitwidth=8, noise_ratio=0., ste=True, axis=compute_best_quant_axis(v.weight))
                elif q_mode=='QEC':
                    quant_layer = QuantEntropy()
                # quant_layer = QuantNoise(bitwidth=8, noise_ratio=0., ste=False, axis=compute_best_quant_axis(v.weight))
                quant_layer.to(list(v.parameters())[0].device)
                torch.nn.utils.parametrize.register_parametrization(v, 'weight', quant_layer)

def get_model_probs(model):
    """
    Get the quantization prob.
    """
    model = unwrap_model(model)
    prob_list = []
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, QuantEntropy):
                prob_list.append(v.prob)
    return prob_list

def set_quantization(args, logger, model, quant_level, quant_noise, quant_ste):
    """
    Set quantization for the model.
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
                
def start_quantization(model):
    model = unwrap_model(model)
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, QuantNoise) or isinstance(v, QuantEntropy):
                v.start = True

def mark_init_param(args, logger, model, model_type='P'):
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, QuantEntropy):
                if args.debug:
                    logger.info(f'     Update quatization: {k}.weight')
                v.model_type = model_type

def _quant_tensor(args, logger, x, quant_level):
    """
    Quantize a tensor.
    """
    q_mode = args.compression_cfg.q_mode
    axis = compute_best_quant_axis(x)
    with torch.no_grad():
        if q_mode == "PQ":
            x_q, x_min, x_scale = _quantize_ste_pq(x, quant_level, axis)
            x_q = x_q.to(torch.int32)
            x_qr = x_min + x_q.to(x.dtype) * x_scale
        elif q_mode == "QAT":
            x_q, x_scale = _quantize_ste(x, quant_level, axis)
            x_min = None
            x_q = x_q.to(torch.int32)
            x_qr = x_q.to(x.dtype) * x_scale

    if args.debug:
        logger.info(f'     Shape: {x.shape}')
        logger.info(f'     quant axis: {axis}')

    # Return the quantisied tensors (both int/float) and config
    meta = {
        'x_min': x_min,
        'axis': axis,
        'scale': x_scale.half()
    }

    return x_q, x_qr, meta

def _quant_tensor_QEC(args, logger, x, q_step, scale):
    """
    Quantize a tensor.
    """
    with torch.no_grad():
        q_step = max(q_step, torch.ones_like(q_step) * 1e-3)
        x_q = _ste(x / q_step)
        x_q = x_q.to(torch.int32)
        x_qr = x_q.to(x.dtype) * q_step
    meta = {
        'q_step': q_step,
        'scale': scale,
    }
    return x_q, x_qr, meta



def quant_model_dict(args, logger, model_dict, quant_level, excluded_keys=None):
    """
    Quantize the full model.
    """
    # model = unwrap_model(model)
    q_mode = args.compression_cfg.q_mode
    assert q_mode in ['PQ', 'QAT', 'QEC'], f'Invalid quantization mode: {q_mode}'

    # Quantize
    qr_state_dict = copy.deepcopy(model_dict)
    q_state_dict = {}
    
    if q_mode in ["PQ", "QAT"]:
        q_config = {'quant_level': quant_level}
        if quant_level == 32:
            pass
        elif quant_level <= 16:
            for k, v in model_dict.items():
                if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                    q_state_dict[k], qr_state_dict[k], q_config[k] = \
                        _quant_tensor(args, logger, v, quant_level)
        else:
            raise ValueError
        
    elif q_mode == "QEC":
        q_config = {}
        aaaa = model_dict
        for k, v in model_dict.items():
            if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                # print(k)
                q_step  = model_dict[k[:k.rfind('.') + 1] + '0.q_step']
                scale   = torch.clamp_min(model_dict[k[:k.rfind('.') + 1] + '0.scale'], 2e-3)
                q_state_dict[k], qr_state_dict[k], q_config[k] = \
                    _quant_tensor_QEC(args, logger, v, q_step, scale)
        

    # # Load quantisied state dict
    # model.load_state_dict(qr_state_dict)

    # Ruturn configs
    return q_state_dict, q_config, qr_state_dict


def quant_model0(args, logger, model, quant_level):
    """
    Quantize the full model.
    """
    model = unwrap_model(model)
    q_mode = args.compression_cfg.q_mode
    assert q_mode in ['PQ', 'QAT', 'QEC'], f'Invalid quantization mode: {q_mode}'

    # Get the quantisation targets
    excluded_keys = set()

    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix):
            excluded_keys.add(k)

    # Quantize
    qr_state_dict = copy.deepcopy(model.state_dict())
    q_state_dict = {}
    
    if q_mode in ["PQ", "QAT"]:
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
        
    elif q_mode == "QEC":
        q_config = {}
        aaaa = model.state_dict()
        for k, v in model.state_dict().items():
            if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                q_step = model.state_dict()[k[:k.rfind('.') + 1] + '0.q_step']
                q_state_dict[k], qr_state_dict[k], q_config[k] = \
                    _quant_tensor_QEC(args, logger, v, q_step)
        

    # Load quantisied state dict
    model.load_state_dict(qr_state_dict)

    # Ruturn configs
    return q_state_dict, q_config




