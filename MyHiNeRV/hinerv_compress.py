"""
MyHiNeRV 压缩流程（模型 → 码流）：
- I 模型（intra）：HiNeRV 原路径 —— 剪枝固化 → 整数量化 → 算术编码；
- P 模型（inter，DCVC-RT 借鉴）：量化对象是相对参考模型（已解码的上一 clip 模型）
  的残差权重（quant_utils.quant_model 的 residual 路径），解码端加回参考；
- decompress 支持 reference_state_dict（P 模型解码）。
被训练主循环（hinerv_main.py / myhinerbv_main.py）调用。
"""
from utils import *
import compression
from compression import prune_utils, quant_utils, codec_utils
from compression.prune_utils import set_pruning, get_sparsity, PruningMask
from compression.quant_utils import (set_quantization, set_reference, start_quantization,
                                     QuantNoise, ResidualQuantNoise, QuantEntropy)


def initial_parametrizations(args, logger, model):
    """初始化参数化：为所有可剪枝/可量化层注册 PruningMask / 量化参数化（noise/qec 模式）。"""
    prune_utils.init_pruning(args, logger, model)
    quant_utils.init_quantization(args, logger, model)


def fix_parametrizations(model):
    """
    固化剪枝：把剪枝掩码直接乘进原始权重（original.mul_(mask)），
    使导出的 state_dict 中不再依赖参数化即可体现剪枝结果；
    量化参数化跳过（评估模式下它们本身就是恒等映射）。
    """
    with torch.no_grad():
        for k, v in model.named_modules():
            if hasattr(v, 'parametrizations'):
                for w, p_list in v.parametrizations.items():
                    for p in p_list:
                        if isinstance(p, (QuantNoise, ResidualQuantNoise, QuantEntropy)):
                            pass
                        elif isinstance(p, PruningMask):
                            p_list.original.mul_(p.mask)
                        else:
                            raise NotImplementedError


def set_zero(model):
    """把模型全部权重置零（mask 除外，置 1）——用于码流解压前的校验。"""
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k.endswith('.mask'):
            v.fill_(1.)
        else:
            v.zero_()


@torch.no_grad()
def snapshot_model_state(model):
    """
    阶段快照：保存"计算后"的规范化权重（剪枝掩码已烘焙进权重），
    主训练/剪枝阶段一结束立即调用，供后续失败重跑时直接加载。
    与 get_bitstream_state_dict 的区别：parametrized 权重取 forward 计算值
    （mask×original），掩码信息不丢失。
    """
    from compression.quant_utils import _iter_quant_parametrizations, _canon_key
    model = compression.utils.unwrap_model(model)
    no_prefix = getattr(model, 'no_bitstream_prefix', ())
    computed = {}
    was_training = model.training
    model.eval()  # 量化噪声参数化在 eval 下恒等，快照纯浮点权重
    for owner_path, weight_name, p in _iter_quant_parametrizations(model):
        computed[f'{owner_path}.{weight_name}'] = model.get_submodule(owner_path).weight.detach().clone()
    if was_training:
        model.train()
    out = {}
    for k, v in model.state_dict().items():
        if not k.startswith(model.bitstream_prefix) or any(k.startswith(pp) for pp in no_prefix) or k.endswith('.mask'):
            continue
        if '.parametrizations.' in k and not k.endswith('.original'):
            continue
        ck = _canon_key(k)
        out[ck] = computed.get(ck, v.detach().clone())  # 参数化权重优先取计算值
    return out


@torch.no_grad()
def load_bitstream_state(model, canon_state):
    """
    把（快照/解码的）规范化权重加载进模型：参数化权重写回对应 original，
    其余键直接加载。返回成功加载的张量数。
    """
    from compression.quant_utils import _canon_key
    model = compression.utils.unwrap_model(model)
    target = model.state_dict()
    mapped = {}
    for k, v in canon_state.items():
        if k in target and target[k].shape == v.shape:
            mapped[k] = v
        else:
            for tk in target:
                if tk.endswith('.original') and _canon_key(tk) == k and target[tk].shape == v.shape:
                    mapped[tk] = v
                    break
    model.load_state_dict(mapped, strict=False)
    return len(mapped)


def get_bitstream_state_dict(model):
    """
    导出干净的（规范化键名）码流权重表：
    - 参数化权重取 original（已含剪枝结果），并规范化为"宿主.权重"键名；
    - 排除 mask 与训练期辅助头（no_bitstream_prefix）；
    该表作为 P 模型的参考（时间上下文）与跨模型权重复用的统一格式。
    """
    from compression.quant_utils import _canon_key
    model = compression.utils.unwrap_model(model)
    no_prefix = getattr(model, 'no_bitstream_prefix', ())
    out = {}
    for k, v in model.state_dict().items():
        if not k.startswith(model.bitstream_prefix) or any(k.startswith(p) for p in no_prefix) or k.endswith('.mask'):
            continue
        if '.parametrizations.' in k and not k.endswith('.original'):
            continue  # 跳过参数化内部 buffer（bitwidth/noise_ratio/deadzone 等）
        out[_canon_key(k)] = v.detach().clone()
    return out


def compress_bitstream(args, logger, accelerator, model, output_dir, quant_level, reference_state_dict=None):
    """
    把模型权重压缩成码流并保存为 zip，返回字节数。
    reference_state_dict=None：I 模型 intra 编码；否则 P 模型对残差编码。
    """
    model = compression.utils.unwrap_model(model)

    # Save the parametrized model
    state_dict = copy.deepcopy(model.state_dict())

    # Fix parametrizations
    fix_parametrizations(model)

    # Quantize weights into integers (intra or residual vs the reference model)
    quant_state_dict_int, quant_config = quant_utils.quant_model(args, logger, model, quant_level,
                                                                 reference_state_dict=reference_state_dict)

    # Compress bitstream
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        num_bytes = codec_utils.compress_and_save_model(args, logger, model,
                                                        output_dir, f'Q{quant_level}',
                                                        quant_state_dict_int=quant_state_dict_int,
                                                        quant_config=quant_config)
    else:
        num_bytes = 0
    num_bytes, = accelerate.utils.broadcast_object_list([num_bytes])

    # Reset parametrizations（恢复压缩前的浮点权重，保证训练可以继续）
    model.load_state_dict(state_dict)

    return num_bytes


def decompress(args, logger, accelerator, model, output_dir, quant_level, reference_state_dict=None):
    """
    从码流解压并恢复模型权重：主进程解码 → 广播 state_dict → 各进程加载。
    P 模型（残差码流）必须提供 reference_state_dict（已解码的上一模型）。
    返回码流字节数。
    """
    model = compression.utils.unwrap_model(model)

    # Decompress bitstream
    if accelerator.is_main_process:
        num_bytes = codec_utils.decompress_and_load_model(args, logger, model, output_dir, f'Q{quant_level}',
                                                          reference_state_dict=reference_state_dict)
    else:
        num_bytes = 0
    num_bytes, = accelerate.utils.broadcast_object_list([num_bytes])
    state_dict = accelerate.utils.broadcast(model.state_dict())
    model.load_state_dict(state_dict)

    return num_bytes


def set_compression_args(parser):
    """注册压缩相关参数：剪枝、量化（含 P 模型残差量化与 DCVC-RT 风格 scale 档位）。"""
    group = parser.add_argument_group('Model compression parameters')
    group.add_argument('--prune-ratio', default=0., type=float,
                        help='Pruning ratio for each time of pruning')
    group.add_argument('--prune-weight', default=0., type=float,
                        help='Weight for computing pruning scores')
    group.add_argument('--quant-level', default=[8], type=int, nargs='+',
                        help='Quantization level (default: 8)')
    group.add_argument('--quant-noise', default=0.9, type=float,
                        help='Quantization noise ratio (default: 0.1)')
    group.add_argument('--quant-ste', default=False, type=str_to_bool,
                        help='Quantization with ste')
    group.add_argument('--quant-mode', default='noise', type=str, choices=['noise', 'qec', 'hyper'],
                        help='Quantization mode: noise (HiNeRV Quant-Noise QAT) / qec (UAR-NVC QEC: per-layer learnable q_step + scalar Gaussian entropy) / hyper (DCVC-RT-style mini hyperprior: per-element mu/sigma entropy coding, overfitted per model).')
    group.add_argument('--hyper-embed-dim', default=16, type=int, help='layer embedding dim of the weight hyperprior.')
    group.add_argument('--hyper-hidden', default=64, type=int, help='hidden width of the weight hyperprior MLP.')
    group.add_argument('--hyper-down', default=8, type=int, help='downsample factor of the weight hyperprior g_a.')
    group.add_argument('--quant-residual', default=False, type=str_to_bool,
                        help='Quantize the residual w.r.t. the reference (previous decoded) model weights (P-model).')
    group.add_argument('--quant-residual-deadzone', default=1, type=int,
                        help='Dead-zone for residual quantization: residuals with |q|<dz are zeroed and skipped (DCVC-RT force-zero style). 1: disabled')
    group.add_argument('--compression-interval', default=0, type=int,
                        help='periodic compression probe every N epochs (prune+QAT+compress+eval). 0: disabled.')
    group.add_argument('--probe-batches', default=100, type=int,
                        help='number of QAT batches per compression probe (speed vs accuracy tradeoff).')
    group.add_argument('--quant-scale-bits', default=0, type=int,
                        help='Snap per-channel quant scales to a log-uniform table with 2^bits levels and code the indices (DCVC-RT scale-table style). 0: disabled')
