"""
HiNeRV 压缩流程（模型 → 码流）：
compress_bitstream: 固化剪枝 → 整数量化(quant_model) → 算术编码落盘(codec_utils)；
decompress:         解压码流 → 反量化 → 恢复模型权重。
被 hinerv_main.py 在 QAT 评估与 --bitstream 评估时调用。
"""
from utils import *
import compression
from compression import prune_utils, quant_utils, codec_utils
from compression.prune_utils import set_pruning, get_sparsity, PruningMask
from compression.quant_utils import set_quantization, QuantNoise


def initial_parametrizations(args, logger, model):
    """初始化参数化：为所有可剪枝/可量化层注册 PruningMask / QuantNoise 参数化。"""
    prune_utils.init_pruning(args, logger, model)
    quant_utils.init_quantization(args, logger, model)


def fix_parametrizations(model):
    """
    固化剪枝：把剪枝掩码直接乘进原始权重（original.mul_(mask)），
    使导出的 state_dict 中不再依赖参数化即可体现剪枝结果；
    QuantNoise 参数化则跳过（评估模式下它本身就是恒等映射）。
    """
    with torch.no_grad():
        for k, v in model.named_modules():
            if hasattr(v, 'parametrizations'):
                for w, p_list in v.parametrizations.items():
                    for p in p_list:
                        if isinstance(p, QuantNoise):
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


def compress_bitstream(args, logger, accelerator, model, output_dir, quant_level):
    """
    把模型权重压缩成码流并保存为 zip，返回字节数。
    流程：保存浮点 state_dict 副本 → fix_parametrizations 固化剪枝
    → quant_model 量化成整数 → 主进程算术编码+落盘 → 恢复浮点权重。
    """
    model = compression.utils.unwrap_model(model)

    # Save the parametrized model
    state_dict = copy.deepcopy(model.state_dict())

    # Fix parametrizations
    fix_parametrizations(model)

    # Quantize weights into integers
    quant_state_dict_int, quant_config = quant_utils.quant_model(args, logger, model, quant_level)

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

    # Reset parametrizations
    # 恢复压缩前的浮点权重，保证训练可以继续
    model.load_state_dict(state_dict)

    return num_bytes


def decompress(args, logger, accelerator, model, output_dir, quant_level):
    """
    从码流解压并恢复模型权重：主进程解码 → 广播 state_dict → 各进程加载。
    返回码流字节数（用于计算 bpp）。
    """
    model = compression.utils.unwrap_model(model)

    # Decompress bitstream
    if accelerator.is_main_process:
        num_bytes = codec_utils.decompress_and_load_model(args, logger, model, output_dir, f'Q{quant_level}')
    else:
        num_bytes = 0
    num_bytes, = accelerate.utils.broadcast_object_list([num_bytes])
    state_dict = accelerate.utils.broadcast(model.state_dict())
    model.load_state_dict(state_dict)

    return num_bytes


def set_compression_args(parser):
    """注册压缩相关命令行参数：剪枝比例/评分权重、量化级别/噪声/STE。"""
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