"""
熵编码（算术编码，基于 torchac）—— MyHiNeRV 版。
在 HiNeRV 基础上的改动：
1) _get_target_keys 排除 no_bitstream_prefix（decoder.aux / aux 等训练期辅助头不进码流）；
2) DCVC-RT 风格 scale 档位表：若量化配置里带 scale_index（uint8 档位），
   码流只编码档位索引（算术编码）+ 表头参数，解码端按对数表重建逐通道 scale，
   不再存 half 浮点 scale；
3) P 模型的整数符号本身已是"相对参考模型的残差"（见 quant_utils.quant_model），
   解码端反量化后需要加回参考权重 —— 由 hinerv_compress.decompress 负责，
   本模块只负责（残差）整数的熵编解码。
"""
from .utils import *
from .quant_utils import _canon_key
import math

import torchac
import zlib
import shutil
import tempfile


def _remove_zeros(x, mask):
    """按掩码移除被剪除（0）的元素，只编码非零权重（稀疏存储）。"""
    y = torch.masked_select(x.view(-1), mask.view(-1))
    return y


def _restore_zeros(y, mask):
    """_remove_zeros 的逆操作：把非零元素按掩码放回原位置。"""
    idx = torch.cumsum(mask.long().view(-1), dim=0)
    x = F.pad(y.view(-1), (1, 0))[idx].view_as(mask) * mask
    return x


def _arithmetic_encoding(x):
    """Arithmetic encoding for a tensor with torchac. """
    # 统计符号频率 → CDF → torchac 编码；编码后立即解码校验无损
    with torch.no_grad():
        x = x.detach().view(-1).cpu()
        sym, inverse, counts = x.unique(return_inverse=True, return_counts=True)
        inverse = inverse.to(torch.int16)
        counts = torch.concat([torch.zeros([1], dtype=torch.int64, device=counts.device), counts])
        cdf = torch.cumsum(counts, dim=0).float() / counts.sum().float()
        byte_stream = torchac.encode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), inverse, check_input_bounds=True, needs_normalization=True)
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), byte_stream).int()
        assert inverse_out.equal(inverse)
        x_out = sym[inverse_out.int()]
        assert x_out.equal(x)
        return sym, cdf, byte_stream


def _arithmetic_decoding(byte_stream, sym, cdf, shape):
    """Arithmetic decoding for a tensor with torchac. """
    # 按符号表 sym 与 CDF 解码出整数值，恢复为 shape 形状
    with torch.no_grad():
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(shape), 1), byte_stream).int()
        x_out = sym[inverse_out.int()].view(shape)
        return x_out


def _compress_tensor(args, logger, x, dim, output_dir, name):
    """Compress a tensor and store its meta data. """
    # 压缩一个整型张量并落盘（可选沿 dim 分组分别编码，当前调用均传 dim=None）
    assert x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

    if args.debug:
        logger.info(f'     Shape: {x.shape}')
        logger.info(f'     Raw size @ 8bit: {math.prod(x.shape)} bytes')
        logger.info(f'     Number of symbols: {len(x.unique())}')

    if x.numel() > 0:
        # arithmetic_encoding requires positive integers
        # 算术编码要求非负整数：减去最小值偏移后再编码
        offset = x.min() if x.numel() > 0 else 0
        x = x - offset

        if dim is None:
            # 整张量一次性编码
            sym, cdf, bits = _arithmetic_encoding(x)
            bits_length = len(bits)
        else:
            # 沿 dim 切片分别编码（每片有独立符号表/CDF/字节长度）
            sym = []
            cdf = []
            bits = None
            bits_length = []
            x_splits = torch.split(x, 1, dim=dim)
            for x_i in x_splits:
                sym_i, cdf_i, bits_i = _arithmetic_encoding(x_i.contiguous())
                sym.append(sym_i)
                cdf.append(cdf_i)
                bits = bits + bits_i if bits is not None else bits_i
                bits_length.append(len(bits_i))

        if args.debug:
            logger.info(f'     Compressed: {len(bits)} bytes')
            logger.info(f'     Compression ratio @ 8bit: {len(bits) / (math.prod(x.shape))}')

        # meta data
        meta = {
            'dim': dim,
            'sym': sym,
            'cdf': cdf,
            'bits_length': bits_length,
            'shape': x.shape,
            'offset': offset
        }
    else:
        # meta data
        meta = {
            'dim': None,
            'sym': None,
            'cdf': None,
            'bits_length': 0,
            'shape': None,
            'offset': None
        }

    # Either compress the tensor and return the meta data, or return the tensor
    os.makedirs(output_dir, exist_ok=True)

    # byte_stream
    if x.numel() > 0:
        with open(os.path.join(output_dir, name + '.bit'), 'wb') as f:
            f.write(bits)

    return meta


def _decompress_tensor(state_dict, output_dir, name):
    """Decompress a tensor. """
    # 读取元信息与字节流 → 算术解码 → 加回偏移，恢复张量
    dim = state_dict['dim']
    sym = state_dict['sym']
    cdf = state_dict['cdf']
    bits_length = state_dict['bits_length']
    shape = state_dict['shape']
    offset = state_dict['offset']

    # Read bitstreams
    if bits_length > 0:
        with open(os.path.join(output_dir, name + '.bit'), 'rb') as f:
            byte_stream = f.read()

        if dim is None:
            x = _arithmetic_decoding(byte_stream, sym, cdf, shape)
        else:
            x_splits = []
            bits_count = 0
            for i in range(shape[dim]):
                x_splits.append(_arithmetic_decoding(byte_stream[bits_count:bits_count + bits_length[i]], sym[i], cdf[i], list(shape[:dim]) + [1] + list(shape[dim+1:])))
                bits_count += bits_length[i]
            x = torch.concat(x_splits, dim=dim)

        x += offset.to(x.device)
    else:
        x = torch.zeros([0])

    return x


def _qec_gaussian_cdf(scale: float, R: int) -> torch.Tensor:
    """
    构造 QEC 的离散高斯 CDF 表（UAR-NVC 用 constriction 的 QuantizedGaussian；
    这里用 torchac 等价实现）：符号 −R..R，p_k = Φ((k+0.5)/s) − Φ((k−0.5)/s) 归一化。
    返回 [1, 2R+2] 的 float CDF（torchac 约定）。
    """
    k = torch.arange(-R, R + 1, dtype=torch.float64)
    mu, sigma = 0., max(float(scale), 2e-3)
    cdf_hi = 0.5 * (1 + torch.erf((k + 0.5 - mu) / (sigma * math.sqrt(2))))
    cdf_lo = 0.5 * (1 + torch.erf((k - 0.5 - mu) / (sigma * math.sqrt(2))))
    p = (cdf_hi - cdf_lo).clamp_min(1e-12)
    p = p / p.sum()
    cdf = torch.cumsum(p, dim=0)
    cdf = torch.clamp(cdf, max=1.)  # 防止浮点误差越界
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf])  # [2R+2]
    return cdf[None].float()  # [1, 2R+2]


def _compress_tensor_qec(x, output_dir, name, scale):
    """
    QEC 熵编码（UAR-NVC QEC 的 torchac 实现）：
    整数符号按零均值高斯（std=scale，训练中学习/估计）的 CDF 做算术编码。
    meta 只存 (scale, R, shape) —— 概率模型不需要经验符号表。
    """
    assert x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
    with torch.no_grad():
        orig_shape = tuple(x.shape)
        x = x.detach().view(-1).cpu()
        R = int(x.abs().max().item())
        if x.numel() == 0 or R == 0:
            # 全零张量：无需码流（meta 里 R=0 即可恢复全零）
            meta = {'scale': float(scale), 'R': 0, 'shape': orig_shape, 'bits_length': 0}
            os.makedirs(output_dir, exist_ok=True)
            return meta
        sym = (x + R).to(torch.int16)
        cdf = _qec_gaussian_cdf(scale, R).repeat(sym.numel(), 1)  # [N, 2R+2]
        bits = torchac.encode_float_cdf(cdf, sym, check_input_bounds=True, needs_normalization=True)
        # 解码校验（无损性保证）
        sym_out = torchac.decode_float_cdf(cdf, bits).int()
        assert sym_out.equal(sym.int()), 'QEC roundtrip check failed'
    meta = {'scale': float(scale), 'R': R, 'shape': orig_shape, 'bits_length': len(bits)}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, name + '.bit'), 'wb') as f:
        f.write(bits)
    return meta


def _decompress_tensor_qec(state_dict, output_dir, name):
    """QEC 熵解码：按 (scale, R) 重建高斯 CDF 表后算术解码。"""
    scale = state_dict['scale']
    R = int(state_dict['R'])
    shape = state_dict['shape']
    numel = int(np.prod(shape)) if shape else 0
    if R == 0 or numel == 0:
        return torch.zeros(list(shape) if shape else [0], dtype=torch.int32)
    with torch.no_grad():
        with open(os.path.join(output_dir, name + '.bit'), 'rb') as f:
            byte_stream = f.read()
        cdf = _qec_gaussian_cdf(scale, R).repeat(numel, 1)
        sym_out = torchac.decode_float_cdf(cdf, byte_stream).int()
        x = sym_out - R
    return x.view(list(shape)).to(torch.int32)


def _get_target_keys(model):
    """收集需要写入码流的 state_dict 键（bitstream_prefix 内、非 mask、非 no_bitstream_prefix）。"""
    target_keys = []
    no_prefix = getattr(model, 'no_bitstream_prefix', ())
    for k in model.state_dict().keys():
        if k.startswith(model.bitstream_prefix) and not any(k.startswith(p) for p in no_prefix) and not 'mask' in k:
            target_keys.append(k)
    return target_keys


def _get_mask_key(state_dict, k):
    """找到权重 k 对应的剪枝掩码键（参数化后的 .weight.{i}.mask），最多查 5 层参数化。"""
    if k.endswith('.weight.original'):
        for i in range(5):  # just assuming upto 5 parameterizations
            mask_k = k.replace('.weight.original', f'.weight.{i}.mask')
            if mask_k in state_dict:
                return mask_k
    return None


def _prepare_saved_quant_config(quant_config, state_dict_temp, args, logger, output_dir, model=None):
    """
    整理要落盘的量化配置：
    - 若某键带 scale_index（DCVC-RT 风格档位索引），则把索引也用算术编码保存，
      并从配置中删去 scale 浮点（解码端按表重建）；
    - hyper 模式：额外保存（half 固化的）超网络参数与 z_hat 整数流。
    返回（可能精简过的）quant_config 副本。
    """
    saved = {}
    for k, v in quant_config.items():
        if isinstance(v, dict) and 'scale_index' in v:
            state_dict_temp[k + '_scaleidx'] = _compress_tensor(args, logger, v['scale_index'].to(torch.int16), None, output_dir, k + '_scaleidx')
            v = {kk: vv for kk, vv in v.items() if kk not in ('scale', 'scale_index')}
        saved[k] = v
    if isinstance(quant_config.get('quant_mode'), str) and quant_config.get('quant_mode') == 'hyper' and model is not None:
        model = unwrap_model(model)
        hyper = getattr(model, 'hyper', None)
        if hyper is not None:
            state_dict_temp['hyper_state'] = hyper.transmitted_state()
    return saved


def _rebuild_scale_from_table(cfg_k, state_dict_temp, temp_dir, k):
    """解码端：若有档位索引，则按对数表重建逐通道 scale；否则直接用配置里的 half scale。"""
    if (k + '_scaleidx') in state_dict_temp:
        idx = _decompress_tensor(state_dict_temp[k + '_scaleidx'], temp_dir, k + '_scaleidx').to(torch.uint8)
        table = state_dict_temp['quant_config']['scale_table']
        lo, hi, bits = table['lo'], table['hi'], table['bits']
        n = 2 ** bits
        log_lo, log_hi = math.log(lo), math.log(hi)
        scale = torch.exp(log_lo + idx.float() / (n - 1) * (log_hi - log_lo))
        return scale.half().view(idx.shape).cpu()
    return cfg_k['scale'].cpu()


def compress_and_save_model(args, logger, model, output_dir, name, quant_state_dict_int, quant_config):
    """
    Compress and save the model.
    压缩并保存整个模型：
    - 量化张量 → 移除剪枝零元素 → 算术编码(.bit 文件 + 元信息)；
    - scale 档位索引（若有）→ 算术编码；
    - 其余张量 → 半精度直接存入 state_dict；
    - 全部写入临时目录 → torch.save → zlib → 打包 zip 码流，返回字节数。
    """
    model = unwrap_model(model)

    # hyper 模式：再次（幂等地）固化超网络参数，确保编码侧 synthesize 与
    # 解码端（加载 half 固化值）使用完全一致的 μ/σ（quant_model 末尾的
    # load_state_dict 会把超网络参数恢复到冻结前，必须在此重新固化）
    _hyper = getattr(model, 'hyper', None)
    if _hyper is not None and isinstance(quant_config, dict) and quant_config.get('quant_mode') == 'hyper':
        _hyper.freeze_transmitted()

    output_path = os.path.join(output_dir, name)
    temp_dir = tempfile.TemporaryDirectory(dir=output_dir)

    target_keys = _get_target_keys(model)
    state_dict = model.state_dict()
    state_dict_temp = {}

    # 量化配置（可能把 scale 替换为档位索引编码；hyper 模式附带超网络参数）
    state_dict_temp['quant_config'] = _prepare_saved_quant_config(quant_config, state_dict_temp, args, logger, temp_dir.name, model=model)

    for k in target_keys:
        v = state_dict[k].cpu()  # 熵编码统一在 CPU 进行

        if args.debug:
            logger.info(f'Saving mask: {k}')
        # 若存在剪枝掩码，则掩码本身也用算术编码保存
        mask_k = _get_mask_key(state_dict, k)
        if mask_k is not None:
            mask = state_dict[mask_k].cpu()
            state_dict_temp[k + '_mask'] = _compress_tensor(args, logger, mask.to(torch.int8), None, temp_dir.name, k + '_mask')
        else:
            mask = None

        if args.debug:
            logger.info(f'Saving state: {k}')
        if quant_config is not None and k in quant_config:
            if quant_config[k].get('hyper', False):
                # DCVC-RT 风格超先验：μ/σ 由（half 固化的）超网络从 z_hat 重建
                from .hyperprior import hyper_compress_tensor
                hyper = unwrap_model(model).hyper
                z_int = quant_state_dict_int[k + '__zhat'].cpu()
                mu, sig = hyper.synthesize(_canon_key(k), z_int.float(), tuple(v.shape))
                quant_v = quant_state_dict_int[k].cpu()
                quant_v = _remove_zeros(quant_v, mask) if mask is not None else quant_v
                sig_v = _remove_zeros(sig, mask) if mask is not None else sig
                state_dict_temp[k] = hyper_compress_tensor(quant_v, sig_v, temp_dir.name, k)
                state_dict_temp[k + '__zhat'] = _compress_tensor_qec(z_int, temp_dir.name, k + '__zhat',
                                                                     quant_config[k]['sigma_z'])
            elif quant_config[k].get('qec', False):
                # UAR-NVC QEC：学习到的高斯熵模型编码（meta 只存 scale/R）
                quant_v = quant_state_dict_int[k].cpu()
                quant_v = _remove_zeros(quant_v, mask) if mask is not None else quant_v
                state_dict_temp[k] = _compress_tensor_qec(quant_v, temp_dir.name, k, quant_config[k]['scale'])
            else:
                # Only use coding with the quantized tensors, since they have limited number of symbols
                # 量化张量符号有限，算术编码收益最大；先移除被剪掉的零元素
                quant_v = quant_state_dict_int[k].cpu()
                quant_v = _remove_zeros(quant_v, mask) if mask is not None else quant_v
                state_dict_temp[k] = _compress_tensor(args, logger, quant_v, None, temp_dir.name, k)
        else:
            # 未量化张量：半精度直接保存
            v = _remove_zeros(v, mask) if mask is not None else v
            state_dict_temp[k] = v.half() if v.dtype == torch.float32 else v

    ckpt_path = os.path.join(temp_dir.name, 'model.pth.tar')
    compress_ckpt_path = os.path.join(temp_dir.name, 'model_compressed.pth.tar')

    # Save the state_dict
    torch.save(state_dict_temp, ckpt_path)

    # Compress the state_dict
    with open(ckpt_path, mode="rb") as f_in:
        with open(compress_ckpt_path, mode="wb") as f_out:
            f_out.write(zlib.compress(f_in.read(), zlib.Z_NO_COMPRESSION))
    os.remove(ckpt_path)

    # Archive the full folder
    shutil.make_archive(output_path, 'zip', temp_dir.name)
    temp_dir.cleanup()

    logger.info(f'Model bitstream saved in {output_path}.zip')

    return os.path.getsize(output_path + '.zip')


def decompress_and_load_model(args, logger, model, output_dir, name, reference_state_dict=None):
    """
    Decompress and load the model.
    解压并加载模型（compress_and_save_model 的逆过程）：
    解包 zip → zlib 解压 → 逐张量解码（掩码恢复零元素、量化值乘 scale 反量化，
    P 模型残差模式再加回参考权重）→ load_state_dict 写回模型，返回码流字节数。
    """
    model = unwrap_model(model)

    output_path = os.path.join(output_dir, name)
    temp_dir = tempfile.TemporaryDirectory(dir=output_dir)

    target_keys = _get_target_keys(model)
    state_dict_temp = {}
    state_dict_out = {}
    model_state_dict = model.state_dict()

    ckpt_path = os.path.join(temp_dir.name, 'model.pth.tar')
    compress_ckpt_path = os.path.join(temp_dir.name, 'model_compressed.pth.tar')

    shutil.unpack_archive(output_path + '.zip', temp_dir.name)

    # Decompress the state_dict
    with open(compress_ckpt_path, mode="rb") as f_in:
        with open(ckpt_path, mode="wb") as f_out:
            f_out.write(zlib.decompress(f_in.read()))

    # Load the state_dict
    state_dict_temp = torch.load(ckpt_path)
    quant_config = state_dict_temp['quant_config']
    ref_canon = {_canon_key(k): v for k, v in reference_state_dict.items()} if reference_state_dict is not None else {}

    # hyper 模式：从码流头恢复超网络参数（half → float，与编码端 freeze 后一致）
    hyper = None
    if isinstance(quant_config.get('quant_mode'), str) and quant_config.get('quant_mode') == 'hyper':
        hyper = getattr(unwrap_model(model), 'hyper', None)
        if hyper is None:
            from .hyperprior import WeightHyperprior
            raise RuntimeError('quant_mode=hyper requires the model to have an attached WeightHyperprior (attach_hyperprior) for decoding.')
        hyper.load_transmitted_state(state_dict_temp['hyper_state'])

    for k in target_keys:
        if args.debug:
            logger.info(f'Loading state: {k}')

        device, dtype = model_state_dict[k].device, model_state_dict[k].dtype

        # 解码剪枝掩码（恢复零元素等操作统一在 CPU，最后再搬设备）
        if (k + '_mask') in state_dict_temp:
            mask = _decompress_tensor(state_dict_temp[k + '_mask'], temp_dir.name, k + '_mask').to(torch.bool)
        else:
            mask = None

        if k in quant_config:
            if quant_config[k].get('hyper', False):
                # DCVC-RT 风格超先验：解码 z_hat → g_s 重建 μ/σ → 解码 s → q = s + round(μ)
                from .hyperprior import hyper_decompress_tensor
                z_int = _decompress_tensor_qec(state_dict_temp[k + '__zhat'], temp_dir.name, k + '__zhat')
                mu, sig = hyper.synthesize(_canon_key(k), z_int.float(), tuple(model_state_dict[k].shape))
                s = hyper_decompress_tensor(state_dict_temp[k], sig, temp_dir.name, k)
                s = _restore_zeros(s, mask) if mask is not None else s
                q = s.float() + mu.round().float()
                v = q * float(quant_config[k]['q_step'])
            elif quant_config[k].get('qec', False):
                quant_v = _decompress_tensor_qec(state_dict_temp[k], temp_dir.name, k).to(dtype)
                quant_v = _restore_zeros(quant_v, mask) if mask is not None else quant_v
                v = quant_v.float() * float(quant_config[k]['q_step'])
            else:
                quant_v = _decompress_tensor(state_dict_temp[k], temp_dir.name, k).to(dtype)
                quant_v = _restore_zeros(quant_v, mask) if mask is not None else quant_v
                scale = _rebuild_scale_from_table(quant_config[k], state_dict_temp, temp_dir.name, k)
                v = quant_v.float() * scale.float()
            if quant_config[k].get('residual', False):
                ref_v = ref_canon.get(_canon_key(k))
                assert ref_v is not None, f'missing reference weight for residual decoding: {_canon_key(k)}'
                v = v + ref_v.float().cpu()
        else:
            # 非量化张量：直接取出并恢复零元素
            v = state_dict_temp[k].to(dtype)
            v = _restore_zeros(v, mask) if mask is not None else v
        state_dict_out[k] = v.to(device)

    # Cleanup the full folder
    temp_dir.cleanup()

    # Load weight
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_out, strict=False)

    logger.info(f'Model restored from {output_path}.zip')
    if args.debug:
        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'Unexpected keys: {unexpected_keys}')

    return os.path.getsize(output_path + '.zip')
