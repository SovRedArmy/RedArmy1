"""
熵编码（算术编码，基于 torchac）：
压缩侧：整数张量 → 统计符号直方图 → 算术编码为字节流，连同 CDF/符号表等元信息保存；
解压侧：读字节流 → 算术解码 → 恢复张量。
compress_and_save_model / decompress_and_load_model 把整套 state_dict
（含剪枝掩码与量化 scale）打包成 zip 码流。
"""
from .utils import *

import torchac
import zlib
import shutil
import tempfile


def _remove_zeros(x, mask):
    """ Remove entries from the tensor with a mask."""
    # 按掩码移除被剪除（0）的元素，只编码非零权重（稀疏存储）
    y = torch.masked_select(x.view(-1), mask.view(-1))
    return y


def _restore_zeros(y, mask):
    """ Restore entries from the tensor with a mask."""
    # _remove_zeros 的逆操作：把非零元素按掩码放回原位置
    idx = torch.cumsum(mask.long().view(-1), dim=0)
    x = F.pad(y.view(-1), (1, 0))[idx].view_as(mask) * mask
    return x


def _arithmetic_encoding(x):
    """ Arithmetic encoding for a tensor with torchac. """
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
    """ Arithmetic decoding for a tensor with torchac. """    
    # 按符号表 sym 与 CDF 解码出整数值，恢复为 shape 形状
    with torch.no_grad():
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(shape), 1), byte_stream).int()
        x_out = sym[inverse_out.int()].view(shape)
        return x_out


def _compress_tensor(args, logger, x, dim, output_dir, name):
    """ Compress a tensor and store its meta data. """
    # 压缩一个整型张量并落盘：可选沿 dim 分组分别编码（当前调用均传 dim=None）
    assert x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

    if args.debug:
        logger.info(f'     Shape: {x.shape}')
        logger.info(f'     Raw size @ 8bit: {math.prod(x.shape)} bytes')
        logger.info(f'     Number of symbols: {len(x.unique())}')

    if x.numel() > 0:
        # arithmetic_encoding requires postive integers
        # 算术编码要求非负整数：减去最小值偏移后再编码
        offset = x.min() if x.numel() > 0 else 0
        x -= offset

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
        # 元信息：符号表/CDF/长度/形状/偏移（与 .bit 字节流配套）
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
        # 空张量：无需码流
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
    """ Decompress a tensor. """
    # 读取元信息与字节流 → 算术解码 → 加回偏移，恢复张量
    # Read meta file
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


def _get_target_keys(model):
    """收集需要写入码流的 state_dict 键（bitstream_prefix 内且非 mask）。"""
    target_keys = []
    for k in model.state_dict().keys():
        if k.startswith(model.bitstream_prefix) and not 'mask' in k:
            target_keys.append(k)
    return target_keys


def _get_mask_key(state_dict, k):
    """找到权重 k 对应的剪枝掩码键（参数化后的 .weight.{i}.mask），最多查 5 层参数化。"""
    if k.endswith('.weight.original'):
        for i in range(5): # just assuming upto 5 parameterizations
            mask_k = k.replace('.weight.original', f'.weight.{i}.mask')
            if mask_k in state_dict:
                return mask_k
    return None


def compress_and_save_model(args, logger, model, output_dir, name, quant_state_dict_int, quant_config):
    """
    Compress and save the model.
    Tensors with quantization will be compressed with arithmetic coding, and the rest will be saved with torch.save.
    压缩并保存整个模型：
    - 量化张量 → 移除剪枝零元素 → 算术编码(.bit 文件 + 元信息)；
    - 其余张量 → 半精度直接存入 state_dict；
    - 全部写入临时目录 → torch.save → zlib → 打包 zip 码流，返回字节数。
    """
    model = unwrap_model(model)

    output_path = os.path.join(output_dir, name)
    temp_dir = tempfile.TemporaryDirectory(dir=output_dir)

    target_keys = _get_target_keys(model)
    state_dict = model.state_dict()
    state_dict_temp = {'quant_config': quant_config}

    # Try to use compress_tensor (arithmetic coding) if possible, otherwise save the state with torch.save
    for k in target_keys:
        v = state_dict[k]

        if args.debug:
            logger.info(f'Saving mask: {k}')
        # 若存在剪枝掩码，则掩码本身也用算术编码保存
        mask_k = _get_mask_key(state_dict, k)
        if mask_k is not None:
            mask = state_dict[mask_k]
            state_dict_temp[k + '_mask'] = _compress_tensor(args, logger, mask.to(torch.int8), None, temp_dir.name, k + '_mask')
        else:
            mask = None

        if args.debug:
            logger.info(f'Saving state: {k}')
        if quant_config is not None and k in quant_config:
            # Only use coding with the quantized tensors, since they have limited number of symbols
            # 量化张量符号有限，算术编码收益最大；先移除被剪掉的零元素
            quant_v = quant_state_dict_int[k] 
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
    # zlib 再压一遍元信息（默认无压缩级别，主要靠算术编码）
    with open(ckpt_path, mode="rb") as f_in:
        with open(compress_ckpt_path, mode="wb") as f_out:
            f_out.write(zlib.compress(f_in.read(), zlib.Z_NO_COMPRESSION))
    os.remove(ckpt_path)

    # Archive the full folder
    shutil.make_archive(output_path, 'zip', temp_dir.name)
    temp_dir.cleanup()

    logger.info(f'Model bitstream saved in {output_path}.zip')
    
    return os.path.getsize(output_path + '.zip')


def decompress_and_load_model(args, logger, model, output_dir, name):
    """
    Decompress and load the model.
    解压并加载模型（compress_and_save_model 的逆过程）：
    解包 zip → zlib 解压 → 逐张量解码（掩码恢复零元素、量化值乘 scale 反量化）
    → load_state_dict 写回模型，返回码流字节数。
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

    # Deompress the state_dict
    with open(compress_ckpt_path, mode="rb") as f_in:
        with open(ckpt_path, mode="wb") as f_out:
            f_out.write(zlib.decompress(f_in.read()))

    # Load the state_dict
    state_dict_temp = torch.load(ckpt_path)
    quant_config = state_dict_temp['quant_config']

    for k in target_keys:
        if args.debug:
            logger.info(f'Loading state: {k}')
        
        device, dtype = model_state_dict[k].device, model_state_dict[k].dtype

        # 解码剪枝掩码
        if (k + '_mask') in state_dict_temp:
            mask = _decompress_tensor(state_dict_temp[k + '_mask'], temp_dir.name, k + '_mask').to(torch.bool).to(device)
        else:
            mask = None

        if k in quant_config:
            # 量化张量：解码整数 → 恢复零元素 → 乘 scale 反量化
            quant_v = _decompress_tensor(state_dict_temp[k], temp_dir.name, k).to(dtype).to(device)
            quant_v = _restore_zeros(quant_v, mask) if mask is not None else quant_v
            v = quant_v.float() * quant_config[k]['scale'].to(quant_v.device).float()
        else:
            # 非量化张量：直接取出并恢复零元素
            v = state_dict_temp[k].to(dtype).to(device)
            v = _restore_zeros(v, mask) if mask is not None else v
        state_dict_out[k] = v

    # Cleanup the full folder
    temp_dir.cleanup()

    # Load weight
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_out, strict=False)

    logger.info(f'Model restored from {output_path}.zip')
    if args.debug:
        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'Unexpected keys: {unexpected_keys}')
        
    return os.path.getsize(output_path + '.zip')