import torch
import torch.utils.data
import os

from util.hinerv_utils import *
import compression
from compression import prune_utils, quant_utils, codec_utils

from model.hnerv_models.hnerv_utils import quant_tensor, dequant_tensor
    
'''
    model compression
    input: model_params_dict, quant_bit, etc.
    output: {original_model_dict, quanted_model_dict}
'''
def model_comresssion(accelerator, global_cfg, local_cfg, model, not_quanted_items, ref_items=None):
    # arguments port
    args        = global_cfg.compression_cfg
    pixel_all   = local_cfg.train_loader.dataset.pixel_all()
    logger      = local_cfg.local_logger
    save_path   = local_cfg.stream_path
    quant_bit   = [args.quant_embed_bit, args.quant_model_bit]
    
    # accelerator = None
    output_dir = save_path
    quant_level= quant_bit[1]
    
    #================= HiNeRV compress:
    model = compression.utils.unwrap_model(model)
    
    # Get the quantisation targets
    excluded_keys = set()
    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix):
            excluded_keys.add(k)
    
    residual_ref(items=not_quanted_items, ref_items=ref_items, excluded_keys=excluded_keys, mode='menu')
    
    quant_state_dict_int, quant_config, qr_state_dict = \
        quant_utils.quant_model_dict(global_cfg, logger, model_dict=not_quanted_items['model_dict'], 
                                quant_level=quant_level, excluded_keys=excluded_keys)
        
    param_nums = 0
    for k, v in quant_state_dict_int.items():
        param_nums += v.numel()
        
    if global_cfg.model_cfg.model == 'HNeRV':
        quant_t, best_new_t = quant_tensor(not_quanted_items['embed_dict']['embedding'], bits=quant_bit[0])
        quant_state_dict_int.update({'embed_dict.embedding': quant_t['quant']})
        quant_config.update({'embed_dict.embedding': {'min': quant_t['min'], 'scale': quant_t['scale']}})
        
    # Compress bitstream
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        num_bytes = codec_utils.compress_and_save_model(global_cfg, logger, model,
                                                        output_dir, f'Q{quant_level}',
                                                        quant_state_dict_int=quant_state_dict_int,
                                                        quant_config=quant_config)
    else:
        num_bytes = 0
    num_bytes, = accelerate.utils.broadcast_object_list([num_bytes])

    num_bpp, out_dict = model_decomression(accelerator, global_cfg, local_cfg, model, ref_items=ref_items)
    return out_dict, round(num_bytes * 8 / param_nums, 3), round(num_bytes * 8 / pixel_all, 3)


def model_decomression(accelerator, global_cfg, local_cfg, model, ref_items=None):
    # arguments port
    args        = global_cfg.compression_cfg
    pixel_all   = local_cfg.train_loader.dataset.pixel_all()
    logger      = local_cfg.local_logger
    save_path   = local_cfg.stream_path
    quant_bit   = [args.quant_embed_bit, args.quant_model_bit]
    
    #######################
    # accelerator = None
    output_dir = save_path
    quant_level= quant_bit[1]
    
    #================= HiNeRV decompress:
    model = compression.utils.unwrap_model(model)
    # Get the quantisation targets
    excluded_keys = set()
    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix):
            excluded_keys.add(k)

    # Decompress bitstream
    if accelerator is None or accelerator.is_main_process:
        num_bytes, embed_quant_t = codec_utils.decompress_and_load_model(global_cfg, logger, model, output_dir, f'Q{quant_level}')
    else:
        num_bytes, embed_quant_t = 0, None
    num_bytes, = accelerate.utils.broadcast_object_list([num_bytes])
    state_dict = accelerate.utils.broadcast(model.state_dict())
    model.load_state_dict(state_dict)
    
    out_dict = {
        'embed_dict': {'embedding': None},
        'model_dict': model.state_dict(),
    }
    if embed_quant_t is not None:
        out_dict['embed_dict']['embedding'] = dequant_tensor(embed_quant_t)
    residual_ref(items=out_dict, ref_items=ref_items, excluded_keys=excluded_keys, mode='add')
    
    return round(num_bytes * 8 / pixel_all, 3), out_dict


def residual_ref(items, ref_items=None, excluded_keys=None, mode='menu'):
    assert mode in ['menu', 'add'], 'invalid mode {mode}'
    def mode_func(t, mode): return t if mode == 'add' else -t
    
    if ref_items is None:
        return
    else:
        with torch.no_grad():
            for k, v in items['model_dict'].items():
                if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                    items['model_dict'][k] += mode_func(ref_items['model_dict'][k], mode)
                    
            if 'embed_dict' in items and items['embed_dict']['embedding'] is not None:
                for k, v in items['embed_dict'].items():
                    items['embed_dict'][k] += mode_func(ref_items['embed_dict'][k], mode)

    