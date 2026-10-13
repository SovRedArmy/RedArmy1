import os
import torch
import threading

import pandas       as pd

from easydict       import EasyDict
from cProfile       import Profile
from functools      import partial
from collections    import OrderedDict

from util.training          import  Trainer, Tester
from util.dataloader        import  GlobalDataset, create_data_loader
from util.utils             import  log_in_config, create_logger
from util.initialize_model  import  get_best_initialization, computer_GOP_gap

from model.hnerv_models.hnerv_model_all  import HNeRV
from model.nerv_models.model_nerv   import NeRV
from model.enerv_models.E_NeRV      import ENeRV
from compression.model_compress     import model_comresssion
from compression.quant_utils        import set_quantization, start_quantization, QuantNoise, QuantEntropy


import accelerate
from util.hinerv_compress import *


def compress_worker(args0, group_idx: int):
    args            = args0
    global_dataset  = args['dataset_cfg']['global_dataset']
    #==========================================  1 execute GPU_TASK ==========================================#
    re_dict = train_group_models(args, group_idx, global_dataset)
    
    #==========================================  2 result record ==========================================#
    if 'finished' not in re_dict:
        gom         = args['dataset_cfg']['gom']
        models_info = ''
        for i in range(gom):
            model_idx   = group_idx * gom + i
            if model_idx >= args['dataset_cfg']['model_length']:
                break
            models_info += '\nmodel_idx: {}     models info:  before_psnr: {}, after_psnr: {}, total_bpp: {}, ms_ssim: {}, bp_param: {}'.format(
                            model_idx, 
                            re_dict['before_psnr_list'][i], 
                            re_dict['after_psnr_list'][i], 
                            re_dict['total_bpp_list'][i],
                            re_dict['ms_ssim_list'][i],
                            re_dict['bp_param_list'][i], 
                        )
    
        file_path   = args['log_cfg']['log_dir'] + '/results.csv'
        new_data    = {
            'model_index'   : [str(group_idx * gom + i).zfill(4) for i in range(len(re_dict['before_psnr_list']))],
            'before_psnr'   : re_dict['before_psnr_list'],
            'after_psnr'    : re_dict['after_psnr_list'],
            'total_bpp'     : re_dict['total_bpp_list'],
            'ms_ssim'       : re_dict['ms_ssim_list'],
            'bp_param'      : re_dict['bp_param_list'],
        }
        avg_info = merge_csv(file_path, new_data)
        return {
            'global_info'   : models_info + avg_info,
        }
    else:
        return {
            'global_info'   : '',
        }
        
            
def train_group_models(args, group_idx: int, global_dataset: GlobalDataset):
    gom = args['dataset_cfg']['gom']
    ref_items  = None
    init_of_ref_items = None
    local_val_loader  = None
    # last_dataset = None
    before_psnr_list, after_psnr_list   = [], []
    bp_param_list, total_bpp_list       = [], []
    ms_ssim_list                        = []
    metric_list = ['before_psnr', 'after_psnr', 'total_bpp', 'ms_ssim', 'bp_param']

    #==========================================  1 training preparation ==========================================#
    # create local_dir and local_logger
    local_dir       = '{}/{}'.format(args['log_cfg']['cfg_dir'], str(group_idx).zfill(4))
    local_model_dir = '{}/models'.format(local_dir)
    local_log_dir   = '{}/log'.format(local_dir)
    os.makedirs(local_dir,          exist_ok=True)
    os.makedirs(local_model_dir,    exist_ok=True)
    os.makedirs(local_log_dir,      exist_ok=True)
    local_logger    = create_logger(os.path.join(local_log_dir, 'log.txt'), cmd_handle=False)
    ddp_kwargs = accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator     = accelerate.Accelerator(gradient_accumulation_plugin=accelerate.utils.GradientAccumulationPlugin(
        num_steps=1, adjust_scheduler=False), kwargs_handlers=[ddp_kwargs], mixed_precision='fp16')
    accelerate.utils.set_seed(args.train_cfg.seed, device_specific=True)
    
    #==========================================  2 training models loop ==========================================#
    epochs      = [args['train_cfg']['local_epochs'] for i in range(gom)]
    epochs[0]   = args['train_cfg']['epochs']
    for local_idx in range(gom):
        model_idx       = group_idx * gom + local_idx
        if model_idx >= args['dataset_cfg']['model_length']:
            break
        #-----------------------  2.1 create train_loader and val_loader -----------------------#
        local_dataset   = global_dataset.get_local_dataset(model_idx)
        local_cfg       = {
            'local_logger'  : local_logger,
            'model_dir'     : local_model_dir,
            'log_dir'       : local_log_dir,
            'local_dataset' : local_dataset
        }
        train_loader, val_loader = create_data_loader(args, local_cfg)
        
        base_path_name      = 'model_'
        save_info_name      = base_path_name + 'train_info_' + str(model_idx).zfill(4) + '.pkl'
        save_model_name     = base_path_name + str(model_idx).zfill(4) + '.pkl'
        quant_item_name     = base_path_name + 'quant_' + str(model_idx).zfill(4) + '.pkl'
        quant_stream_name   = base_path_name + 'stream_' + str(model_idx).zfill(4) + '.bin'
        GOP_gap     = computer_GOP_gap(global_cfg=args, val_loader=val_loader, local_val_loader=local_val_loader)
        del local_val_loader
        torch.cuda.empty_cache()
        #-----------------------  2.2 renew local configs -----------------------#
        local_cfg.update({
            'train_loader'  : train_loader,
            'val_loader'    : val_loader,
            'GOP_gap'       : GOP_gap,
            
            'model_idx'     : str(model_idx).zfill(4),
            'epochs'        : epochs[local_idx],
            'info_path'     : os.path.join(local_model_dir, save_info_name),
            'model_path'    : os.path.join(local_model_dir, save_model_name),
            'quant_path'    : os.path.join(local_model_dir, quant_item_name),
            'stream_path'   : os.path.join(local_model_dir, quant_stream_name),
            'Model_type'    : ('I' if local_idx == 0 else 'P') if args['train_cfg']['ref_compress'] else 'I',
        })
        local_cfg   = EasyDict(local_cfg)
        if accelerator.is_main_process:
            log_in_config(local_logger, local_cfg)
        
        #-----------------------  2.3 train model -----------------------#
        re_dict     = train_one_model(accelerator, global_cfg=args, local_cfg=local_cfg, 
                                      ref_items=ref_items, init_of_ref_items=init_of_ref_items)
        
        #-----------------------  2.4 result info -----------------------#
        init_of_ref_items = ref_items
        ref_items           = re_dict['ref_items']
        # eg: before_psnr_list.append(re_dict['before_psnr'])
        for metric in metric_list:
            eval(f"{metric}_list").append(re_dict[metric])

        #-----------------------  2.5 after training -----------------------#
        if local_idx != gom - 1:
            # Record all frames of current GOP for next step MSE calculation
            local_val_loader = val_loader
        # last_dataset = last_dataset
        os.unlink(local_cfg.model_path)
        
    # eg: {'before_psnr_list': before_psnr_list}
    return {k: v for k, v in locals().items() if k in [item + '_list' for item in metric_list]}
        
            
# input：global_cfg, local_cfg, init_model_param
# output: train result{bpp, psnr, quanted_model_param}
def train_one_model(accelerator, global_cfg, local_cfg, ref_items, init_of_ref_items):
    logger      = local_cfg.local_logger
    model_idx   = local_cfg.model_idx
    device      = global_cfg.train_cfg.device
    #==========================================  1 Model init ==========================================#
    from util.initialize_model import create_model, model_init
    model = create_model(global_cfg, local_cfg)
    model = model.to(device)
    
    if accelerator.is_main_process:
        model_info(global_cfg, local_cfg, model)    
    
    # accelerator setting
    model, local_cfg.train_loader, local_cfg.val_loader = accelerator.prepare(
        model, local_cfg.train_loader, local_cfg.val_loader)
    # Set QAT configs
    args = global_cfg
    args['debug'] = False
    if global_cfg.compression_cfg.q_mode in ['QAT', 'QEC']:
        quant_utils.init_quantization(args, logger, model)
        quant_level = global_cfg.compression_cfg.quant_model_bit
        set_quantization(args, logger, model, quant_level=quant_level, quant_noise=0.9, quant_ste=True)
    # set_quantization(args, logger, model, min(args.quant_level), args.quant_noise, args.quant_ste)
    
    model = model_init(global_cfg, local_cfg, model, ref_items, init_of_ref_items)
        
    #==========================================  2 Training ==========================================#
    if global_cfg.compression_cfg.q_mode in ['QAT', 'QEC']:
        start_quantization(model)
    logger.info(f'-' * 100)
    logger.info(f'Start training of model: {str(model_idx).zfill(4)}')
    trainer = Trainer(accelerator, model, global_cfg, local_cfg)
    trainer.loop()

    #==========================================  3 Model quant ==========================================#
    # get param_dict
    model_dict  = torch.load(local_cfg.model_path, map_location='cpu')
    model.load_state_dict(model_dict, strict=False)
    
    # get embedding
    tester      = Tester(model, global_cfg, local_cfg)
    eval_re     = tester.eval_only()
    
    # compress param_dict
    not_quanted_items   = {
        'embed_dict'    : {'embedding': eval_re['embedding']},
        'model_dict'    : OrderedDict((key, value.to(device)) for key, value in model_dict.items()),
    }
    
    # before quant test
    eval_re     = tester.eval_only(embedding=not_quanted_items['embed_dict']['embedding'])
    before_psnr = eval_re['metrics']['psnr']

    # compress
    ref_items   = ref_items if global_cfg['train_cfg']['ref_compress'] else None
    out_dict, bp_param, total_bpp = model_comresssion(accelerator, global_cfg, local_cfg, model, not_quanted_items, ref_items)
    
    # after quant test
    tester      = Tester(model, global_cfg, local_cfg)
    eval_re2    = tester.eval_only(out_dict['embed_dict']['embedding'])
    after_psnr  = eval_re2['metrics']['psnr']
    ms_ssim     = eval_re2['metrics']['ms_ssim']
    #==========================================  4 result info ==========================================#
    logger.info(f'*' * 10)
    logger.info(f'current tiny model: psnr: {after_psnr}, bpp: {total_bpp}, ms_ssim: {ms_ssim}, bp_param: {bp_param}')
    logger.info(f'*' * 10)
    
    ref_items = out_dict
    return_list = ['before_psnr', 'after_psnr', 'total_bpp', 'ms_ssim', 'bp_param'] + ['ref_items']
    # eg: {'before_psnr': before_psnr}
    return {k: v for k, v in locals().items() if k in return_list}

        
# print model structure, parameters nums and flops
def model_info(global_cfg, local_cfg, model):
    logger      = local_cfg.local_logger
    model_idx   = local_cfg.model_idx
    
    if global_cfg.train_cfg.checkpoint == '' and model_idx == str(0).zfill(len(model_idx)):
        logger.info(f'Model info:')
        logger.info(model)
    
        # Compute number of parameters & MACs
        with torch.no_grad():
            logger.info(f'Number of parameters:')
            num_params = model.get_num_parameters()
            for k, v in num_params.items():
                logger.info(f'    {k.ljust(16)}: {v / 10**6:.2f}M')

            # Can only import here, 
            # as this import operation will clear "os.environ['CUDA_VISIBLE_DEVICES']" setting
            try:
                from deepspeed.profiling.flops_profiler import get_model_profile
            except ImportError:
                logger.info('deepspeed not available, skipping FLOPs profiling')
                return
            # Compute MACs
            if global_cfg['model_cfg']['model'] in ['NeRV', 'ENeRV']:
                test_input = next(iter(local_cfg.val_loader))['idx'].to(global_cfg.train_cfg.device)
            elif global_cfg['model_cfg']['model'] == 'HNeRV':
                test_input = next(iter(local_cfg.val_loader))['img'].to(global_cfg.train_cfg.device)
            elif global_cfg.model_cfg.model_name == 'HiNeRV':
                from util.dataloader            import parse_input
                test_input, targets = parse_input(local_cfg.val_loader, next(iter(local_cfg.val_loader)), training=False)
                # test_input = test_input.to(global_cfg.train_cfg.device)
                for key, value in test_input.items():
                    if isinstance(value, torch.Tensor):
                        test_input[key] = value.to(global_cfg.train_cfg.device)
            else:
                NotImplementedError, "error model name!"
                
            with torch.no_grad():
                model.eval()
                _, macs, _ = get_model_profile(model=model, 
                                            args=[test_input],
                                            print_profile=False, detailed=False, warm_up=1, as_string=False)
                if global_cfg['model_cfg']['model'] in ['NeRV', 'ENeRV', 'HNeRV']:
                    macs /= test_input.shape[0]
                elif global_cfg.model_cfg.model_name == 'HiNeRV':
                    macs /= test_input['batch_size']
                else:
                    NotImplementedError, "error model name!"
                # macs /= test_input.shape[0]
            logger.info(f'MACs: {macs / 10 ** 9 :.2f}G')
            
            
# storage result into result.csv
def merge_csv(file_path, new_data):
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path, dtype={'model_index': str})
        
    df = pd.DataFrame(new_data)
    
    if os.path.exists(file_path):
        df = pd.concat([df_old, df], ignore_index=True)
        
    df_sorted   = df.sort_values(by='model_index')
    df_sorted.to_csv(file_path, index=False)
    
    data_dict   = df_sorted.to_dict('list')
    
    keys = ['before_psnr', 'after_psnr', 'total_bpp', 'ms_ssim', 'bp_param']
    # eg: 'before_psnr: l_avg(data_dict['before_psnr'])'
    avg_info = f"\navg models info(compute from {len(data_dict['model_index'])} models ):" + \
    ', '.join([f"{key}: {l_avg(data_dict[key])}" for key in keys])
    return avg_info


# get avgerage of list 
def l_avg(lst, round_n=3):
    if not lst:
        return None
    total = sum(lst)
    length = len(lst)
    return round(total / length, round_n)

