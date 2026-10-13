import os
import torch
import math
import threading
import torchvision

import pandas       as pd

from thop           import profile
from easydict       import EasyDict
from cProfile       import Profile
from functools      import partial

from util.initialize_model  import  recover_ckpt, computer_GOP_gap
from util.training          import Trainer, Tester
from util.dataloader        import GlobalDataset, create_data_loader
from util.utils             import log_in_config, create_logger
from model.hnerv_models.hnerv_model_all  import HNeRV
from compression.model_compress   import model_comresssion, model_decomression

from util.hinerv_compress import *

def decompress_worker(args0, group_idx):
    args            = args0
    global_dataset  = args['dataset_cfg']['global_dataset']
    #==========================================  1 execute GPU_TASK ==========================================#
    re_dict = test_group_models(args, group_idx, global_dataset)
    
    #==========================================  2 result record ==========================================#
    gom         = args['dataset_cfg']['gom']
    dec_info    = ''
    for i in range(gom):
        model_idx   = group_idx * gom + i
        if model_idx >= args['dataset_cfg']['model_length']:
            break
        dec_info += '\nmodel_idx: {:<4}   frame_size: {:<4}   decode bpp: {:<8}   psnr: {:<8}dB   ms_ssim: {:<8}  time: {:<8}s    fps: {:<8}'.format(
                    model_idx, 
                    re_dict['frame_size_list'][i],
                    re_dict['dec_bpp_list'][i],
                    re_dict['dec_psnr_list'][i],
                    re_dict['dec_ms_ssim_list'][i],
                    re_dict['dec_time_list'][i],
                    round(re_dict['frame_size_list'][i] / re_dict['dec_time_list'][i], 3),
        )
        
    file_path   = args['log_cfg']['log_dir'] + '/results_decode.csv'
    new_data    = {
        'model_index'   : [str(group_idx * gom + i).zfill(4) for i in range(len(re_dict['dec_psnr_list']))],
        'dec_bpp'       : re_dict['dec_bpp_list'],
        'dec_psnr'      : re_dict['dec_psnr_list'],
        'dec_ms_ssim'   : re_dict['dec_ms_ssim_list'],
        'dec_time'      : re_dict['dec_time_list'],
    }
    avg_info = merge_csv_decode(file_path, new_data)
    return {
        'global_info'   : dec_info + avg_info,
        'frame_size'    : re_dict['frame_size_list'],
        'dec_bpp'       : re_dict['dec_bpp_list'],
        'dec_psnr'      : re_dict['dec_psnr_list'],
        'dec_ms_ssim'   : re_dict['dec_ms_ssim_list'],
        'dec_time'      : re_dict['dec_time_list'],
    }
        
            
def test_group_models(args, group_idx, global_dataset):
    gom = args['dataset_cfg']['gom']
    ref_items  = None
    init_of_ref_items = None
    frame_size_list = []
    dec_bpp_list    = []
    dec_psnr_list   = []
    dec_time_list   = []
    dec_ms_ssim_list= []
    #==========================================  1 training preparation ==========================================#
    # create local_dir and local_logger
    local_dir       = '{}/{}'.format(args['log_cfg']['cfg_dir'], str(group_idx).zfill(4))
    local_model_dir = '{}/models'.format(local_dir)
    local_log_dir   = '{}/log'.format(local_dir)
    local_rec_img_dir= '{}/rec_img'.format(args['log_cfg']['cfg_dir'])
    os.makedirs(local_rec_img_dir,      exist_ok=True)
    local_logger    = create_logger(os.path.join(local_log_dir, 'test_log.txt'), cmd_handle=False, rewirte=True)
    
    #==========================================  3 training models loop ==========================================#
    local_val_loader  = None
    for local_idx in range(gom):
        model_idx       = group_idx * gom + local_idx
        if model_idx >= args['dataset_cfg']['model_length']:
            break
        #-----------------------  3.1 create train_loader and val_loader -----------------------#
        local_dataset   = global_dataset.get_local_dataset(model_idx)
        local_cfg       = {
            'local_logger'  : local_logger,
            'model_dir'     : local_model_dir,
            'log_dir'       : local_log_dir,
            'local_dataset' : local_dataset}
        train_loader, val_loader = create_data_loader(args, local_cfg)
        
        base_path_name      = 'model_'
        save_info_name      = base_path_name + 'train_info_' + str(model_idx).zfill(4) + '.pkl'
        save_model_name     = base_path_name + str(model_idx).zfill(4) + '.pkl'
        quant_item_name     = base_path_name + 'quant_' + str(model_idx).zfill(4) + '.pkl'
        quant_stream_name   = base_path_name + 'stream_' + str(model_idx).zfill(4) + '.bin'
        GOP_gap     = computer_GOP_gap(global_cfg=args, val_loader=val_loader, local_val_loader=local_val_loader)
        #-----------------------  3.2 renew local configs -----------------------#
        local_cfg.update({
            'train_loader'  : train_loader,
            'val_loader'    : val_loader,
            'GOP_gap'       : GOP_gap,
            
            'model_idx'     : str(model_idx).zfill(4),
            'epochs'        : args['train_cfg']['epochs'] if local_idx == 0 else args['train_cfg']['local_epochs'],
            'info_path'     : os.path.join(local_model_dir, save_info_name),
            'model_path'    : os.path.join(local_model_dir, save_model_name),
            'quant_path'    : os.path.join(local_model_dir, quant_item_name),
            'stream_path'   : os.path.join(local_model_dir, quant_stream_name),
            'local_rec_img_dir' : local_rec_img_dir,
            'Model_type'    : ('I' if local_idx == 0 else 'P') if args['train_cfg']['ref_compress'] else 'I',
        })
        local_cfg   = EasyDict(local_cfg)
        # log_in_config(local_logger, local_cfg)
        
        #-----------------------  3.3 train model -----------------------#
        re_dict     = test_one_model(global_cfg=args, local_cfg=local_cfg, 
                                     ref_items=ref_items, init_of_ref_items=init_of_ref_items)
        
        init_of_ref_items = ref_items
        #-----------------------  3.4 result info -----------------------#
        frame_size_list.append(re_dict['frame_size'])
        dec_bpp_list.append(re_dict['dec_bpp'])
        dec_psnr_list.append(re_dict['dec_psnr'])
        dec_ms_ssim_list.append(re_dict['dec_ms_ssim'])
        dec_time_list.append(re_dict['dec_time'])
        ref_items   = re_dict['ref_items']

        # Record all frames of current GOP for next step MSE calculation
        local_val_loader = val_loader
        
        
    return {
        'frame_size_list'   : frame_size_list,
        'dec_bpp_list'      : dec_bpp_list,
        'dec_psnr_list'     : dec_psnr_list,
        'dec_ms_ssim_list'  : dec_ms_ssim_list,
        'dec_time_list'     : dec_time_list,
    }
        
            
# input：global_cfg, local_cfg, init_model_param
# output: train result{bpp, psnr, quanted_model_param}
def test_one_model(global_cfg, local_cfg, ref_items, init_of_ref_items=None):
    logger      = local_cfg.local_logger
    model_idx   = local_cfg.model_idx
    device      = global_cfg.train_cfg.device
    #==========================================  1 Load and decompress model ==========================================#
    from util.initialize_model import create_model, model_init
    model = create_model(global_cfg, local_cfg)
    model = model.to(device) 
    
    args = global_cfg
    args['debug'] = False
    if global_cfg.compression_cfg.q_mode in ['QAT', 'QEC']:
        quant_utils.init_quantization(args, logger, model)
        quant_level = global_cfg.compression_cfg.quant_model_bit
        set_quantization(args, logger, model, quant_level=quant_level, quant_noise=0.9, quant_ste=True)

    model = model_init(global_cfg, local_cfg, model, ref_items, init_of_ref_items)
          
    accelerator = None
    dec_bpp, out_dict = model_decomression(accelerator, global_cfg, local_cfg, model, ref_items)

    #==========================================  2 load model ==========================================#
    model.load_state_dict(out_dict['model_dict'], strict=False)
    
    #==========================================  3 forward ==========================================#
    tester      = Tester(model, global_cfg, local_cfg)
    eval_re2    = tester.eval_only(embedding=out_dict['embed_dict']['embedding'], return_img=False)
    dec_psnr    = eval_re2['metrics']['psnr']
    dec_ms_ssim = eval_re2['metrics']['ms_ssim']
    dec_time    = eval_re2['dec_time']
    
    #==========================================  4 result info ==========================================#
    dec_time    = round(sum(dec_time), 3)
    logger.info(f'decode model: {str(model_idx).zfill(4)}   \tdec_psnr: {dec_psnr}dB, \tdec_ms_ssim: {dec_ms_ssim},' + 
                f'\tdec_bpp: {dec_bpp} \tdec_time: {dec_time}s')
    
    return {
        'frame_size': len(local_cfg.local_dataset),
        'dec_bpp'   : dec_bpp,
        'dec_psnr'  : dec_psnr,
        'dec_time'  : dec_time,
        'dec_ms_ssim':dec_ms_ssim,
        'ref_items' : out_dict if global_cfg['train_cfg']['ref_compress'] else None,
    }
    
            
# storage result into result.csv
def merge_csv_decode(file_path, new_data):
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path, dtype={'model_index': str})
        
    df = pd.DataFrame(new_data)
    
    if os.path.exists(file_path):
        df = pd.concat([df_old, df], ignore_index=True)
        
    df_sorted   = df.sort_values(by='model_index')
    df_sorted.to_csv(file_path, index=False)
    
    data_dict   = df_sorted.to_dict('list')
    
    keys = ['dec_bpp', 'dec_psnr', 'dec_ms_ssim', 'dec_time']
    # eg: 'before_psnr: l_avg(data_dict['before_psnr'])'
    avg_info = f"\navg models info(compute from {len(data_dict['model_index'])} models ):" + \
    ', '.join([f"{key}: {l_avg(data_dict[key])}" for key in keys])
    return avg_info

# get avgerage of list 
def l_avg(lst):
    if not lst:
        return None
    total = sum(lst)
    length = len(lst)
    return round(total / length, 3)

