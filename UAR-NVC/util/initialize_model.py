import torch
import os

import numpy as np

from copy       import  deepcopy

from util.training          import  Tester
import torch.nn             as nn

from util.hinerv_compress import *
from model.hnerv_models.hnerv_model_all import HNeRV
from model.nerv_models.model_nerv import NeRV
from model.enerv_models.E_NeRV import ENeRV
from compression.quant_utils import _is_quant_target

def model_init(global_cfg, local_cfg, model, ref_items, init_of_ref_items):
    logger      = local_cfg.local_logger
    model_idx   = local_cfg.model_idx
    Model_type  = local_cfg.Model_type
    # device      = global_cfg.train_cfg.device
    seed        = global_cfg.train_cfg.seed + int(model_idx) * 1000
    last_seed   = global_cfg.train_cfg.seed + (int(model_idx) - 1) * 1000
    q_mode      = global_cfg.compression_cfg.q_mode
    
    
    if global_cfg['train_cfg']['init_type']  == 1:
        if Model_type == 'I':
            pass
        else:
            stat_dict = deepcopy(ref_items['model_dict'])
            model.load_state_dict(stat_dict, strict=False)
    elif global_cfg['train_cfg']['init_type']  == 2:
        init_scale = global_cfg.model_cfg.base_grid_init_scale if (
            'base_grid_init_scale' in global_cfg.model_cfg) else 0.001
        if Model_type == 'I':
            initialize_model_params_(model, seed, q_mode, init_scale=init_scale)
        else:
            raw_model = create_model(global_cfg, local_cfg)
            if global_cfg.compression_cfg.q_mode in ['QAT', 'QEC']:
                quant_utils.init_quantization(global_cfg, logger, raw_model)
                set_quantization(global_cfg, logger, raw_model, quant_level=8, quant_noise=0.9, quant_ste=True)
            initialize_model_params_(raw_model, last_seed, q_mode, init_scale=init_scale)  # Initialization method

            ckpt_random = init_of_ref_items['model_dict'] if init_of_ref_items != None else raw_model.state_dict()
            ckpt_last   = ref_items['model_dict']
            model, ckpt_metrics = get_best_initialization(model, ckpt_random, ckpt_last, global_cfg, local_cfg)
            logger.info(ckpt_metrics)
            ref_items['model_dict'] = deepcopy(model.state_dict())
            # for k, v in ref_items['model_dict'].items():
            #     ref_items['model_dict'][k] = v.to(device)
            if global_cfg['train_cfg']['ref_compress']:  # Record only if both initialization and residual encoding are enabled
                quant_utils.mark_init_param(global_cfg, logger, model, model_type=Model_type)
    return model

def initialize_model_params_(model, seed, q_mode='PQ', init_scale=0.001):
    # Cannot guarantee unique results when initializing on GPU due to parallel computation
    torch.manual_seed(seed)
    # Traverse all parameters of the model
    a = dict(model.named_parameters())
    for name, param in model.named_parameters():
        model0 = compression.utils.unwrap_model(model)
        if (q_mode == 'PQ' and _is_quant_target(model0, name, param)) or 'original' in name:
            device = param.device
            param.data = param.data.to('cpu')
            if 'encoding' in name:
                torch.nn.init.uniform_(param.data, -init_scale, init_scale)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            param.data = param.data.to(device)


def create_model(global_cfg, local_cfg):
    logger      = local_cfg.local_logger
    if global_cfg.model_cfg.model_name in ['NeRV', 'ENeRV', 'HNeRV']:
        model = eval(global_cfg.model_cfg.model_name)(global_cfg, local_cfg)
    elif global_cfg.model_cfg.model_name == 'HiNeRV':
        from model.hinerv_models.hinerv import build_model
        from util.dataloader            import parse_input
        img_sample = parse_input(local_cfg.train_loader, next(iter(local_cfg.train_loader)), training=False)
        model = build_model(global_cfg['model_cfg'], logger, img_sample[0])
    else:
        NotImplementedError, "error model name!"
    return model
    

        
def computer_GOP_gap(global_cfg, val_loader, local_val_loader):
    if local_val_loader == None:
        return None
    assert len(val_loader) == len(local_val_loader), "lengths of DataLoader is not equal"

    device          = global_cfg['train_cfg']['device']
    f_mse           = nn.MSELoss()
    
    mse_all     = 0
    for (batch1, batch0) in zip(val_loader, local_val_loader):
        if global_cfg.model_cfg.model_name in ['NeRV', 'ENeRV', 'HNeRV']:
            img0 = batch0['img'].to(device)
            img1 = batch1['img'].to(device)
        elif global_cfg.model_cfg.model_name == 'HiNeRV':
            img0 = batch0[1].to(device)
            img1 = batch1[1].to(device)
        else:
            NotImplementedError, "error model name!"
        
        mse         = f_mse(img0, img1) * (256 ** 2)
        mse_all     += mse
    return mse_all
        
        
'''
input: 随机初始化的模型， 所有参数（特别是路径）
output: 最有的模型初始化结果，各个可行ckpt的选择度量值
'''
def get_best_initialization(model, ckpt_random, ckpt_last, global_cfg, local_cfg):
    device          = global_cfg.train_cfg.device
    init_type       = global_cfg.train_cfg.init_type
    GOP_gap         = local_cfg.GOP_gap
    
    model = compression.utils.unwrap_model(model)
    
    excluded_keys = set()
    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix):
            excluded_keys.add(k)
    
    for k, v in ckpt_random.items():
        ckpt_random[k]  = ckpt_random[k].to(device)
        ckpt_last[k]    = ckpt_last[k].to(device)
    
    if init_type == 1:
        # Version 1: Initialize directly with network parameters trained from last model
        model.load_state_dict(ckpt_last, strict=False)
        choose_info = 'Initialize from last model'
    elif init_type == 2:
        choose_info = ''
        # Random initialization record
        if global_cfg.train_cfg.is_train:
            tester      = Tester(model, global_cfg, local_cfg)
            eval_re     = tester.eval_only()
            choose_info += 'model: ' + 'init' + ' psnr:' + str(eval_re['metrics']['psnr']) + 'dB\n'

        percent     = compute_percent(GOP_gap)
        ckpt        = ckpt_interpolation(ckpt0=ckpt_random, ckpt1=ckpt_last,
                                         percent=percent, excluded_keys=excluded_keys)
        choose_info += 'GOP_gap:' + str(round(GOP_gap.item(), 3)) + '    percent:' + str(round(percent.item(), 3)) + '\n'
        
        model.load_state_dict(ckpt, strict=False)
        if global_cfg.train_cfg.is_train:
            tester.renew_model(model)
            eval_re     = tester.eval_only()
            psnr        = eval_re['metrics']['psnr']
            choose_info += 'percent: ' + str(round(percent.item(), 3)) + ' psnr' + str(psnr) + 'dB\n'
    else:
        raise NotImplementedError("error init_type! only support [0, 1, 2]")
        
    return model, choose_info

def recover_ckpt(ckpt_random, ckpt_last, global_cfg, local_cfg, excluded_keys):
    init_type       = global_cfg.train_cfg.init_type
    GOP_gap         = local_cfg.GOP_gap

    if init_type == 1:
        ckpt    = ckpt_last
    elif init_type == 2:
        percent = compute_percent(GOP_gap)
        ckpt    = ckpt_interpolation(ckpt0=ckpt_random, ckpt1=ckpt_last, 
                                     percent=percent, excluded_keys=excluded_keys)
    return ckpt
        

def ckpt_interpolation(ckpt0, ckpt1, percent=0.5, excluded_keys=()):
    assert percent >= 0. and percent <= 1.
    
    ckpt = deepcopy(ckpt0)
    for k, v in ckpt0.items():
        if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
            ckpt[k] = percent * ckpt0[k] + (1 - percent) * ckpt1[k]
    return ckpt


def compute_percent(gap, low_bound=500, high_bound=4000):
    percent = -0.6 * torch.exp(-2.2e-5 * gap + 0.5) + 1
    # percent = (gap - low_bound) / (high_bound - low_bound)
    torch.clamp_(percent, min=0, max=1)
    return percent




