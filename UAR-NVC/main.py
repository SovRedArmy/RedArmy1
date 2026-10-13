import argparse
import os
import torch
import yaml
import math
import multiprocessing
import concurrent.futures

from tqdm           import tqdm
from easydict       import EasyDict
from copy           import  deepcopy

from util.dataloader    import GlobalDataset
from util.utils         import fix_seed, log_in_config, create_logger, Global_log_config
from util.train_model   import compress_worker, l_avg
from util.test_model    import decompress_worker

from model.hinerv_models import model_cls

def parse_args():
    parser      = argparse.ArgumentParser(description="Example training/testing script")
    args_info   = 'INR video compression, base framework'

    # Default model configuration
    model_name  = 'HiNeRV'
    base_cfg    = './config/base_cfg_hinerv.yaml'
    config      = './config/cfg_hinerv.yaml'

    # Default parameters
    modelsize   = 0.3
    gop         = 6
    epoch_lr    = [1, 1, 0.005]  # [epochs, local_epochs, learning_rate]
    Lambda      = round(5.0 * modelsize / float(gop), 4)

    # Dataset configuration
    # UVG: 'HoneyBee', 'Bosphorus', 'Beauty', 'YachtRide', 'Jockey', 'ReadySetGo'
    # VVC ClassB: 'BasketballDrive', 'BQTerrace', 'Cactus', 'MarketPlace', 'RitualDance'
    dataset     = 'BasketballDrive'
    data_path   = './data/VVC_ClassB500/' + dataset

    # Initialization and compression settings
    init_type       = 2  # 0: no init, 1: auto_regressive init, 2: choose init
    ref_compress    = True

    # Output directory
    base_dir    = 'train_result'

    # Checkpoint for continuing training or testing
    checkpo     = ''
    is_train    = True
    
    parser.add_argument("--exp_info",       type=str,   default=args_info)
    parser.add_argument('--base_cfg',         type=str,   default=base_cfg)
    parser.add_argument('--config',         type=str,   default=config)
    parser.add_argument("--model_name",     type=str,   default=model_name)
    parser.add_argument('--checkpoint',     type=str,   default=checkpo)
    
    parser.add_argument("--dataset",        type=str,   default=dataset)
    parser.add_argument("--data_path",      type=str,   default=data_path)
    
    parser.add_argument("--is_train",       type=bool,  default=is_train)
    parser.add_argument("--base_dir",       type=str,   default=base_dir)
    
    parser.add_argument("--init_type",      type=float, default=init_type, 
                        help='0 for no init, 1 for auto_regressive init, 2 for choose init')
    parser.add_argument("--ref_compress",   type=bool,  default=ref_compress)
    
    parser.add_argument("--gop",            type=int,   default=gop)
    parser.add_argument("--gom",            type=int,   default=5)
    
    parser.add_argument("--epochs",         type=int,   default=epoch_lr[0])
    parser.add_argument("--local_epochs",   type=int,   default=epoch_lr[1])
    parser.add_argument("--lr",             type=float, default=epoch_lr[2])
    
    parser.add_argument("--modelsize",      type=float, default=modelsize)
    parser.add_argument("--Lambda",         type=float, default=Lambda)
    parser.add_argument("--q_mode",         type=str, default='QEC')
    parser.add_argument("--group-only",     type=int, default=-1,
                        help='[repro patch] only run this group idx (-1 = all)')
    
    args = parser.parse_args()
    return args

def main():
    #==========================================  1 Merge configs ==========================================#
    args = parse_args()
    with open(args.base_cfg, 'r') as f:
        base_cfg    = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config, 'r') as f:
        config_add = yaml.load(f, Loader=yaml.FullLoader)
    
    # merge config 1
    for k, v in config_add.items():
        if k in base_cfg.keys():
            base_cfg[k].update(v)
        else:
            base_cfg[k] = v
    config_dict = deepcopy(base_cfg)
    # merge config 2
    config_dict.update({
                                    'exp_info'      : args.exp_info,
                                    'config'        : args.config,
    })
    config_dict['compression_cfg'].update({
                                    'q_mode'        : args.q_mode,
                                    })
    config_dict['model_cfg'].update({
                                    'model_name'    : args.model_name,
                                    'modelsize'     : args.modelsize,
                                    })
    config_dict['dataset_cfg'].update({
                                    'dataset'       : args.dataset,
                                    'data_path'     : args.data_path,
                                    'gop'           : args.gop,
                                    'gom'           : args.gom,
                                    })
    config_dict['train_cfg'].update({
                                    'checkpoint'    : args.checkpoint,
                                    'device'        : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                    'is_train'      : args.is_train,
                                    'init_type'     : args.init_type,
                                    'ref_compress'  : args.ref_compress,
                                    'epochs'        : args.epochs,
                                    'local_epochs'  : args.local_epochs,
                                    'Lambda'        : args.Lambda,
                                    'lr'            : args.lr,
                                    'group_only'    : args.group_only,
        })
    config_dict['log_cfg'].update({'base_dir'       : args.base_dir})
    args = EasyDict(config_dict)

    # Adjust model size via channels_nums
    if args['model_cfg']['model_name'] == 'HiNeRV':
        gop = args['dataset_cfg']['gop']
        args['model_cfg']['base_grid_size'][0] = max(1, gop // 4)
        if args['model_cfg']['base_grid_size'][0] < 2:
            args['model_cfg']['base_grid_level_scale'][0] = 1
        
        if gop < 4:
            args['model_cfg']['enc_grid_level_scale'][0] = 1
        
        args['model_cfg']['channels'] = int(math.sqrt(3e4 * args['model_cfg']['modelsize']))
    elif args['model_cfg']['model_name'] == 'NeRV':
        ms_scale =  math.sqrt(3.1e-1 * args['model_cfg']['modelsize'])
        args['model_cfg']['lower_width'] = int(args['model_cfg']['lower_width'] * ms_scale)
        args['model_cfg']['stem_dim_num'] = str(int(512 * ms_scale)) + '_1'
        args['model_cfg']['fc_hw_dim']  = '9_16_' + str(int(26 * ms_scale))
    elif args['model_cfg']['model_name'] == 'ENeRV':
        ms_scale =  math.sqrt(4.9e-2 * args['model_cfg']['modelsize'])
        args['model_cfg']['lower_width']    = int(args['model_cfg']['lower_width'] * ms_scale)
        args['model_cfg']['stem_dim_num']   = str(int(512 * ms_scale))
        args['model_cfg']['fc_hw_dim']      = '9_16_' + str(int(196 * ms_scale))
        args['model_cfg']['mlp_dim']        = int(128 * ms_scale)
        args['model_cfg']['head_dim']       = int(64 * ms_scale)
        args['model_cfg']['block_dim']      = int(256 * ms_scale)
    else:
        raise NotImplementedError(f"Model {args['model_cfg']['model_name']} not implemented")

    # No reference encoding for init_type=0
    assert args['train_cfg']['init_type'] in [0, 1, 2]
    if args['train_cfg']['init_type'] == 0:
        args['train_cfg']['ref_compress'] = False
    if args['compression_cfg']['q_mode'] in ['PQ', 'QAT']:
        assert args['train_cfg']['init_type'] in [0, 1], "Interpolation init does not support Post Quant and QAT"

    #==========================================  2 Training preparation ==========================================#
    fix_seed(args['train_cfg']['seed'])

    # Create logger
    log_config      = Global_log_config(args)
    args            = log_config.get_dict(args)
    if args['train_cfg']['is_train']:
        global_logger   = create_logger(os.path.join(args['log_cfg']['log_dir'], 'log.txt'))
    else:
        global_logger   = create_logger(os.path.join(args['log_cfg']['log_dir'], 'test_log.txt'), rewirte=True)
    args['train_cfg'].update({'global_logger': global_logger})

    # Divide dataset
    global_dataset  = GlobalDataset(main_dir=args['dataset_cfg']['data_path'], gop=args['dataset_cfg']['gop'])
    t_length        = global_dataset.get_len()
    model_length    = math.ceil(t_length        / args['dataset_cfg']['gop'])
    group_length    = math.ceil(model_length    / args['dataset_cfg']['gom'])
    args['dataset_cfg'].update({
                                'global_dataset': global_dataset,
                                't_length'      : t_length,
                                'model_length'  : model_length,
                                'group_length'  : group_length,
        })
    log_in_config(global_logger, args)

    #==========================================  3 Multiple INR model pipeline ==========================================#
    if args['train_cfg']['is_train']:
        worker  = compress_worker
    else:
        worker  = decompress_worker

    # [repro patch] --group-only: 只跑指定组(GOM 分组天然独立, 多卡并行时每卡一组);
    # 未指定(-1)时行为与原版完全一致
    _go = args['train_cfg'].get('group_only', -1)

    results = []
    pbar = tqdm(total=group_length)
    for group_idx in range(group_length):
        if _go >= 0 and group_idx != _go:
            pbar.update(1)
            continue
        result = worker(args, group_idx)
        results.append(result)
        global_logger.info(result['global_info'])
        pbar.update(1)

if __name__ == "__main__":
    main()