"""
    Generate a cfg object according to a cfg file and args, then spawn Trainer(rank, cfg).
    Fast-NeRV 训练总入口（NeRV-Enc 超网络的训练）：
    main() -> parse_args() 解析命令行 -> make_cfg() 合并 yaml 配置与 --opts 覆盖项
    -> main_worker() 按配置名实例化 trainer（enc_trainer）并调用 trainer.run() 开始训练。
    多 GPU 时用 mp.spawn 为每张卡各启动一个 main_worker（DDP）。
"""

import argparse
import os
from mergedeep import merge

import yaml
import torch
import torch.multiprocessing as mp
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import utils
import trainers
import numpy as np
import random

torch.use_deterministic_algorithms(True, warn_only=True) # for cudnn deterministic
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def parse_args(args=None):
    """解析命令行参数：--cfg 指定 yaml 配置，--opts 以 key value 对覆盖配置项，
    其余为数据路径、帧数 frame_num、输入尺寸 input_size、batch_size 等训练设置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--data_path', default='data/k400')
    parser.add_argument('--csv_file', default='k400_train.js')
    parser.add_argument('--t_duration', type=float, default=1.0)
    parser.add_argument('--frame_num', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--num_workers', '-j', type=int, default=16)
    parser.add_argument('--out_path', default='output path')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--port_offset', '-p', type=int, default=0)
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    parser.add_argument('--opts', type=str, nargs='*', default=[], help='cfg args to update')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    return args

def make_cfg(args):
    """读取 yaml 配置，把形如 '$frame_num$' 的占位符替换为命令行参数值，
    再按 '--opts key.subkey value' 的形式逐项覆盖配置，最后生成实验名与保存目录。"""
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        """递归遍历配置字典，把 '$xxx$' 字符串替换成 args 中同名参数的值。"""
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                if v.startswith('$') and v.endswith('$'):
                    v = getattr(args, v.replace('$', ''))
                d[k] = v
    translate_cfg_(cfg)

    if args.tag != '':
        args.tag = args.tag + '_'

    env = dict()
    tot_gpus = torch.cuda.device_count()
    env['tot_gpus'] = tot_gpus
    env['cudnn'] = args.cudnn
    env['wandb_upload'] = args.wandb_upload
    cfg['env'] = env

    def build_tree(tree_list):
        """把 ['a','b','c', value] 这样的键列表递归构建成嵌套 dict {'a': {'b': {'c': value}}}。"""
        if len(tree_list)>=2:
            return {tree_list[0]: build_tree(tree_list[1:]) if len(tree_list)>2 else tree_list[-1]}

    def nested_v(dict, keys):
        """按键路径 keys 逐层下钻，取出配置中当前值（用于推断覆盖项的目标类型）。"""
        for key in keys:
            dict = dict[key]
        return dict

    def convert(type, x):
        """把命令行字符串值转换成配置中对应字段的类型（如 'True' -> bool）。"""
        if type == bool and isinstance(x, str):
            if x.lower() == 'true':
                return True
            elif x.lower() == 'false':
                return False
            else:
                raise ValueError('Cannot convert {} to bool'.format(x))
        else:
            return type(x)

    # 逐个应用 --opts 覆盖项：先按原配置类型转换值，再 merge 回配置树
    assert len(args.opts) % 2 == 0
    for cur_cfg_key, v in zip(args.opts[::2], args.opts[1::2]):
        keys = cur_cfg_key.split('.')
        cur_v = nested_v(cfg, keys)
        try:
            v = convert(type(cur_v), v)
        except:
            raise AttributeError(f'{keys}: {cur_v}, {v}')
        cfg = merge(cfg, build_tree(keys+[v]))

    env['exp_name'] = '{}G{}_F{}_t{}_Size{}_B{}_E{}_lr{}_{}_N{}_{}'.format(
        args.tag, tot_gpus, args.frame_num, args.t_duration, args.input_size, 
        args.batch_size, cfg['max_epoch'],
        cfg['optimizer']['args']['lr'], 
        cfg['optimizer']['lr_type'],
        cfg['train_dataset']['args']['cls_vid_num'], 
        utils.get_exp_name(cfg))
    env['save_dir'] = os.path.join(args.out_path, env['exp_name'])
    env['port'] = 10000 + args.port_offset    
    cfg['manualSeed'] = args.manualSeed

    return cfg


def main_worker(rank, cfg):
    """单个进程的训练入口：固定随机种子保证可复现，然后按 cfg['trainer'] 名字
    从 trainers_dict 取出 trainer 类（如 'enc_trainer'）实例化并 run()。"""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    manualSeed = cfg['manualSeed'] 
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # [复现修复] 严格确定性模式会强制 backward 走慢速 fallback 核（step 耗时约 10 倍），
    # warn_only 保持随机种子级别的可复现性，同时不影响训练结果，仅提升速度：
    torch.use_deterministic_algorithms(True, warn_only=True) # for cudnn deterministic

    trainer = trainers.trainers_dict[cfg['trainer']](rank, cfg)
    trainer.run()


def main():
    """程序入口：解析参数、生成配置、创建输出目录；多卡用 mp.spawn 启动 DDP，单卡直接运行。"""
    args = parse_args()

    cfg = make_cfg(args)
    utils.ensure_path(cfg['env']['save_dir'], args.replace)

    if cfg['env']['tot_gpus'] > 1:
        mp.spawn(main_worker, args=(cfg,), nprocs=cfg['env']['tot_gpus'])
    else:
        main_worker(0, cfg)


if __name__ == '__main__':
    main()
