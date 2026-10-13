"""
判别实验：A0 训练/评估 PSNR 差距（34.0 vs 30.71）的归因。
用同一份浮点快照（state_after_main.pt，对应 full=30.71dB 那次评估的权重）：
  pass 1：帧模式评估（现状口径，interpolate 插值路径）
  pass 2：patch 模式评估（训练读数口径，matmul 插值路径 + 邻域上下文）
解读：
  patch-mode ≈ 34dB → 差距主要来自双前向代码路径差异（训练读数不虚高，口径不同）；
  patch-mode ≈ 30.7dB → 差距主要来自局部拟合 vs 全局组合（整帧连贯性是真实瓶颈）。
用法：
  python scripts/eval_dual_mode.py --exp-dir output/ablation/A0-baseline \
      --dataset /remote-home/gdzj21/dataset/UVG/png --dataset-name ReadySetGo
"""
import os
import sys
import argparse
import types
import yaml
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import create_dataset, create_loader
from models import myhinerbv
from losses import compute_metric
from hinerv_compress import load_bitstream_state


def load_args(exp_dir):
    with open(os.path.join(exp_dir, 'args.yaml')) as f:
        return types.SimpleNamespace(**yaml.safe_load(f))


@torch.no_grad()
def evaluate(model, loader, device):
    psnrs, mses = [], []
    for batch in loader:
        idx, x = batch
        idx = idx.to(device)
        x = x.to(device)
        inp = {'idx': idx, 'idx_max': loader.dataset.num_patches,
               'video_size': loader.dataset.video_size, 'patch_size': loader.dataset.patch_size,
               'batch_size': float(x.shape[0])}
        out = model(inp).clamp(0., 1.).contiguous(memory_format=torch.contiguous_format)
        x = x.contiguous(memory_format=torch.contiguous_format)
        m = compute_metric('psnr', out, x)  # [N, T]
        psnrs.append(m.mean().item())
    return float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--snapshot', default='state_after_main.pt', help='base 目录下的快照文件名')
    ap.add_argument('--device', default='cuda')
    args_cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger('dual')

    args = load_args(args_cli.exp_dir)
    args.dataset = args_cli.dataset
    args.dataset_name = args_cli.dataset_name
    args.model = 'MyHiNeRV'
    args.workers = 2
    args.cached = 'none'
    args.pin_mem = False
    args.eval_batch_size = 1
    args.flow_head = False          # A0 无流头（MODULES_OFF）
    args.multiscale = False
    args.log_eval = False
    args.eval_metric = ['psnr']

    device = torch.device(args_cli.device if torch.cuda.is_available() else 'cpu')

    # 评估数据集（帧模式：每样本一整帧）
    dataset = create_dataset(logger=logger, args=args, training=False)
    loader = create_loader(logger=logger, args=args, training=False, dataset=dataset)

    # 模型（不注册剪枝/量化参数化，直接加载规范化快照）
    sample = {'idx': torch.zeros(1, 3, dtype=int), 'idx_max': dataset.num_patches,
              'video_size': dataset.video_size, 'patch_size': dataset.patch_size, 'batch_size': 1.0}
    model = myhinerbv.build_model(args, logger, sample)
    # 优先加载快照；A0 无快照时回退到 checkpoint_best 的 safetensors（epoch 119 = 主训练结束浮点权重）
    snap_path = os.path.join(args_cli.exp_dir, 'base', args_cli.snapshot)
    if not os.path.isfile(snap_path):
        snap_path = os.path.join(args_cli.exp_dir, args_cli.snapshot)
    if os.path.isfile(snap_path):
        n = load_bitstream_state(model, torch.load(snap_path, map_location='cpu'))
        logger.info(f'loaded {n} tensors from {snap_path}')
    else:
        from safetensors.torch import load_file
        ckpt = os.path.join(args_cli.exp_dir, 'checkpoints', 'checkpoint_best', 'model.safetensors')
        raw = load_file(ckpt)
        # 参数化键 -> 规范化键（去掉 .parametrizations.weight.original 后缀，丢弃 bitwidth 等标量 buffer）
        state = {}
        for k, v in raw.items():
            if '.parametrizations.weight.' in k:
                base, suffix = k.split('.parametrizations.weight.')
                if suffix == 'original':
                    state[base + '.weight'] = v
            elif not k.endswith('.mask'):
                state[k] = v
        n = load_bitstream_state(model, state)
        logger.info(f'loaded {n} tensors from {ckpt} (accelerate checkpoint, canonicalized)')
    model = model.to(device).eval()

    # pass 1：帧模式（评估口径）
    model.eval_patch_size = None
    psnr_frame = evaluate(model, loader, device)
    logger.info(f'PASS1 帧模式  PSNR = {psnr_frame:.3f} dB')

    # pass 2：patch 模式（训练读数口径；patch_to_video 自动重组整帧）
    model.eval_patch_size = list(args.patch_size)
    psnr_patch = evaluate(model, loader, device)
    logger.info(f'PASS2 patch模式 PSNR = {psnr_patch:.3f} dB')

    d = psnr_patch - psnr_frame
    logger.info('=' * 60)
    logger.info(f'差距 patch-frame = {d:+.3f} dB')
    if d > 1.5:
        logger.info('结论：差距主要来自 patch/帧双前向代码路径差异（训练读数口径不同，并非虚高）。')
    elif d < 0.5:
        logger.info('结论：两种前向基本一致 → 训练读数与评估的差距主要来自测量时机/统计口径，'
                    '以及 patch 逐批读数与整帧组合的差异本身也小。')
    else:
        logger.info('结论：两种来源混合（路径差异贡献中等）。')


if __name__ == '__main__':
    main()
