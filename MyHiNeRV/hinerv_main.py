"""
MyHiNeRV 单模型训练入口（调试/基线用）：在整段视频（或 --frame-range 指定的 clip）上
训练一个 MyHiNeRV 模型并走完整 剪枝→量化→熵编码 流程。
多 clip 的 I/P 管线请用 myhinerbv_main.py。

用法示例：
  python hinerv_main.py --model MyHiNeRV --dataset <root> --dataset-name <name> \
      --output output --frame-range 0 300 --init-from probe_state.pt \
      --quant-residual true --ref-model prev_decoded.pt ...
"""
from utils import *
from datasets import create_dataset, create_loader, set_dataset_args, compute_content_features
from hinerv_tasks import VideoRegressionTask, set_task_args
from hinerv_compress import set_compression_args, initial_parametrizations, get_bitstream_state_dict
from dynamics import set_dynamics_args, PatchDynamicsTracker
from train_lib import train_model, eval_step
from compression.hyperprior import attach_hyperprior

import argparse

parser = argparse.ArgumentParser()

# Script
group = parser.add_argument_group('Script parameters')
group.add_argument('--model', default='MyHiNeRV', type=str, help='Model to build, e.g. MyHiNeRV.')
group.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
group.add_argument('--eval-epochs', type=int, default=30, help='Number of epochs for every evaluation.')
group.add_argument('--log-epochs', type=int, default=30, help='Number of epochs for every output logging.')
group.add_argument('--prune-epochs', default=0, type=int, help='Number of epochs with pruning.')
group.add_argument('--prune-warmup-epochs', default=0, type=int, help='Number of warmup epochs with pruning.')
group.add_argument('--prune-lr-scale', default=1., type=float, help='Learning rate scale with pruning.')
group.add_argument('--quant-epochs', default=0, type=int, help='Number of epochs with QAT.')
group.add_argument('--quant-warmup-epochs', default=0, type=int, help='Number of warmup epochs with QAT.')
group.add_argument('--quant-lr-scale', default=0.1, type=float, help='Learning rate scale with QAT')
group.add_argument('--output', default='output', type=str, help='Path to output folder.')
group.add_argument('--exp-name', default=None, type=str, help='Suffix of the output folder.')
group.add_argument('--debug', action='store_true', default=False, help='Set debug mode and watch additional outputs.')
group.add_argument('--eval-only', action='store_true', default=False, help='Skip the training stage.')
group.add_argument('--resume', default='', type=str, help='Checkpoint path for resuming training.')
group.add_argument('--grad-accum', type=int, default=1, help='Number of gradient accumulation steps.')
group.add_argument('--seed', type=int, default=0, help='Random seed.')

# MyHiNeRV single-model
group.add_argument('--frame-range', type=int, nargs=2, default=None, metavar=('T0', 'T1'),
                   help='train on frames [T0, T1) only (a single clip).')
group.add_argument('--init-from', default='', type=str,
                   help='warm-start weights from a canonical state dict (e.g. probe model).')
group.add_argument('--save-state', default='', type=str,
                   help='path to save the final (canonical) state dict for warm-start/residual reference.')
group.add_argument('--ref-model', default='', type=str,
                   help='canonical state dict of the reference (previously decoded) model, for P-model residual coding.')
group.add_argument('--use-flow', type=str_to_bool, default=True,
                   help='attach RAFT pseudo-label flow to the training set if available.')
group.add_argument('--profile', action='store_true', default=False, help='Run profiler.')

# Dataset
set_dataset_args(parser)

# Task
set_task_args(parser)

# Training dynamics
set_dynamics_args(parser)

# Optimizer & scheduler
set_optimizer_args(parser)

# Compression
set_compression_args(parser)


def load_compatible_state(model, state_dict, logger):
    """
    把（规范化键名的）state_dict 加载进模型，自动跳过形状不匹配的键
    （clip 帧数不同的 CRSPE/时间网格等重新初始化）。
    """
    model = unwrap_model(model)
    target = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in target and target[k].shape == v.shape and target[k].dtype == v.dtype:
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(f'Warm start: loaded {len(filtered)} tensors, skipped {len(skipped)} '
                f'(missing {len(missing)}, unexpected {len(unexpected)}).')
    return len(filtered)


def main():
    """单模型训练主函数（详见模块 docstring）。"""
    start_time = datetime.datetime.now()

    args = parse_args(parser)

    accelerator, logger, output_dir = get_accelerator_logger(args)
    logger.info(f'Output dir: {output_dir}')

    logger.info(f'Set seed: {args.seed}')
    accelerate.utils.set_seed(args.seed, device_specific=True)

    # Task setting
    train_task = VideoRegressionTask(args, logger, accelerator, root=os.path.join(output_dir, 'train_output'),
                                      training=True, enable_log_eval=False)
    eval_task = VideoRegressionTask(args, logger, accelerator, root=os.path.join(output_dir, 'eval_output'),
                                     training=False, enable_log_eval=args.log_eval)

    # Datasets & Loaders
    frame_range = tuple(args.frame_range) if args.frame_range is not None else None
    logger.info(f'Create training dataset & loader: {os.path.join(args.dataset, args.dataset_name)}')
    train_dataset = create_dataset(logger=logger, args=args, training=True, frame_range=frame_range, with_flow=args.use_flow)
    train_loader = create_loader(logger=logger, args=args, training=True, dataset=train_dataset)

    logger.info(f'Create evaluation dataset & loader: {os.path.join(args.dataset, args.dataset_name)}')
    eval_dataset = create_dataset(logger=logger, args=args, training=False, frame_range=frame_range)
    eval_loader = create_loader(logger=logger, args=args, training=False, dataset=eval_dataset)

    # Model
    logger.info(f'Create model: {args.model}')
    model = model_cls[args.model].build_model(args, logger, train_task.parse_input(train_loader, next(iter(train_loader)))[0])
    logger.info(f'Model info:')
    logger.info(model)

    logger.info(f'Number of parameters:')
    num_params = model.get_num_parameters()
    for k, v in num_params.items():
        logger.info(f'    {k}: {v / 10**6:.2f}M')

    # CRSPE 内容初始化（GT 下采样特征）
    if getattr(model, 'crspe', None) is not None and getattr(args, 'crspe_content_init', True):
        feats = compute_content_features(eval_dataset, size=getattr(args, 'crspe_content_size', 32))
        model.init_from_content(feats, seed=args.seed)
        logger.info(f'CRSPE initialized from downsampled GT content: {tuple(feats.shape)}')

    # Warm start
    if args.init_from:
        state = torch.load(args.init_from, map_location='cpu')
        load_compatible_state(model, state, logger)

    # Reference model for residual (P-model) coding
    reference_state_dict = None
    if args.ref_model:
        reference_state_dict = torch.load(args.ref_model, map_location='cpu')
        args.quant_residual = True
        logger.info(f'Loaded reference model for residual coding: {args.ref_model}')

    # Pruning & quantization parametrizations
    initial_parametrizations(args, logger, model)
    attach_hyperprior(args, logger, model)

    # Place model & loaders
    model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)

    # Training dynamics tracker
    tracker = PatchDynamicsTracker(train_dataset.num_patches, window=args.dyn_window, var_weight=args.dyn_var_weight)

    # 中途断点续训：检查已有 checkpoint（崩溃重启时从最近 checkpoint 恢复，不从头开始）
    ckpt_root = os.path.join(output_dir, 'checkpoints')
    resume_ckpt = None
    if os.path.isdir(ckpt_root):
        cands = []
        for d in os.listdir(ckpt_root):
            if d.startswith('checkpoint_') and d.split('_')[-1].isdigit():
                cands.append((int(d.split('_')[-1]), os.path.join(ckpt_root, d)))
        if cands:
            cands.sort()
            resume_ckpt = cands[-1][1]
            logger.info(f'Resume from mid-training checkpoint: {resume_ckpt} (epoch {cands[-1][0]})')
        # 清理旧 checkpoint（只保留最新的，避免磁盘膨胀）
        if len(cands) > 3:
            for ep, path in cands[:-3]:
                import shutil; shutil.rmtree(path, ignore_errors=True)

    # r 系列接力：主训练完成的浮点快照在（state_after_main.pt）→ 载入并跳过 main（prune+QAT 照跑）。
    # 与 myhinerbv_main 的 base 阶梯同款逻辑，用于「已完成实验 + 加强压缩预算重跑」的变体。
    init_state, skip_main = None, False
    snap_main = os.path.join(output_dir, 'state_after_main.pt')
    if resume_ckpt is None and os.path.isfile(snap_main):
        init_state = torch.load(snap_main, map_location='cpu')
        skip_main = True
        logger.info('Resume from state_after_main.pt (skip main, prune + QAT).')

    # Train（main → prune → quant → compress）
    results = train_model(args, logger, accelerator, output_dir, model, train_dataset, train_loader, eval_loader,
                          train_task, eval_task, dynamics_tracker=tracker,
                          reference_state_dict=reference_state_dict, model_name='single',
                          resume_checkpoint=resume_ckpt,
                          skip_main=skip_main, init_state_dict=init_state)

    # Save the final canonical state for downstream use
    if args.save_state:
        final_state = get_bitstream_state_dict(model)
        torch.save(final_state, args.save_state)
        logger.info(f'Saved final canonical state to {args.save_state}')

    # Save summary
    if accelerator.is_main_process:
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        with open(os.path.join(output_dir, 'results', 'single_summary.txt'), 'w') as f:
            for k, v in results.items():
                if k == 'decoded_state':
                    continue
                if k == 'quant':
                    for q, r in v.items():
                        f.write(f'Q{q}: bpp={r["bpp"]:.4f} bytes={r["num_bytes"]} metrics={r["metrics"]}\n')
                elif v is not None:
                    f.write(f'{k}: {v}\n')

    logger.info(f'Training completed in: {datetime.datetime.now() - start_time}')
    logger.info(f'Output are located in: {output_dir}')


if __name__ == '__main__':
    main()
