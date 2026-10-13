"""
MyHiNeRV 顶层管线（入口）：两级（低动态 base + 高动态 GOP 小模型）视频压缩。

流程：
  Phase A（低动态 base）：在全视频上训练一个"宽而深"的 MyHiNeRV base 模型，
        学习整段视频的通用信息，剪枝 + QAT + intra 压缩编码一次、全帧共享；
        同时用 PatchDynamicsTracker 记录逐 patch 训练动态（patch 丢弃采样 +
        场景切分的依据）。
  Phase B（自适应切分）：按"相邻帧内容余弦散度（TeNeRV）+ 训练动态得分（dynamicAL）"
        融合分数自适应切分 GOP（Content-Aware GoP Partitioning 贪心算法）。
        不采用 UAR-NVC 的固定帧数切分。
  Phase C（高动态 GOP 小模型）：每个 GOP 一个"小而轻"的 MyHiNeRV：
        - 参数由 base 宽模型确定性推导（derive.py：通道/深度切片 + 时间维插值），
          所有 GOP 只依赖 base、彼此独立 —— 避免 UAR-NVC 最差变体 D-init 的
          顺序依赖与误差累积；
        - 单独完整训练（补足 epochs）+ 残差 QAT：权重相对 derive(base 解码权重)
          的残差量化（noise 模式 dead-zone 跳过 / qec 模式可学习 q_step 高斯熵编码），
          只传残差，base 的通用信息被全部 GOP 共享。
  Phase D（汇总）：各量化级聚合 base + Σ GOP 字节 → bpp；GOP 逐帧指标 → 全局 PSNR/MS-SSIM。

用法示例（UVG 1080p）：
  dataset_dir=~/Datasets/UVG/1920x1080; dataset_name=ReadySetGo
  python scripts/precompute_flow.py --dataset ${dataset_dir} --dataset-name ${dataset_name} --input-size 1080 1920
  train_cfg=$(cat cfgs/train/myhinerbv_1920x1080.txt)
  model_cfg=$(cat cfgs/models/uvg-myhinerbv-s_1920x1080.txt)
  python myhinerbv_main.py --dataset ${dataset_dir} --dataset-name ${dataset_name} \
      --output output ${train_cfg} ${model_cfg}
"""
import time
from utils import *
from datasets import create_dataset, create_loader, set_dataset_args, compute_content_features
from hinerv_tasks import VideoRegressionTask, set_task_args
from hinerv_compress import (set_compression_args, initial_parametrizations, get_bitstream_state_dict)
from compression.hyperprior import attach_hyperprior
from dynamics import (set_dynamics_args, PatchDynamicsTracker, compute_content_divergence,
                      segment_clips)
from derive import derive_state_dict
from train_lib import train_model

import argparse

parser = argparse.ArgumentParser()

# Script
group = parser.add_argument_group('Script parameters')
group.add_argument('--model', default='MyHiNeRV', type=str, help='Model to build, e.g. MyHiNeRV.')
group.add_argument('--epochs', type=int, default=300, help='Number of main-training epochs.')
group.add_argument('--eval-epochs', type=int, default=30, help='Number of epochs for every evaluation.')
group.add_argument('--log-epochs', type=int, default=-1, help='Number of epochs for every output logging.')
group.add_argument('--prune-epochs', default=60, type=int, help='Number of pruning fine-tuning epochs (base model).')
group.add_argument('--prune-warmup-epochs', default=0, type=int, help='Warmup epochs with pruning.')
group.add_argument('--prune-lr-scale', default=1., type=float, help='Learning rate scale with pruning.')
group.add_argument('--quant-epochs', default=30, type=int, help='Number of QAT epochs.')
group.add_argument('--quant-warmup-epochs', default=0, type=int, help='Warmup epochs with QAT.')
group.add_argument('--quant-lr-scale', default=0.1, type=float, help='Learning rate scale with QAT.')
group.add_argument('--output', default='output', type=str, help='Path to output folder.')
group.add_argument('--exp-name', default=None, type=str, help='Suffix of the output folder.')
group.add_argument('--debug', action='store_true', default=False, help='Set debug mode.')
group.add_argument('--grad-accum', type=int, default=1, help='Number of gradient accumulation steps.')
group.add_argument('--seed', type=int, default=0, help='Random seed.')
group.add_argument('--use-flow', type=str_to_bool, default=True,
                   help='attach RAFT pseudo-label flow to training sets if the file exists.')

# Pipeline (two-level MyHiNeRV)
group = parser.add_argument_group('Pipeline parameters')
group.add_argument('--base-channels', type=int, default=-1,
                   help='channels of the wide base model (-1: same as --channels).')
group.add_argument('--main-extra-epochs', type=int, default=0,
                   help='extra main-training epochs for the base model from the state_after_main snapshot (fresh schedule), e.g. 100 for 120+100. 0: disabled.')
group.add_argument('--base-w-rate', type=float, default=-1,
                   help='rate-distortion weight (lambda) of the base model (-1: inherit --w-rate). Typically smaller than the GOP lambda.')
group.add_argument('--seg-mode', type=str, default='adaptive', choices=['adaptive', 'fixed'],
                   help='GOP segmentation mode: adaptive (content+dynamics) or fixed (equal length, ablation control).')
group.add_argument('--base-depths', type=int, nargs='+', default=[],
                   help='depths of the wide base model (empty: same as --depths).')
group.add_argument('--base-exps', type=float, nargs='+', default=[],
                   help='exps of the wide base model (empty: same as --exps).')
group.add_argument('--p-epochs', type=int, default=-1,
                   help='main-training epochs of each GOP small model (-1: same as --epochs, i.e. fully compensated).')
group.add_argument('--p-quant-epochs', type=int, default=-1,
                   help='QAT epochs of each GOP small model (-1: same as --quant-epochs).')
group.add_argument('--num-clips', type=int, default=-1,
                   help='number of GOPs. -1: auto-estimated from --clip-len (adaptive boundaries).')
group.add_argument('--clip-len', type=int, default=64, help='target GOP length (only used to estimate K).')
group.add_argument('--min-clip-len', type=int, default=8, help='minimum GOP length constraint.')
group.add_argument('--max-clip-len', type=int, default=-1, help='maximum GOP length constraint; overlong GOPs are split at interior divergence peaks (-1: unlimited).')
group.add_argument('--clips-file', default='', type=str, help='load GOP boundaries from this yaml (skip segmentation).')
group.add_argument('--gop-init-mode', default='interp', type=str, choices=['base', 'interp'],
                   help='GOP P-model init: interp = UAR-NVC II-init fused with base derive '
                        '(eps*derive(base) + (1-eps)*prev_gop); base = pure base truncation (A1 legacy, local-optimum trap).')
group.add_argument('--gop-init-a', default=0.6, type=float, help='II-init epsilon: eps = clamp(1 - a*exp(-b*gap + c), 0, 1).')
group.add_argument('--gop-init-b', default=2.2e-5, type=float, help='II-init epsilon decay rate (UAR-NVC fitted).')
group.add_argument('--gop-init-c', default=0.5, type=float, help='II-init epsilon offset (UAR-NVC fitted).')
group.add_argument('--gop-channels', default=-1, type=int,
                   help='GOP 分支模型通道数覆盖（-1 = 跟随主模型配置）。需 ≤ base 通道数以保证 derive 可映射。')

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


def apply_base_overrides(args, logger):
    """把 base-* 结构覆盖应用到 args 副本（构建宽 base 模型用）。"""
    base_args = copy.deepcopy(args)
    if args.base_channels > 0:
        base_args.channels = args.base_channels
    if len(args.base_depths) > 0:
        assert len(args.base_depths) == len(args.depths)
        base_args.depths = list(args.base_depths)
    if len(args.base_exps) > 0:
        assert len(args.base_exps) == len(args.exps)
        base_args.exps = list(args.base_exps)
    base_args.quant_residual = False  # base 永远 intra 编码
    base_args.gop_mode = 'full'      # base 自身永远完整重建模式
    # 时间网格：base 用固定 150 级（覆盖全视频），与命令行参数顺序无关
    base_args.base_grid_size = [150] + list(base_args.base_grid_size[1:])
    # λ 分级：base 宽模型的率失真权重可小于 GOP 小模型（--base-w-rate -1 = 继承 --w-rate）
    if args.base_w_rate >= 0:
        base_args.w_rate = args.base_w_rate
    return base_args


def latest_checkpoint(output_dir, logger=None):
    """返回 CheckpointManager 保存的最新 checkpoint_{N} 目录（无则 None）。"""
    ckpt_root = os.path.join(output_dir, 'checkpoints')
    if not os.path.isdir(ckpt_root):
        return None
    cands = []
    for d in os.listdir(ckpt_root):
        if d.startswith('checkpoint_') and d.split('_')[-1].isdigit():
            cands.append((int(d.split('_')[-1]), os.path.join(ckpt_root, d)))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


def state_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def resume_base(args, logger):
    """
    断点续跑：base 已完成（decoded_state + 码流存在）时重建 results 与帧动态得分，
    返回 (results, base_decoded, frame_scores)；不可续跑返回 None。
    """
    base_dir = os.path.join(args.output_dir_current, 'base')
    dec_path = os.path.join(base_dir, 'decoded_state.pt')
    dyn_path = os.path.join(base_dir, 'patch_dynamics.npy')
    if not (os.path.isfile(dec_path) and os.path.isfile(dyn_path)):
        return None
    logger.info('[base] Resume: found finished base — skip base training.')
    base_decoded = state_to_cpu(torch.load(dec_path, map_location='cpu'))
    patch_scores = np.load(dyn_path)
    n_h = max(1, args.input_size[0] // args.patch_size[1])
    n_w = max(1, args.input_size[1] // args.patch_size[2])
    frame_scores = patch_scores.reshape(-1, n_h * n_w).mean(axis=1)
    results = {'quant': {}}
    for q in args.quant_level:
        z = os.path.join(base_dir, 'bitstreams', f'Q{q}.zip')
        if os.path.isfile(z):
            results['quant'][q] = {'num_bytes': os.path.getsize(z)}
    return results, base_decoded, frame_scores


def train_base(args, logger, accelerator):
    """
    Phase A：低动态 base 宽模型（全视频）。
    完整训练 + 剪枝 + QAT + intra 压缩（编码一次，全帧共享）；
    同时记录逐 patch 训练动态。已完成时自动断点续跑（见 resume_base）。
    返回 (base_results, base_decoded_state, patch_dynamics_scores)。
    """
    resumed = resume_base(args, logger)
    if resumed is not None:
        return resumed

    base_args = apply_base_overrides(args, logger)

    base_dir = os.path.join(args.output_dir_current, 'base')
    # 分级续跑：QAT/压缩阶段失败过 → 从最近的阶段快照继续（主训练/剪枝不重跑）
    init_state, skip_main, skip_prune = None, False, False
    main_epochs = args.epochs
    snap_prune = os.path.join(base_dir, 'state_after_prune.pt')
    snap_main = os.path.join(base_dir, 'state_after_main.pt')
    resume_ckpt = None
    extra = int(getattr(args, 'main_extra_epochs', 0))
    extra_done = os.path.join(base_dir, 'state_after_extra.done')
    if extra > 0 and os.path.isfile(snap_main) and not os.path.isfile(extra_done):
        # EXTRA main training with mid-training checkpoint resume:
        # 先查最新 checkpoint（可能来自 EXTRA 的部分进度），有就直接续训。
        # 不移动旧 checkpoint——即使来自原始 120ep 运行（权重仍比裸快照好），
        # 调度器会按 main_epochs=extra 重建，LR 失配可接受。
        resume_ckpt = latest_checkpoint(base_dir, logger)
        if resume_ckpt is not None:
            logger.info(f'[base] EXTRA resume from mid-training checkpoint: {resume_ckpt}')
        else:
            logger.info(f'[base] EXTRA main training: +{extra} epochs from state_after_main.pt (fresh schedule).')

        init_state = torch.load(snap_main, map_location='cpu')
        main_epochs = extra
    elif os.path.isfile(snap_prune):
        init_state, skip_main, skip_prune = torch.load(snap_prune, map_location='cpu'), True, True
        logger.info('[base] Resume from state_after_prune.pt (skip main + prune, QAT only).')
    elif os.path.isfile(snap_main):
        init_state, skip_main = torch.load(snap_main, map_location='cpu'), True
        logger.info('[base] Resume from state_after_main.pt (skip main, prune + QAT).')
    else:
        resume_ckpt = latest_checkpoint(base_dir, logger)
        if resume_ckpt is not None:
            logger.info(f'[base] Resume from mid-training checkpoint {resume_ckpt}.')
    if accelerator.is_main_process:
        os.makedirs(base_dir, exist_ok=True)

    train_task = VideoRegressionTask(base_args, logger, accelerator, root=os.path.join(base_dir, 'train_output'),
                                      training=True, enable_log_eval=False)
    eval_task = VideoRegressionTask(base_args, logger, accelerator, root=os.path.join(base_dir, 'eval_output'),
                                     training=False, enable_log_eval=args.log_eval)

    train_dataset = create_dataset(logger=logger, args=base_args, training=True, with_flow=args.use_flow)
    train_loader = create_loader(logger=logger, args=base_args, training=True, dataset=train_dataset)
    eval_dataset = create_dataset(logger=logger, args=base_args, training=False)
    eval_loader = create_loader(logger=logger, args=base_args, training=False, dataset=eval_dataset)

    model = model_cls[base_args.model].build_model(base_args, logger,
                                                   train_task.parse_input(train_loader, next(iter(train_loader)))[0])
    num_params = model.get_num_parameters()
    logger.info(f'[base] Wide model parameters: ' + '  '.join([f'{k}={v / 10**6:.2f}M' for k, v in num_params.items()]))

    if getattr(model, 'crspe', None) is not None and getattr(base_args, 'crspe_content_init', True):
        feats = compute_content_features(eval_dataset, size=getattr(base_args, 'crspe_content_size', 32))
        model.init_from_content(feats, seed=args.seed)
        logger.info(f'[base] CRSPE content-initialized from {tuple(feats.shape)}')

    initial_parametrizations(base_args, logger, model)
    attach_hyperprior(base_args, logger, model)
    model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)

    tracker = PatchDynamicsTracker(train_dataset.num_patches, window=args.dyn_window, var_weight=args.dyn_var_weight)

    results = train_model(base_args, logger, accelerator, base_dir, model, train_dataset, train_loader,
                          eval_loader, train_task, eval_task, dynamics_tracker=tracker,
                          main_epochs=main_epochs,
                          reference_state_dict=None, model_name='base',
                          skip_main=skip_main, skip_prune=skip_prune, init_state_dict=init_state,
                          resume_checkpoint=resume_ckpt)
    if extra > 0 and not os.path.isfile(extra_done) and 'decoded_state' in results:
        open(extra_done, 'w').write(f'{extra} epochs done at {time.strftime("%F %T")}')

    base_decoded = state_to_cpu(results.get('decoded_state', get_bitstream_state_dict(model)))
    if accelerator.is_main_process:
        torch.save(base_decoded, os.path.join(base_dir, 'decoded_state.pt'))
        np.save(os.path.join(base_dir, 'patch_dynamics.npy'), tracker.get_scores())

    del model, train_loader, eval_loader, train_dataset, eval_dataset
    torch.cuda.empty_cache()

    return results, base_decoded, tracker.get_frame_scores()


def segment_video(args, logger, accelerator, dynamics_scores):
    """
    Phase B：自适应场景切分（内容余弦散度 + 训练动态；不使用 UAR-NVC 的固定 gop）。
    返回 clips [(t0, t1), ...]。
    """
    eval_dataset = create_dataset(logger=logger, args=args, training=False)
    divergence = compute_content_divergence(eval_dataset, size=args.seg_feature_size)
    del eval_dataset

    if args.seg_mode == 'fixed':
        # 固定等长切分（消融对照：UAR-NVC 式固定 gop，不含边界自适应）
        T = len(divergence)
        n = args.num_clips if args.num_clips > 0 else max(1, int(round(T / args.clip_len)))
        n = max(1, min(n, T // args.min_clip_len))
        bounds = [int(round(i * T / n)) for i in range(n + 1)]
        clips = list(zip(bounds[:-1], bounds[1:]))
        logger.info(f'Fixed-length GOPs (n={n}): {clips}')
    else:
        clips = segment_clips(divergence, dynamics=dynamics_scores,
                              num_clips=args.num_clips if args.num_clips > 0 else None,
                              clip_len=args.clip_len, min_len=args.min_clip_len,
                              max_len=args.max_clip_len if args.max_clip_len > 0 else None,
                              dyn_weight=args.seg_dyn_weight)
        logger.info(f'Segmented into {len(clips)} adaptive GOPs: {clips}')
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir_current, 'clips.yaml'), 'w') as f:
            yaml.dump({'clips': [list(c) for c in clips],
                       'divergence': divergence.tolist(),
                       'dynamics': np.asarray(dynamics_scores).tolist()}, f)
    return clips


def derive_reference(small_model, base_decoded, logger):
    """
    从 base 解码权重推导小模型权重（初始化与残差参考的基础项）。
    返回可直接 load 的推导权重表（canonical 键名）。
    """
    derived = derive_state_dict(small_model, base_decoded, logger=logger)
    small_model.load_state_dict(derived, strict=False)
    return derived


def interp_gop_init(small_model, derived_state, prev_gop_decoded, eps, logger=None):
    """
    UAR-NVC II-init 融合版：θ' = ε·derive(base) + (1−ε)·θ*_{k−1}。
    前一 GOP 的解码权重先经 derive_state_dict 的形状映射（同为小模型，仅时间维
    长度可能不同，线性插值对齐），再与 base 推导项逐张量插值（仅 ndim>1 浮点张量，
    1D 偏置等保持 base 推导侧）。返回插值结果与参与插值的张量数。
    """
    prev_mapped = derive_state_dict(small_model, prev_gop_decoded, logger=logger)
    out = dict(derived_state)
    n = 0
    for k, dv in derived_state.items():
        pv = prev_mapped.get(k)
        if pv is not None and dv.ndim > 1 and torch.is_floating_point(dv):
            out[k] = (eps * dv.float() + (1.0 - eps) * pv.float()).to(dv.dtype)
            n += 1
    return out, n


def gop_gap_mse(args, t_prev_last, t_cur_first):
    """相邻 GOP 边界帧 MSE×256²（UAR-NVC computer_GOP_gap 的标度量级）。"""
    import numpy as np
    from PIL import Image
    d = os.path.join(args.dataset, args.dataset_name)
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg')))
    a = np.asarray(Image.open(os.path.join(d, files[t_prev_last])).convert('RGB'), np.float32) / 255.
    b = np.asarray(Image.open(os.path.join(d, files[t_cur_first])).convert('RGB'), np.float32) / 255.
    return float(((a - b) ** 2).mean() * 256 ** 2)


def ii_init_epsilon(gap, a=0.6, b=2.2e-5, c=0.5):
    """UAR-NVC 公式(9)：ε = clamp(1 − a·e^{−b·gap + c}, 0, 1)。
    gap 大（相邻 GOP 内容差异大）→ ε→1 信 base 推导；gap 小 → ε→0 信前一 GOP。"""
    eps = 1.0 - a * math.exp(-b * gap + c)
    return float(min(max(eps, 0.0), 1.0))


def train_gop(args, logger, accelerator, gop_idx, t0, t1, base_decoded, prev_gop_decoded=None):
    """
    Phase C 的单 GOP 训练（UAR-NVC II-init 融合版）：
    - 初始化 θ' = ε·derive(base) + (1−ε)·前一 GOP 解码权重（ε 由相邻 GOP 边界帧
      MSE gap 自适应；gop00 无前一 GOP → 纯 base 推导）。纯 base 截断易陷 base
      局部最优，纯复制前一 GOP（D-init）是 UAR-NVC 消融的最差变体，插值取两家之长；
    - 输出层零初始化（恒等于 base 起步），分支只学 base 没学到的内容；
    - 完整 epochs 训练 + 残差 QAT/压缩（参考 = θ'，解码端可重建）。
    返回该 GOP 的 results（含 decoded_state）。
    """
    gop_args = copy.deepcopy(args)
    gop_args.quant_residual = True  # GOP 模型残差编码
    quant_epochs = args.p_quant_epochs if args.p_quant_epochs >= 0 else args.quant_epochs
    p_epochs = args.p_epochs if args.p_epochs > 0 else args.epochs

    gop_dir = os.path.join(args.output_dir_current, f'gop{gop_idx:02d}_{t0:04d}-{t1:04d}')
    if accelerator.is_main_process:
        os.makedirs(gop_dir, exist_ok=True)

    # 分级续跑：该 GOP 的 QAT 失败过 → 从主训练快照继续
    gop_init_state, gop_skip_main, gop_resume_ckpt = None, False, None
    gop_snap_main = os.path.join(gop_dir, 'state_after_main.pt')
    if os.path.isfile(gop_snap_main) and not os.path.isfile(summary_path := os.path.join(gop_dir, 'gop_summary.yaml')):
        gop_init_state, gop_skip_main = torch.load(gop_snap_main, map_location='cpu'), True
        logger.info(f'[gop{gop_idx}] Resume from state_after_main.pt (skip main training).')
    else:
        gop_resume_ckpt = latest_checkpoint(gop_dir)
        if gop_resume_ckpt is not None:
            logger.info(f'[gop{gop_idx}] Resume from mid-training checkpoint {gop_resume_ckpt}.')
    summary_path = os.path.join(gop_dir, 'gop_summary.yaml')
    if accelerator.is_main_process and os.path.isfile(summary_path):
        with open(summary_path) as f:
            # unsafe_load：summary 含 numpy 标量标签（bpp 等），safe_load 会拒绝
            saved = yaml.unsafe_load(f)
        results = {'decoded_state': torch.load(os.path.join(gop_dir, 'decoded_state.pt'), map_location='cpu'),
                   'quant': {int(q): {'num_bytes': r['bytes'], 'bpp': r['bpp_gop'],
                                      'metrics': r['metrics'], 'frame_metrics': r['frame_metrics']}
                             for q, r in saved['quant'].items()}}
        logger.info(f'[gop{gop_idx}] Resume: gop [{t0},{t1}) already done — skip.')
        return results

    train_task = VideoRegressionTask(gop_args, logger, accelerator, root=os.path.join(gop_dir, 'train_output'),
                                      training=True, enable_log_eval=False)
    eval_task = VideoRegressionTask(gop_args, logger, accelerator, root=os.path.join(gop_dir, 'eval_output'),
                                     training=False, enable_log_eval=args.log_eval)

    train_dataset = create_dataset(logger=logger, args=gop_args, training=True, frame_range=(t0, t1), with_flow=args.use_flow)
    train_loader = create_loader(logger=logger, args=gop_args, training=True, dataset=train_dataset)
    eval_dataset = create_dataset(logger=logger, args=gop_args, training=False, frame_range=(t0, t1))
    eval_loader = create_loader(logger=logger, args=gop_args, training=False, dataset=eval_dataset)

    gop_args.gop_mode = args.gop_mode if args.gop_mode in ('full', 'residual') else 'full'
    # 时间网格：GOP 跟随 clip 长度（-1），避免 600 帧级网格塞进 16 帧小模型
    gop_args.base_grid_size = [-1] + list(gop_args.base_grid_size[1:])
    # 分支通道数覆盖（A3：加宽分支缓解 0.41M 容量饥饿；须 ≤ base 通道保证 derive 可映射）
    if getattr(args, 'gop_channels', -1) > 0:
        gop_args.channels = args.gop_channels
    model = model_cls[gop_args.model].build_model(gop_args, logger,
                                                  train_task.parse_input(train_loader, next(iter(train_loader)))[0])
    num_params = model.get_num_parameters()
    logger.info(f'[gop{gop_idx}] small model ({gop_args.gop_mode}) frames [{t0},{t1})    '
                + '  '.join([f'{k}={v / 10**6:.2f}M' for k, v in num_params.items()]))

    # 残差模式：构建冻结的 base 参考模型（加载解码权重；不训练、不进码流）。
    # 注意 attach 推迟到 accelerator.prepare 之后（base_ref 不注册进模块树，
    # 设备转移需显式做）。
    base_ref = None
    if gop_args.gop_mode == 'residual':
        # base 参考必须按完整视频构建（其时间网格覆盖全视频），GOP 前向时局部坐标平移到全局
        sample_in = train_task.parse_input(train_loader, next(iter(train_loader)))[0]
        T_full = args.full_video_frames  # 由 main() 从全视频数据集获得
        full_in = {'idx': torch.zeros(1, 3, dtype=int),
                   'idx_max': (T_full, sample_in['idx_max'][1], sample_in['idx_max'][2]),
                   'video_size': (T_full, sample_in['video_size'][1], sample_in['video_size'][2]),
                   'patch_size': sample_in['patch_size']}
        base_args_local = apply_base_overrides(args, logger)
        base_ref = model_cls[base_args_local.model].build_model(base_args_local, logger, full_in)
        base_ref.load_state_dict({k: v for k, v in base_decoded.items()}, strict=False)
        base_ref.eval()
        for pp in base_ref.parameters():
            pp.requires_grad_(False)

    # ---- II-init 融合版初始化：θ' = ε·derive(base) + (1−ε)·θ*_{k−1} ----
    derived_state = derive_state_dict(model, base_decoded, logger=logger)
    init_state, eps_used = dict(derived_state), 1.0
    if prev_gop_decoded is not None and getattr(args, 'gop_init_mode', 'interp') == 'interp':
        gap = gop_gap_mse(args, t0 - 1, t0)  # 前一 GOP 尾帧 vs 当前 GOP 首帧
        eps_used = ii_init_epsilon(gap, a=getattr(args, 'gop_init_a', 0.6),
                                   b=getattr(args, 'gop_init_b', 2.2e-5), c=getattr(args, 'gop_init_c', 0.5))
        init_state, n_interp = interp_gop_init(model, derived_state, prev_gop_decoded, eps_used, logger=logger)
        logger.info(f'[gop{gop_idx}] II-init: gap={gap:.1f}  eps={eps_used:.3f}  '
                    f'interp tensors={n_interp}/{len(derived_state)}')
    model.load_state_dict(init_state, strict=False)
    derived_ref = init_state  # 残差编码参考 = 插值后的 θ'（解码端可重建）

    # 残差模式：derive/插值载入后把输出头重新清零，保证"恒等于 base"起步
    if gop_args.gop_mode == 'residual':
        with torch.no_grad():
            for m in model.decoder.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    initial_parametrizations(gop_args, logger, model)
    attach_hyperprior(gop_args, logger, model)
    model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)

    # attach 推迟到 prepare 之后：base_ref 显式移到模型所在设备
    if gop_args.gop_mode == 'residual' and base_ref is not None:
        base_ref.to(next(model.parameters()).device)
        model.attach_base_ref(base_ref, frame_offset=t0, full_video_size=full_in['video_size'])
        logger.info(f'[gop{gop_idx}] residual mode: frozen base reference attached '
                    f'(offset={t0}, full={full_in["video_size"]}, zero-init head, II-init eps={eps_used:.3f}).')

    tracker = PatchDynamicsTracker(train_dataset.num_patches, window=args.dyn_window, var_weight=args.dyn_var_weight)

    results = train_model(gop_args, logger, accelerator, gop_dir, model, train_dataset, train_loader, eval_loader,
                          train_task, eval_task, dynamics_tracker=tracker,
                          main_epochs=p_epochs,            # 补足训练 epochs
                          enable_pruning=False,            # 小模型不剪枝（残差编码足够稀疏）
                          quant_epochs=quant_epochs,
                          reference_state_dict=derived_ref,  # 残差参考（base 推导）
                          model_name=f'gop{gop_idx}[{t0}:{t1}]',
                          skip_main=gop_skip_main, init_state_dict=gop_init_state,
                          resume_checkpoint=gop_resume_ckpt)

    # GOP 的解码权重与逐项结果（断点续跑依据：gop_summary.yaml 存在即视为已完成）
    if 'decoded_state' not in results:
        results['decoded_state'] = get_bitstream_state_dict(model)
    results['decoded_state'] = state_to_cpu(results['decoded_state'])
    if accelerator.is_main_process:
        torch.save(results['decoded_state'], os.path.join(gop_dir, 'decoded_state.pt'))
        with open(os.path.join(gop_dir, 'gop_summary.yaml'), 'w') as f:
            yaml.dump({'gop': [t0, t1],
                       'ii_init_eps': float(eps_used),
                       'quant': {str(q): {'bytes': r['num_bytes'], 'bpp_gop': r['bpp'],
                                          'metrics': r['metrics'], 'frame_metrics': r['frame_metrics']}
                                 for q, r in results['quant'].items()}}, f)

    del model, train_loader, eval_loader, train_dataset, eval_dataset
    torch.cuda.empty_cache()

    return results


def aggregate_results(args, logger, accelerator, clips, base_results, gop_results):
    """
    Phase D：聚合 base + 各 GOP 的码流与逐帧指标，输出全局 RD 点。
    """
    T_global = clips[-1][1]
    sample = create_dataset(logger=logger, args=args, training=False)
    H, W = sample.video_size[1], sample.video_size[2]
    total_pixels = T_global * H * W
    del sample

    summary = {'clips': [list(c) for c in clips], 'total_pixels': int(total_pixels), 'quant': {}}
    for q in args.quant_level:
        # base 字节（编码一次，全帧共享）
        base_entry = base_results['quant'].get(q) if base_results is not None else None
        total_bytes = base_entry['num_bytes'] if base_entry is not None else 0

        frame_metric_vals = None
        per_gop = []
        for (t0, t1), res in zip(clips, gop_results):
            if q not in res['quant']:
                continue
            total_bytes += res['quant'][q]['num_bytes']
            fm = res['quant'][q]['frame_metrics']
            if frame_metric_vals is None:
                frame_metric_vals = {name: [] for name in fm.keys()}
            for name, vals in fm.items():
                frame_metric_vals[name].extend(vals)
            per_gop.append({'frames': [t0, t1], 'bytes': res['quant'][q]['num_bytes'],
                            'metrics': res['quant'][q]['metrics']})
        if frame_metric_vals is None:
            continue
        bpp = total_bytes * 8 / total_pixels
        metrics = {name: float(np.mean(vals)) for name, vals in frame_metric_vals.items()}
        summary['quant'][str(q)] = {'bytes': int(total_bytes), 'bpp': float(bpp), 'metrics': metrics,
                                    'base_bytes': int(base_entry['num_bytes']) if base_entry is not None else 0,
                                    'per_gop': per_gop}
        logger.info(f'*** Q{q}: total {total_bytes / 10**6:.2f}MB (base '
                    f'{(base_entry["num_bytes"] if base_entry is not None else 0) / 10**6:.2f}MB)    {bpp:.4f}bpp    '
                    + '    '.join([f'{k}: {v:.4f}' for k, v in metrics.items()]))

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir_current, 'summary.yaml'), 'w') as f:
            yaml.dump(summary, f)
        with open(os.path.join(args.output_dir_current, 'summary.txt'), 'w') as f:
            f.write(f'clips: {clips}\n')
            for q, r in summary['quant'].items():
                f.write(f'Q{q}: bpp={r["bpp"]:.4f} bytes={r["bytes"]} (base={r["base_bytes"]}) ' +
                        ' '.join([f'{k}={v:.4f}' for k, v in r['metrics'].items()]) + '\n')
    return summary


def main():
    """顶层管线主函数（详见模块 docstring）。"""
    start_time = datetime.datetime.now()

    args = parse_args(parser)

    accelerator, logger, output_dir = get_accelerator_logger(args)
    args.output_dir_current = output_dir
    logger.info(f'Output dir: {output_dir}')

    logger.info(f'Set seed: {args.seed}')
    accelerate.utils.set_seed(args.seed, device_specific=True)

    args.full_video_frames = len([f for f in os.listdir(os.path.join(args.dataset, args.dataset_name)) if not f.startswith('.')])

    # Phase A: 低动态 base 宽模型（全视频，编码一次全帧共享）+ 训练动态记录
    base_results, base_decoded, frame_dynamics = train_base(args, logger, accelerator)

    # Phase B: 自适应 GOP 切分（TeNeRV + dynamicAL；不学 UAR-NVC 的固定切分）
    if args.clips_file and os.path.isfile(args.clips_file):
        with open(args.clips_file, 'r') as f:
            data = yaml.safe_load(f)
        clips = [tuple(c) for c in data['clips']]
        logger.info(f'Loaded GOP boundaries from {args.clips_file}: {clips}')
    else:
        clips = segment_video(args, logger, accelerator, frame_dynamics)

    # Phase C: GOP 小模型（II-init：ε·derive(base)+(1−ε)·前一 GOP；串行链 k→k+1）
    gop_results = []
    prev_dec = None
    for k, (t0, t1) in enumerate(clips):
        logger.info(f'==================== GOP {k + 1}/{len(clips)}: frames [{t0}, {t1}) ====================')
        res = train_gop(args, logger, accelerator, k, t0, t1, base_decoded, prev_gop_decoded=prev_dec)
        gop_results.append(res)
        prev_dec = res['decoded_state']  # 下一 GOP 的 II-init 来源 + 残差参考基准

    # Phase D: 汇总
    summary = aggregate_results(args, logger, accelerator, clips, base_results, gop_results)

    logger.info(f'Pipeline completed in: {datetime.datetime.now() - start_time}')
    logger.info(f'Output are located in: {output_dir}')


if __name__ == '__main__':
    main()
