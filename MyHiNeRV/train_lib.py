"""
训练循环库（MyHiNeRV）：被 hinerv_main.py（单模型）与 myhinerbv_main.py（多 clip 管线）共用。

相对 HiNeRV 训练循环的改动：
1) 帧级训练动态记录：train_step 把逐帧 MSE 按（clip 内）帧号聚合，每个 epoch 结束
   喂给 dynamics.FrameDynamicsTracker —— 既是 dynamicAL 帧权重采样的依据，
   也是场景切分（TeNeRV + 训练动态）的输入之一；
2) dynamicAL 帧权重采样：每 dyn_update_epochs 依据跟踪器得分重建
   WeightedRandomSampler 的 DataLoader（损失内帧等权不变）；
3) 动态高频注入：train_task.set_progress 同步训练进度（hf_start 后启用）；
4) P 模型支持：三阶段（main/prune/quant）中 prune 可关；quant 阶段注入参考模型
   权重做残差 QAT（ResidualQuantNoise），压缩/解压走残差码流（DCVC-RT 借鉴）；
5) eval_step 额外返回逐帧指标数组（跨 clip 聚合全局指标用）。
"""
from utils import *
from datasets import create_loader
from dynamics import (FrameDynamicsTracker, PatchDynamicsTracker, compute_frame_weights,
                      frame_patch_weights, compute_patch_drop_mask,
                      compute_patch_train_weights, compute_patch_loss_weights)
from compression.quant_utils import set_soft_round_anneal
from structured_sparsity import ColumnPruner, HierarchicalSampler
from hinerv_compress import (initial_parametrizations, set_pruning, get_sparsity, set_quantization,
                             start_quantization, set_reference, compress_bitstream, decompress,
                             set_zero, get_bitstream_state_dict, snapshot_model_state, load_bitstream_state)


def get_stage_optimizer_scheduler(args, logger, accelerator, model, loader, stage, epochs_override=None,
                                  lr_scale_override=None):
    """
    为训练阶段（main / prune / quant）创建独立的优化器与调度器；
    epochs_override/lr_scale_override 允许管线为 P 模型指定更少的轮数（p_epochs 等）。
    """
    args = copy.deepcopy(args)
    if stage == 'main':
        args.epochs = max(epochs_override if epochs_override is not None else args.epochs, 1)
    elif stage == 'prune':
        args.lr *= args.prune_lr_scale
        args.warmup_lr *= args.prune_lr_scale
        args.min_lr *= args.prune_lr_scale
        args.epochs = max(epochs_override if epochs_override is not None else args.prune_epochs, 1)
        args.warmup_epochs = args.prune_warmup_epochs
    elif stage == 'quant':
        scale = lr_scale_override if lr_scale_override is not None else args.quant_lr_scale
        args.lr *= scale
        args.warmup_lr *= scale
        args.min_lr *= scale
        args.epochs = max(epochs_override if epochs_override is not None else args.quant_epochs, 1)
        args.warmup_epochs = args.quant_warmup_epochs
    else:
        raise ValueError
    logger.info(f'Create optimizer & scheduler for {stage} stage ({args.epochs} epochs)')
    return get_optimizer_scheduler(args, logger, accelerator, model, loader)


def _accumulate_frame_losses(frame_loss_sum, frame_loss_cnt, inputs, frame_mse,
                             patch_loss_sum=None, patch_loss_cnt=None):
    """
    把一个 batch 的逐帧 MSE [N, T] 按（clip 内）全局帧号累加进聚合表；
    同时（可选）按补丁线性索引累加 patch 级损失（patch 动态跟踪用）。
    帧号 = 补丁 t 坐标 × patch_size[0] + 帧内偏移。
    """
    T_patch = inputs['patch_size'][0]
    idx_max = inputs['idx_max']
    n_hw = idx_max[1] * idx_max[2]
    for n in range(frame_mse.shape[0]):
        t0 = int(inputs['idx'][n, 0].item()) * T_patch
        if patch_loss_sum is not None:
            t_idx, h_idx, w_idx = int(inputs['idx'][n, 0].item()), int(inputs['idx'][n, 1].item()), int(inputs['idx'][n, 2].item())
            p = t_idx * n_hw + h_idx * idx_max[2] + w_idx
            patch_loss_sum[p] += float(frame_mse[n].mean().item())
            patch_loss_cnt[p] += 1
        for dt in range(frame_mse.shape[1]):
            t = t0 + dt
            frame_loss_sum[t] += float(frame_mse[n, dt].item())
            frame_loss_cnt[t] += 1
    return frame_loss_sum, frame_loss_cnt, patch_loss_sum, patch_loss_cnt


def train_step(args, logger, suffix, epoch, model, loader, optimizer, scheduler, task, accelerator,
               log_output=False, frame_loss_sum=None, frame_loss_cnt=None,
               patch_loss_sum=None, patch_loss_cnt=None):
    """
    一个 epoch 的训练循环（HiNeRV train_step 的扩展版）：
    - task.step 返回 6 元组（含 loss_dict）；
    - 逐帧 / 逐 patch MSE 累计（供训练动态跟踪与 patch 丢弃）；
    - 日志按 loss_dict 分项打印。
    """
    start_time = time.time()
    model.train()

    accum_loss = None
    accum_metrices = None
    accum_loss_dict = None
    reduced_loss = None
    reduced_metrics = None
    counts = 0
    samples = 0

    num_updates = epoch * len(loader)
    scheduler.step(epoch=epoch)
    optimizer.zero_grad()

    logger.info(f'Train - Epoch {epoch}: fetching first batch (loader={len(loader)} batches) ...')
    # C4 D/R 交替：率项按 batch 序号门控（每 K 步开一次、乘 K——与逐步联合数学等效）
    dr_k = int(getattr(args, 'dr_alternate', 0) or 0)
    for i, batch in enumerate(loader):
        if hasattr(task, 'set_dr_gate'):
            task.set_dr_gate(i, dr_k)
        # Train step
        with accelerator.accumulate(model):
            inputs, _, outputs, loss, metrics, loss_dict = task.step(model, loader, batch)
            mean_loss = loss.mean()

            # 反向传播（支持梯度累积与混合精度）
            accelerator.backward(mean_loss)
            if accelerator.sync_gradients and args.max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm, args.norm_type)

            optimizer.step()
            scheduler.step_update(num_updates=num_updates)
            optimizer.zero_grad()
            num_updates += 1

        # 累计损失/指标/分项损失
        mean_metrics = {k: v.mean() for k, v in metrics.items()}
        if i == 0:
            accum_loss = mean_loss.detach()
            accum_metrices = {k: mean_metrics[k].detach() for k in mean_metrics}
            accum_loss_dict = {k: v.mean().detach() for k, v in loss_dict.items() if v is not None and v.numel() > 0}
        else:
            accum_loss = accum_loss + mean_loss.detach()
            accum_metrices = {k: accum_metrices[k] + mean_metrics[k].detach() for k in mean_metrics}
            if accum_loss_dict is not None:
                for k, v in loss_dict.items():
                    if v is not None and v.numel() > 0:
                        accum_loss_dict[k] = accum_loss_dict[k] + v.mean().detach() if k in accum_loss_dict else v.mean().detach()

        # 逐帧 / 逐 patch 损失累计（训练动态跟踪）
        if frame_loss_sum is not None and 'frame_mse' in metrics:
            frame_loss_sum, frame_loss_cnt, patch_loss_sum, patch_loss_cnt = \
                _accumulate_frame_losses(frame_loss_sum, frame_loss_cnt, inputs, metrics['frame_mse'].detach(),
                                         patch_loss_sum, patch_loss_cnt)

        counts += 1
        samples += inputs['batch_size'] * accelerator.num_processes

        if (i + 1) % max(len(loader) // 4, 1) == 0 or (i + 1) == len(loader):
            # 跨进程归约后打印训练日志
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean')
            reduced_metrics = accelerator.reduce(accum_metrices, reduction='mean')
            log_msg = f'Train' + ('' if not suffix else f' ({suffix})') + f' - Epoch {epoch} [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    lr: {np.mean([group["lr"] for group in optimizer.param_groups]):.2e}'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss.item() / counts:.4f}'
            if accum_loss_dict is not None:
                log_msg = log_msg + '    [' + '  '.join([f'{k}: {v.item() / counts:.4f}' for k, v in accum_loss_dict.items()]) + ']'
            for k, v in reduced_metrics.items():
                if k not in ('frame_mse',):
                    log_msg = log_msg + f'    {k}: {v.item() / counts:.4f}'
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_eval(dir_name=f'{epoch}_{suffix.lower()}', inputs=inputs, outputs=outputs, metrics=metrics)

    frame_losses = None
    if frame_loss_sum is not None:
        frame_losses = {t: frame_loss_sum[t] / max(frame_loss_cnt[t], 1) for t in range(len(frame_loss_sum))}

    return (reduced_loss.item() / counts,
            {k: v.item() / counts for k, v in reduced_metrics.items() if k != 'frame_mse'},
            frame_losses,
            patch_loss_sum,
            patch_loss_cnt)


def eval_step(args, logger, suffix, epoch, model, loader, task, accelerator, log_output=False):
    """
    一个 epoch 的评估循环：返回 (平均损失, 平均指标, 逐帧指标 dict {metric_name: [T] 列表})。
    """
    start_time = time.time()
    model.eval()

    accum_loss = None
    accum_metrices = None
    reduced_loss = None
    reduced_metrics = None
    counts = 0
    samples = 0

    frame_metric_values = None  # {name: [按帧序排列的值]}

    for i, batch in enumerate(loader):
        # Eval step
        with torch.no_grad():
            inputs, _, outputs, loss, metrics, _ = task.step(model, loader, batch)
            mean_loss = loss.mean()
            mean_metrics = {k: v.mean() for k, v in metrics.items()}

        # 逐帧指标收集（按帧号排序；帧模式 loader 每 batch 是若干连续帧）
        if frame_metric_values is None:
            frame_metric_values = {k: [] for k in metrics.keys()}
            frame_ids_all = []
        T_patch = inputs['patch_size'][0]
        for n in range(inputs['idx'].shape[0]):
            t0 = int(inputs['idx'][n, 0].item()) * T_patch
            for dt in range(metrics[list(metrics.keys())[0]].shape[1]):
                frame_ids_all.append(t0 + dt)
                for k in metrics.keys():
                    frame_metric_values[k].append(float(metrics[k][n, dt].item()))

        if i == 0:
            accum_loss = mean_loss
            accum_metrices = {k: mean_metrics[k] for k in mean_metrics}
        else:
            accum_loss = accum_loss + mean_loss
            accum_metrices = {k: accum_metrices[k] + mean_metrics[k] for k in mean_metrics}

        counts += 1
        samples += inputs['batch_size'] * accelerator.num_processes

        if (i + 1) % max(len(loader) // 4, 1) == 0 or (i + 1) == len(loader):
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean')
            reduced_metrics = accelerator.reduce(accum_metrices, reduction='mean')
            log_msg = f'Eval' + ('' if not suffix else f' ({suffix})') + f' - [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss.item() / counts:.4f}'
            for k, v in reduced_metrics.items():
                log_msg = log_msg + f'    {k}: {v.item() / counts:.4f}'
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_eval(dir_name=f'{epoch}' + ('' if not suffix else f'_{suffix.lower()}'), inputs=inputs, outputs=outputs, metrics=metrics)

    # 按帧号排序逐帧指标
    order = np.argsort(frame_ids_all)
    frame_metrics = {k: np.asarray(v)[order].tolist() for k, v in frame_metric_values.items()}

    return reduced_loss.item() / counts, {k: v.item() / counts for k, v in reduced_metrics.items()}, frame_metrics


def do_eval(args, epoch, num_epochs):
    return (epoch + 1) % args.eval_epochs == 0 or (epoch + 1) == num_epochs


def do_log(args, epoch, num_epochs):
    return (args.log_epochs > 0 and (epoch + 1) % args.log_epochs == 0) or \
           (args.log_epochs == -1 and (epoch + 1) == num_epochs)


def _shutdown_loader_workers(loader):
    """
    主动关闭旧 DataLoader 的 persistent workers。
    加速器注册表会持有已 prepare 的 loader 对象导致其无法被 GC——不关闭的话，
    每次动态权重重建（每 5ep）都会泄漏 2 个空闲 worker 进程（内存与 PID 随训练线性增长，
    2026-09-22 夜间 7 任务并发时曾把系统可用内存压到 7GB 触发 OOM）。
    """
    try:
        it = getattr(loader, '_iterator', None)
        if it is not None and hasattr(it, '_shutdown_workers'):
            it._shutdown_workers()
            loader._iterator = None
    except Exception:
        pass


def make_hierarchical_loader(args, logger, accelerator, dataset, patch_weights, frame_scores):
    """
    B5 Tree-NeRV 式层级采样：帧级权重（时间变化驱动）× patch 级权重（μ+λσ）。
    混合比例由 --tree-ratio 控制（0=纯 LossGrad, 1=纯 TreeNeRV）。
    """
    from structured_sparsity import HierarchicalSampler
    idx_max = dataset.num_patches  # (T, H, W)
    hw = idx_max[1] * idx_max[2]
    # 帧级权重：帧内 patch 均值（≈ Tree-NeRV 的时间变化度代理）
    fw = np.asarray(patch_weights).reshape(-1, hw).mean(axis=1)
    fw = fw / (fw.sum() + 1e-8)
    sampler = HierarchicalSampler(fw, patch_weights, idx_max)
    tree_ratio = float(getattr(args, 'tree_ratio', 0.5))

    class _HierDataset(torch.utils.data.Dataset):
        """包装原数据集，用层级采样替代平面采样。"""
        def __init__(self, base, sampler, n, tree_ratio, seed=0):
            self.base = base; self.sampler = sampler; self.n = n
            self.tree_ratio = tree_ratio; self.rng = np.random.RandomState(seed)
        def __len__(self): return self.n
        def __getitem__(self, i):
            coords = self.sampler.mixed_sample(1, tree_ratio=self.tree_ratio, rng=self.rng)
            t, h, w = coords[0]
            flat = t * (self.sampler.H * self.sampler.W) + h * self.sampler.W + w
            return self.base[flat]

    wrapped = _HierDataset(dataset, sampler, len(dataset), tree_ratio)
    loader = torch.utils.data.DataLoader(
        wrapped, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    if args.workers > 0:
        loader = accelerator.prepare(loader)
    logger.info(f'  Hierarchical loader: tree_ratio={tree_ratio}, '
                 f'{len(dataset)} patches, {idx_max}')
    return loader


def make_weighted_loader(args, logger, accelerator, dataset, patch_weights):
    """
    构建 patch 级加权采样的 DataLoader（不丢弃任何 patch）。
    patch_weights: [P] numpy 数组，全部 > 0（w_min 保证）。
    """
    weights = torch.as_tensor(patch_weights, dtype=torch.double)
    loader = create_loader(args, logger, training=True, dataset=dataset,
                           sample_weights=weights, num_samples=len(dataset))
    loader = accelerator.prepare(loader)
    return loader


def compression_probe(args, logger, accelerator, model, train_loader, eval_loader,
                      train_task, eval_task, output_dir, epoch, eval_pixels, probe_batches=100):
    """
    快速压缩探针：prune(即时) → 短QAT(100 batch) → compress(Q8) → eval → 恢复浮点权重。
    用于在主训练过程中每 N epoch 获取一个中间 RD 点，比较不同方法的收敛速率。
    耗时约 5 分钟/次（vs 主训练每 epoch ~5 分钟），几乎不影响总训练时间。

    joint-rate 模式（UAR-NVC 式全程联合训练）下：权重本身已在量化+率约束下训练，
    探针跳过剪枝与短 QAT，直接用当前可学习 q_step 压缩出真实 RD 点。
    """
    joint = bool(getattr(args, 'joint_rate', False)) and args.quant_mode in ('qec', 'hyper')
    saved_state = copy.deepcopy(unwrap_model(model).state_dict())

    if not joint:
        logger.info(f'  [probe@ep{epoch}] saving float weights, applying prune+QAT...')

        # 1) 剪枝（即时，无微调）
        zeros_p, total_p = set_pruning(args, logger, model, args.prune_ratio, args.prune_weight)
        logger.info(f'  [probe@ep{epoch}] pruned {zeros_p}/{total_p} ({zeros_p/total_p:.1%})')

        # 2) 短 QAT（固定 batch 数，独立优化器）
        set_quantization(args, logger, model, min(args.quant_level), args.quant_noise, args.quant_ste)
        model.train()
        probe_opt = torch.optim.Adam([p for p in unwrap_model(model).parameters() if p.requires_grad],
                                     lr=args.lr * 0.01)
        batch_count = 0
        probe_loss = 0.
        for batch in train_loader:
            if batch_count >= probe_batches:
                break
            with accelerator.accumulate(model):
                _, _, _, loss, _, _ = train_task.step(model, train_loader, batch)
                accelerator.backward(loss.mean())
                probe_opt.step()
                probe_opt.zero_grad()
            probe_loss += loss.mean().item()
            batch_count += 1
        logger.info(f'  [probe@ep{epoch}] QAT {batch_count} batches, loss={probe_loss/max(batch_count,1):.4f}')
    else:
        logger.info(f'  [probe@ep{epoch}] joint-rate mode: direct compress with learned q_step (no prune/QAT)')

    # 3) 压缩 + 解码 + 评估（用最高量化级 Q8 做探针点）
    probe_dir = os.path.join(output_dir, f'probe_ep{epoch}')
    quant_level = max(args.quant_level)
    qat_state = copy.deepcopy(unwrap_model(model).state_dict())
    num_bytes = compress_bitstream(args, logger, accelerator, model, probe_dir, quant_level)
    set_zero(model)
    decompress(args, logger, accelerator, model, probe_dir, quant_level)
    model.eval()
    _, metrics, _ = eval_step(args, logger, f'Probe-Q{quant_level}', 0, model,
                              eval_loader, eval_task, accelerator)
    bpp = num_bytes * 8 / eval_pixels
    result = {'epoch': epoch, 'quant_level': quant_level,
              'bytes': num_bytes, 'bpp': bpp,
              'psnr': metrics.get('psnr'), 'ms_ssim': metrics.get('ms_ssim')}
    logger.info(f'  [probe@ep{epoch}] Q{quant_level}: {bpp:.4f}bpp  PSNR={result["psnr"]:.3f}dB')

    # 4) 恢复浮点权重，继续主训练
    unwrap_model(model).load_state_dict(saved_state)
    model.train()
    logger.info(f'  [probe@ep{epoch}] float weights restored, continuing main training')
    return result


def train_model(args, logger, accelerator, output_dir, model, train_dataset, train_loader, eval_loader,
                train_task, eval_task, dynamics_tracker=None,
                main_epochs=None, prune_epochs=None, quant_epochs=None, enable_pruning=None,
                reference_state_dict=None, model_name='model', eval_dataset_size=None,
                skip_main=False, skip_prune=False, init_state_dict=None,
                resume_checkpoint=None):
    """
    完整的单模型三阶段训练（main → prune → quant）+ 压缩评估。
    返回 results: {
        'full'/'pruned': 指标,
        'quant': {quant_level: {'num_bytes', 'bpp', 'metrics', 'frame_metrics'}},
        'decoded_state': 最小量化级的解码权重（作为下一 clip 的参考）,
    }
    - dynamics_tracker 非 None 时启用帧权重采样（dyn_* 参数控制）；
    - enable_pruning=False 跳过剪枝（P 模型）；
    - reference_state_dict 非 None 时 quant 阶段走残差 QAT/编码（P 模型，DCVC-RT 借鉴）。
    """
    # Stage configs（P 模型可用更少轮数）
    main_epochs = args.epochs if main_epochs is None else main_epochs
    prune_epochs = args.prune_epochs if prune_epochs is None else prune_epochs
    quant_epochs = args.quant_epochs if quant_epochs is None else quant_epochs
    enable_pruning = (args.prune_epochs > 0 and args.prune_ratio > 0.) if enable_pruning is None else enable_pruning
    if not enable_pruning:
        prune_epochs = 0

    results = {'full': None, 'pruned': None, 'quant': {}}

    # 断点续跑：加载快照权重并跳过已完成阶段（base 主训练/剪枝一结束即快照，见下）
    if init_state_dict is not None:
        n = load_bitstream_state(model, init_state_dict)
        logger.info(f'[{model_name}] Resume: loaded {n} snapshot tensors, skip main={skip_main} prune={skip_prune}')

    # Optimizers & schedulers
    opt_sch, prune_opt_sch, quant_opt_sch = [
        get_stage_optimizer_scheduler(args, logger, accelerator, model, train_loader, stage,
                                      epochs_override={'main': main_epochs, 'prune': prune_epochs, 'quant': quant_epochs}[stage])
        for stage in ['main', 'prune', 'quant']]
    opt_sch, prune_opt_sch, quant_opt_sch = accelerator.prepare(*opt_sch), accelerator.prepare(*prune_opt_sch), accelerator.prepare(*quant_opt_sch)

    # Frame/patch-weighted sampling state
    weighted_loader = train_loader
    use_dyn_sampling = dynamics_tracker is not None and getattr(args, 'dyn_enabled', True)
    patch_drop_ratio = float(getattr(args, 'patch_drop_ratio', 0.))
    num_patches = int(np.prod(train_dataset.num_patches))
    is_patch_tracker = isinstance(dynamics_tracker, PatchDynamicsTracker)

    num_params = unwrap_model(model).get_num_parameters()
    checkpoint_manager = CheckpointManager(logger, accelerator, os.path.join(output_dir, 'checkpoints'), args.eval_metric[0])

    # 主训练中途断点续训：从 CheckpointManager 的完整状态（模型+优化器+调度器）恢复
    if resume_checkpoint is not None:
        logger.info(f'[{model_name}] Resume: calling accelerator.load_state on {resume_checkpoint} ...')
        resumed_epoch = checkpoint_manager.load(resume_checkpoint)
        logger.info(f'[{model_name}] Resume: load_state done.')
        epoch = resumed_epoch + 1
        logger.info(f'[{model_name}] Resume from checkpoint {resume_checkpoint} (epoch {resumed_epoch}) -> continue at epoch {epoch}')

    total_epochs = main_epochs + prune_epochs + quant_epochs
    logger.info(f'[{model_name}] Start training for {total_epochs} epochs '
                f'(main {main_epochs} / prune {prune_epochs} / quant {quant_epochs}, pruning={"on" if enable_pruning else "off"})')

    # ============ Main training ============
    # 联合率失真训练（UAR-NVC QEC 式）：量化前向从主训练 epoch 0 激活，
    # 权重与 q_step 在率+失真联合损失下共同优化；率项经 rate_warmup_epochs 线性 ramp-in。
    joint_rate = bool(getattr(args, 'joint_rate', False)) and args.quant_mode in ('qec', 'hyper')

    # D1: 列级结构化稀疏（FaNeRV+TACO 启发）
    col_pruner = None
    col_target = float(getattr(args, 'col_prune_target', 0) or 0)
    if col_target > 0 and not skip_main:
        col_pruner = ColumnPruner(logger, target_col_sparsity=col_target,
                                   start_epoch=getattr(args, 'col_prune_start', 60),
                                   end_epoch=getattr(args, 'col_prune_end', 300),
                                   prune_every=getattr(args, 'col_prune_every', 20))
        logger.info(f'[{model_name}] D1 ColumnPruner: target={col_target}, '
                     f'start={getattr(args, "col_prune_start", 60)}, end={getattr(args, "col_prune_end", 300)}')

    if joint_rate and not skip_main:
        start_quantization(model)
        # 残差参考在主训练就注入（A3）：率损失估计的是残差 θ−θ' 的熵，
        # 与最终残差编码一致；无参考（单模型 C 系列）时跳过
        if reference_state_dict is not None:
            n_ref = set_reference(model, reference_state_dict)
            logger.info(f'[{model_name}] Joint R-D training: residual reference injected into {n_ref} layers.')
        logger.info(f'[{model_name}] Joint rate-distortion training ON: quantization active from main epoch 0 '
                    f'(quant_mode={args.quant_mode}, w_rate={getattr(args, "w_rate", 0.)}, '
                    f'rate_warmup={getattr(args, "rate_warmup_epochs", 15)} epochs)')
    if resume_checkpoint is not None:
        pass  # epoch 已由上面的 checkpoint 恢复设定
    else:
        epoch = main_epochs if skip_main else 0
    if skip_main:
        logger.info(f'[{model_name}] Resume: skip main training ({main_epochs} epochs).')
    # C4：NVRC 式软量化退火（temp 从 t0 线性降到 t1）与 D/R 交替周期
    soft_anneal = getattr(args, 'quant_soft_anneal', '')
    dr_alt = int(getattr(args, 'dr_alternate', 0) or 0)
    if joint_rate and isinstance(soft_anneal, str) and soft_anneal:
        t0, t1 = [float(x) for x in soft_anneal.split(',')]
        logger.info(f'[{model_name}] C4 soft-round anneal: {t0} -> {t1}; D/R alternate: {dr_alt or "off"}')
    else:
        t0 = t1 = 0.

    while epoch < main_epochs:
        train_task.set_progress(epoch / max(main_epochs, 1))
        if joint_rate:
            train_task.set_rate_ramp(min(1., epoch / max(getattr(args, 'rate_warmup_epochs', 15), 1)))
            # C4 软量化退火：按训练进度线性降温（与 NVRC set_epoch 同式）
            if t0 > 0:
                ratio = epoch / max(main_epochs - 1, 1)
                temp = t0 + (t1 - t0) * ratio
                set_soft_round_anneal(model, temp)
        # D1: 列级渐进剪枝（每 epoch 更新）+ 掩码写入权重
        if col_pruner is not None:
            from compression.quant_utils import apply_col_prune
            apply_col_prune(model, col_pruner)
            # D1：用 get_bitstream_state_dict 获取 canonical 权重表（已排除量化参数、
            # 含全部实际权重的规范化键名——与 col_pruner/apply_col_prune 的键空间一致）
            _sd = get_bitstream_state_dict(model)
            col_pruner.step(epoch, {k: v for k, v in _sd.items()
                                     if torch.is_tensor(v) and v.ndim >= 2})
        frame_sum = np.zeros(train_dataset.video_size[0]) if use_dyn_sampling else None
        frame_cnt = np.zeros(train_dataset.video_size[0]) if use_dyn_sampling else None
        patch_sum = np.zeros(num_patches) if (use_dyn_sampling and is_patch_tracker) else None
        patch_cnt = np.zeros(num_patches) if (use_dyn_sampling and is_patch_tracker) else None

        _, _, frame_losses, patch_sum, patch_cnt = \
            train_step(args, logger, '', epoch, model, weighted_loader, opt_sch[0], opt_sch[1],
                       train_task, accelerator, False, frame_sum, frame_cnt, patch_sum, patch_cnt)

        # 训练动态更新 + 采样器重建（帧权重 + dynamicAL patch 丢弃）
        if use_dyn_sampling and frame_losses is not None:
            if is_patch_tracker:
                dynamics_tracker.update(epoch, patch_sum, patch_cnt)
                frame_scores = dynamics_tracker.get_frame_scores()
                patch_scores = dynamics_tracker.get_scores()
            else:
                dynamics_tracker.update(epoch, frame_losses)
                frame_scores = dynamics_tracker.get_scores()
                patch_scores = None

            if (epoch + 1) >= args.dyn_warmup_epochs and (epoch + 1) % args.dyn_update_epochs == 0 and (epoch + 1) < main_epochs:
                # dynamicAL 双权重系统：采样权重（激进）+ 损失权重（保守），同源于 patch 级得分
                if patch_scores is not None:
                    sample_w = compute_patch_train_weights(patch_scores,
                                                          temp=getattr(args, 'dyn_sampling_temp', 1.0),
                                                          w_min=getattr(args, 'dyn_sampling_w_min', 0.5),
                                                          w_max=getattr(args, 'dyn_sampling_w_max', 2.0))
                    # 采样权重余弦窗口：早晚期接近均匀，中期全量差异化
                    sw = getattr(args, 'sampling_window', '')
                    if isinstance(sw, str) and sw:
                        p0, p1, p2, p3 = [float(x) for x in sw.split(',')]
                        p_now = epoch / max(main_epochs, 1)
                        if p_now < p0 or p_now >= p3:
                            wm = 0.
                        elif p_now < p1:
                            wm = 0.5 * (1. - math.cos(math.pi * (p_now - p0) / max(p1 - p0, 1e-6)))
                        elif p_now <= p2:
                            wm = 1.
                        else:
                            wm = 0.5 * (1. + math.cos(math.pi * (p_now - p2) / max(p3 - p2, 1e-6)))
                        # 混合：wm=0 → 全 1.0（均匀），wm=1 → 完全差异化
                        sample_w = 1.0 + wm * (sample_w - 1.0)
                    loss_w = compute_patch_loss_weights(patch_scores,
                                                        alpha=getattr(args, 'dyn_loss_alpha', 0.0),
                                                        clip_delta=getattr(args, 'dyn_loss_clip', 0.2))
                    logger.info(f'[{model_name}] Update dynamics weights: '
                                f'sampling [{sample_w.min():.2f}, {sample_w.max():.2f}] '
                                f'loss [{loss_w.min():.3f}, {loss_w.max():.3f}]')
                    _shutdown_loader_workers(weighted_loader)
                    if getattr(args, 'sampling_mode', 'flat') == 'hierarchical':
                        weighted_loader = make_hierarchical_loader(args, logger, accelerator,
                                                                    train_dataset, sample_w, None)
                    else:
                        weighted_loader = make_weighted_loader(args, logger, accelerator, train_dataset, sample_w)
                    train_task.set_patch_loss_weights(loss_w, train_dataset.num_patches)
                else:
                    # 回退到帧级权重（无 patch 级跟踪器时）
                    fw = compute_frame_weights(frame_scores, temperature=args.dyn_temperature,
                                               max_ratio=args.dyn_max_ratio)
                    pw = frame_patch_weights(fw, train_dataset.num_patches)
                    logger.info(f'[{model_name}] Update sampling (frame-level): mean={fw.mean():.3f} max={fw.max():.3f}')
                    _shutdown_loader_workers(weighted_loader)
                    weighted_loader = make_weighted_loader(args, logger, accelerator, train_dataset, pw.numpy())

        if do_eval(args, epoch, main_epochs):
            _, metrics, _ = eval_step(args, logger, '', epoch, model, eval_loader, eval_task, accelerator,
                                      do_log(args, epoch, main_epochs))
            results['full'] = metrics
            checkpoint_manager.save(epoch, metrics)

        # 周期性压缩探针：每 compression_interval 个 epoch 做一次 prune→QAT→compress→eval
        ci = int(getattr(args, 'compression_interval', 0))
        if ci > 0 and (epoch + 1) % ci == 0 and (epoch + 1) < main_epochs:
            eval_px = np.prod(train_dataset.video_size) if eval_dataset_size is None else eval_dataset_size
            probe = compression_probe(args, logger, accelerator, model, weighted_loader,
                                      eval_loader, train_task, eval_task, output_dir,
                                      epoch + 1, eval_px,
                                      probe_batches=int(getattr(args, 'probe_batches', 100)))
            results.setdefault('probes', []).append(probe)
            # 保存探针结果到 YAML（与磁盘已有记录按 epoch 合并，重启/续训不丢历史）
            import yaml as _yml
            probe_path = os.path.join(output_dir, 'probe_results.yaml')
            merged = []
            if os.path.exists(probe_path):
                try:
                    with open(probe_path) as f:
                        merged = [d for d in _yml.unsafe_load(f) if isinstance(d, dict)]
                except Exception:
                    merged = []
            known = {int(d.get('epoch', -1)) for d in merged}
            for d in results.get('probes', []):
                if int(d.get('epoch', -1)) not in known:
                    merged.append(d)
            merged.sort(key=lambda d: int(d.get('epoch', 0)))
            results['probes'] = merged
            with open(probe_path, 'w') as f:
                _yml.dump(merged, f)

        epoch += 1

    # 主训练一结束立即快照（float，含未剪枝权重）——QAT/压缩阶段失败时可直接从此重跑。
    # skip_main 时（r 系列快照接力）禁止覆写：覆写会用"加载后的状态"破坏接力源快照
    # （2026-09-23 B3r 事故：56 张量源被覆写成 44，后续 resume 只加载到 13 个）。
    if accelerator.is_main_process and not skip_main:
        torch.save(snapshot_model_state(model), os.path.join(output_dir, 'state_after_main.pt'))
        logger.info(f'[{model_name}] Snapshot saved: state_after_main.pt')

    # ============ Pruning fine-tuning ============
    zeros, total = get_sparsity(model)
    if prune_epochs > 0:
        logger.info(f'[{model_name}] Start pruning fine-tuning for {prune_epochs} epochs.')
        zeros, total = set_pruning(args, logger, model, args.prune_ratio, args.prune_weight)
        logger.info(f'Number of pruned parameters: {zeros}    total: {total}')
        logger.info(f'Sparsity: {zeros / total:.4f}')

    while (not skip_prune) and epoch < main_epochs + prune_epochs:
        prune_epoch = epoch - main_epochs
        train_task.set_progress(epoch / max(total_epochs, 1))
        train_step(args, logger, 'Pruning', prune_epoch, model, weighted_loader, prune_opt_sch[0], prune_opt_sch[1],
                   train_task, accelerator, False)

        if (prune_epoch + 1) % args.eval_epochs == 0 or (prune_epoch + 1) == prune_epochs:
            _, metrics, _ = eval_step(args, logger, 'Pruning', epoch, model, eval_loader, eval_task, accelerator,
                                      do_log(args, prune_epoch, prune_epochs))
            results['pruned'] = metrics
            checkpoint_manager.save(epoch, metrics)

        epoch += 1

    # 剪枝阶段一结束立即快照（掩码烘焙进权重）——QAT/压缩阶段失败时可直接从此重跑
    if accelerator.is_main_process:
        torch.save(snapshot_model_state(model), os.path.join(output_dir, 'state_after_prune.pt'))
        logger.info(f'[{model_name}] Snapshot saved: state_after_prune.pt')

    if skip_prune:
        epoch = main_epochs + prune_epochs  # QAT 循环从正确 epoch 起（上面 while 已被跳过）

    # ============ QAT fine-tuning + compression ============
    if quant_epochs > 0:
        logger.info(f'[{model_name}] Start quant fine-tuning for {quant_epochs} epochs.')
        logger.info(f'Quantization level: {sorted(args.quant_level, reverse=True)}    noise: {args.quant_noise}    '
                    f'residual: {reference_state_dict is not None}    deadzone: {getattr(args, "quant_residual_deadzone", 1)}')

    # Set QAT configs
    if getattr(args, 'quant_mode', 'noise') in ('qec', 'hyper'):
        start_quantization(model)  # QEC/Hyper：启动可学习量化步长与熵估计
    else:
        set_quantization(args, logger, model, min(args.quant_level), args.quant_noise, args.quant_ste)
    if reference_state_dict is not None:
        n_ref = set_reference(model, reference_state_dict)
        logger.info(f'[{model_name}] Injected reference weights into {n_ref} residual quantization layers.')

    eval_pixels = np.prod(train_dataset.video_size) if eval_dataset_size is None else eval_dataset_size

    while epoch < total_epochs:
        quant_epoch = epoch - main_epochs - prune_epochs
        train_task.set_progress(epoch / max(total_epochs, 1))
        train_step(args, logger, 'QAT', quant_epoch, model, weighted_loader, quant_opt_sch[0], quant_opt_sch[1],
                   train_task, accelerator, False)

        if do_eval(args, quant_epoch, quant_epochs):
            qat_state_dict = copy.deepcopy(model.state_dict())
            for quant_level in sorted(args.quant_level, reverse=True):
                logger.info(f'[{model_name}] Compress model weights into bitstream')
                logger.info(f'***  Quant level: {quant_level}bits    Sparsity: {zeros / max(total, 1):.4f}')

                num_bytes = compress_bitstream(args, logger, accelerator, model,
                                               os.path.join(output_dir, 'bitstreams'), quant_level,
                                               reference_state_dict=reference_state_dict)
                bits_per_pixel = num_bytes * 8 / eval_pixels

                logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB    BPP: {bits_per_pixel:.4f}')

                # Set model weights to zero for ensuring the correctness
                set_zero(model)

                # Decompress bitstream
                num_bytes2 = decompress(args, logger, accelerator, model,
                                        os.path.join(output_dir, 'bitstreams'), quant_level,
                                        reference_state_dict=reference_state_dict)
                assert num_bytes == num_bytes2

                # Evaluation with the decoded weights
                _, metrics, frame_metrics = eval_step(args, logger, f'Q{quant_level}', epoch, model, eval_loader,
                                                      eval_task, accelerator, do_log(args, quant_epoch, quant_epochs))
                results['quant'][quant_level] = {
                    'num_bytes': num_bytes,
                    'bpp': bits_per_pixel,
                    'metrics': metrics,
                    'frame_metrics': frame_metrics,
                }

                if quant_level == min(args.quant_level):
                    checkpoint_manager.save(epoch, metrics)
                    # 保留最小比特级（质量最高参考）的解码权重作为下一 clip 的参考
                    results['decoded_state'] = get_bitstream_state_dict(model)

                # Restore from the checkpoint（恢复量化前的浮点权重，继续训练或评估下一量化级）
                model.load_state_dict(qat_state_dict)

        epoch += 1

    logger.info(f'[{model_name}] Training completed.')

    return results
