"""
HiNeRV 训练主脚本（入口）。
完整流程（对应论文的训练→剪枝→量化流水线）：
  1. 解析参数(parse_args)、初始化 accelerate 分布式环境与日志；
  2. 构建 VideoRegressionTask（训练/评估任务）与数据集/加载器；
  3. build_model 构建 HiNeRV，统计参数量与 MACs；
  4. initial_parametrizations 注册剪枝掩码与量化噪声参数化；
  5. 三个训练循环：主训练(train_step×epochs) → 剪枝微调(set_pruning) → QAT 微调(set_quantization)；
  6. QAT 评估时 compress_bitstream 编码成码流、set_zero+decompress 解码回权重，
     再 eval_step 验证，从而得到真实的 bpp-PSNR 点。
用法见 README.md（accelerate launch hinerv_main.py ...）。
"""

from utils import *
from datasets import create_dataset, create_loader, set_dataset_args
from hinerv_tasks import VideoRegressionTask, set_task_args
from hinerv_compress import *

from deepspeed.profiling.flops_profiler import get_model_profile

parser = argparse.ArgumentParser()


# Script
group = parser.add_argument_group('Script parameters')
group.add_argument('--model', default='HiNeRV', type=str, help='Model to build, e.g. HiNeRV.')

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
group.add_argument('--bitstream', default='', type=str, help='Bitstream path for evaluation.')
group.add_argument('--bitstream-q', default='', type=str, help='Bitstream quantization level for evaluation.')

group.add_argument('--grad-accum', type=int, default=1, help='Number of gradient accumuluation steps.')

group.add_argument('--profile', action='store_true', default=False, help='Run profiler.')
group.add_argument('--seed', type=int, default=0, help='Random seed.')


# Dataset
set_dataset_args(parser)

# Task
set_task_args(parser)

# Optimizer & scheduler
set_optimizer_args(parser)

# Compression
set_compression_args(parser)


def get_stage_optimizer_scheduler(args, logger, accelerator, model, loader, stage):
    """
    为三个训练阶段（main 主训练 / prune 剪枝微调 / quant 量化微调）
    分别创建独立的优化器与学习率调度器：
      - prune/quant 阶段使用各自的 lr 缩放系数（prune_lr_scale / quant_lr_scale，默认量化阶段为 0.1 倍）
      - 各阶段有独立的 epochs 与 warmup 配置
    """
    args = copy.deepcopy(args)
    if stage == 'main':
        args.epochs = max(args.epochs, 1)
    elif stage == 'prune':
        # 剪枝阶段：lr 等比缩放，epochs/warmup 换成剪枝配置
        args.lr *= args.prune_lr_scale
        args.warmup_lr *= args.prune_lr_scale
        args.min_lr *= args.prune_lr_scale
        args.epochs = max(args.prune_epochs, 1)
        args.warmup_epochs = args.prune_warmup_epochs
    elif stage == 'quant':
        # 量化阶段：lr 等比缩放（默认 0.1），epochs/warmup 换成量化配置
        args.lr *= args.quant_lr_scale
        args.warmup_lr *= args.quant_lr_scale
        args.min_lr *= args.quant_lr_scale
        args.epochs = max(args.quant_epochs, 1)
        args.warmup_epochs = args.quant_warmup_epochs
    else:
        raise ValueError
    logger.info(f'Create optimizer & scheduler for {stage} stage')
    return get_optimizer_scheduler(args, logger, accelerator, model, loader)


def train_step(args, logger, suffix, epoch, model, loader, optimizer, scheduler, task, accelerator, log_output):
    """
    一个 epoch 的训练循环：
    遍历 loader 中的补丁 batch → task.step(前向+损失) → accelerator.backward(反向)
    → 梯度裁剪(max_norm) → optimizer.step / scheduler.step_update，
    每 1/4 epoch 汇总打印一次损失与指标。
    """
    start_time = time.time()
    model.train()

    accum_loss = None
    accum_metrices = None
    reduced_loss = None
    reduced_metrics = None
    counts = 0
    samples = 0

    num_updates = epoch * len(loader)
    scheduler.step(epoch=epoch)
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        # Train step
        with accelerator.accumulate(model):
            # 前向 + 损失 + 指标（VideoRegressionTask.step）
            inputs, _, outputs, loss, metrics = task.step(model, loader, batch)
            mean_loss = loss.mean()
            mean_metrics = {k: v.mean() for k, v in metrics.items()}

            # 反向传播（支持梯度累积与混合精度）
            accelerator.backward(mean_loss)
            if accelerator.sync_gradients and args.max_norm:
                # 全局梯度范数裁剪（论文：max_norm=1.0）
                accelerator.unscale_gradients(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm, args.norm_type)

            optimizer.step()
            scheduler.step_update(num_updates=num_updates)
            optimizer.zero_grad()
            num_updates += 1

        # 累计损失/指标（用于周期性日志）
        if i == 0:
            accum_loss = mean_loss.detach()
            accum_metrices = {k: mean_metrics[k].detach() for k in mean_metrics}
        else:
            accum_loss = accum_loss + mean_loss.detach()
            accum_metrices = {k: accum_metrices[k] + mean_metrics[k].detach() for k in mean_metrics}

        counts += 1
        samples += inputs['batch_size'] * accelerator.num_processes

        if (i + 1) % (len(loader) // 4) == 0 or (i + 1) == len(loader):
            # Accumulate loss and metrics globally
            # 跨进程归约后打印训练日志（lr、吞吐 img/s、loss、指标）
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean')
            reduced_metrics = accelerator.reduce(accum_metrices, reduction='mean')
            # Logging loss & metrics
            log_msg = f'Train' + ('' if not suffix else f' ({suffix})') +  f' - Epoch {epoch} [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    lr: {np.mean([group["lr"] for group in optimizer.param_groups]):.2e}'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss.item() / counts:.4f}'
            for k, v in reduced_metrics.items():
                log_msg = log_msg + f'    {k}: {v.item() / counts:.4f}'
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_eval(dir_name=f'{epoch}_{suffix.lower()}', inputs=inputs, outputs=outputs, metrics=metrics)

    return reduced_loss.item() / counts, {k: v.item() / counts for k, v in reduced_metrics.items()}


def eval_step(args, logger, suffix, epoch, model, loader, task, accelerator, log_output):
    """
    一个 epoch 的评估循环：与 train_step 类似但 no_grad、不更新参数，
    返回平均损失与指标（psnr / ms_ssim）。
    """
    start_time = time.time()
    model.eval()

    accum_loss = None
    accum_metrices = None
    reduced_loss = None
    reduced_metrics = None
    counts = 0
    samples = 0

    for i, batch in enumerate(loader):
        # Eval step
        with torch.no_grad():
            inputs, _, outputs, loss, metrics = task.step(model, loader, batch)
            mean_loss = loss.mean()
            mean_metrics = {k: v.mean() for k, v in metrics.items()}

        if i == 0:
            accum_loss = mean_loss
            accum_metrices = {k: mean_metrics[k] for k in mean_metrics}
        else:
            accum_loss = accum_loss + mean_loss
            accum_metrices = {k: accum_metrices[k] + mean_metrics[k] for k in mean_metrics}

        counts += 1
        samples += inputs['batch_size'] * accelerator.num_processes

        if (i + 1) % (len(loader) // 4) == 0 or (i + 1) == len(loader):
            # Accumulate loss and metrics globally
            # 跨进程归约后打印评估日志
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean')
            reduced_metrics = accelerator.reduce(accum_metrices, reduction='mean')
            # Logging loss & metrics
            log_msg = f'Eval' + ('' if not suffix else f' ({suffix})') +  f' - [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss.item() / counts:.4f}'
            for k, v in reduced_metrics.items():
                log_msg = log_msg + f'    {k}: {v.item() / counts:.4f}'
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_eval(dir_name=f'{epoch}' + ('' if not suffix else f'_{suffix.lower()}'), inputs=inputs, outputs=outputs, metrics=metrics)

    return reduced_loss.item() / counts, {k: v.item() / counts for k, v in reduced_metrics.items()}


def do_eval(args, epoch, num_epochs): 
    """判断当前 epoch 是否需要评估（每 eval_epochs 个 epoch 或最后一个 epoch）。"""
    return (epoch + 1) % args.eval_epochs == 0 or (epoch + 1) == num_epochs


def do_log(args, epoch, num_epochs):
    """判断当前 epoch 是否需要保存输出图像（log_epochs=-1 表示只在最后保存）。"""
    return (args.log_epochs > 0 and (epoch + 1) % args.log_epochs == 0) or \
           (args.log_epochs == -1 and (epoch + 1) == num_epochs)


def main():
    """主函数：组装全部训练/压缩流程（详见模块 docstring）。"""
    start_time = datetime.datetime.now()

    # Set & store args
    # 解析命令行参数（脚本参数 + 模型专属参数由 set_args 动态注册）
    args = parse_args(parser)

    # Set accelerator, logger, output dir
    # 初始化 accelerate（混合精度/分布式）与 logger、输出目录（保存 args.yaml）
    accelerator, logger, output_dir = get_accelerator_logger(args)
    logger.info(f'Output dir: {output_dir}')

    # Set seed
    logger.info(f'Set seed: {args.seed}')
    accelerate.utils.set_seed(args.seed, device_specific=True)

    # Task setting
    # 训练任务与评估任务（相同类，不同配置：训练用 patch、评估用整帧）
    train_task = VideoRegressionTask(args, logger, accelerator, root=os.path.join(output_dir, 'train_output'), 
                                     training=True, enable_log_eval=False)
    eval_task = VideoRegressionTask(args, logger, accelerator, root=os.path.join(output_dir, 'eval_output'),
                                    training=False, enable_log_eval=args.log_eval)

    # Optimize steps
    # 若启用 torch.compile(dynamo)，编译任务的 step 函数
    if accelerator.state.dynamo_plugin.backend != accelerate.utils.DynamoBackend.NO:
        # Some precision issues with max-autotune exist, so the default cfg is used here.
        train_task.step = torch.compile(train_task.step) #, **accelerator.state.dynamo_plugin.to_kwargs())
        eval_task.step = torch.compile(eval_task.step) #, **accelerator.state.dynamo_plugin.to_kwargs())

    # Datasets & Loaders
    # 训练集按 patch_size 切补丁（随机采样），评估集按整帧加载
    logger.info(f'Create training dataset & loader: {os.path.join(args.dataset, args.dataset_name)}')
    train_dataset = create_dataset(logger=logger, args=args, training=True)
    train_loader = create_loader(logger=logger, args=args, training=True, dataset=train_dataset)

    logger.info(f'Create evaluation dataset & loader: {os.path.join(args.dataset, args.dataset_name)}')
    eval_dataset = create_dataset(logger=logger, args=args, training=False)
    eval_loader = create_loader(logger=logger, args=args, training=False, dataset=eval_dataset)

    # Model
    # 构建模型：先用一个 batch 解析输入尺寸，再调用模型的 build_model
    logger.info(f'Create model: {args.model}')
    model = model_cls[args.model].build_model(args, logger, train_task.parse_input(train_loader, next(iter(train_loader)))[0])
    logger.info(f'Model info:')
    logger.info(model)

    # Compute number of parameters & MACs
    # 统计参数量（分类统计）与 MACs（deepspeed profiler）
    logger.info(f'Number of parameters:')
    num_params = model.get_num_parameters()
    for k, v in num_params.items():
        logger.info(f'    {k}: {v / 10**6:.2f}M')

    # Compute MACs
    with torch.no_grad():
        model.eval()
        _, macs, _ = get_model_profile(model=model, 
                                       args=[eval_task.parse_input(eval_loader, next(iter(eval_loader)))[0]],
                                       print_profile=False, detailed=False, warm_up=1, as_string=False)
        macs /= args.eval_batch_size
    logger.info(f'MACs: {macs / 10 ** 9 :.2f}G')

    # Pruning & quanization
    # 注册剪枝掩码与量化噪声的参数化（parametrizations）
    initial_parametrizations(args, logger, model)

    # Place model & loaders
    # 交给 accelerator 管理（设备移动/混合精度/DDP 包装）
    model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)

    # Optimizer & scheduler
    # 三个阶段各自的优化器与调度器
    logger.info(f'Create scheduler: {args.sched}')
    opt_sch, prune_opt_sch, quant_opt_sch = [get_stage_optimizer_scheduler(args, logger, accelerator, model, train_loader, stage) for stage in ['main', 'prune', 'quant']]

    # Place optimizer & scheduler
    opt_sch, prune_opt_sch, quant_opt_sch = accelerator.prepare(*opt_sch), accelerator.prepare(*prune_opt_sch), accelerator.prepare(*quant_opt_sch)

    # Restoring training state
    # 断点续训：从 checkpoint 恢复（epoch 继续往后）
    checkpoint_manager = CheckpointManager(logger, accelerator, os.path.join(output_dir, 'checkpoints'), args.eval_metric[0])
    if args.resume:
        logger.info(f'Restore from training state: {args.resume}')
        epoch = checkpoint_manager.load(args.resume) + int(not args.eval_only)
    else:
        epoch = 0

    # Restore from bitstream
    # 或直接从压缩码流恢复权重（评估压缩后模型）
    if args.bitstream:
        logger.info(f'Restore model weights from bitstream')
        num_bytes = decompress(args, logger, accelerator, model, os.path.join(args.bitstream, 'bitstreams'), int(args.bitstream_q))
        bits_per_pixel = num_bytes * 8 / np.prod(eval_dataset.video_size)

        logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
        logger.info(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')


    if not args.eval_only:
        # Start training
        # 训练总轮数 = 主训练 + 剪枝微调 + 量化微调
        logger.info(f'Start training for {args.epochs + args.prune_epochs + args.quant_epochs} epochs.')
        logger.info(f'    Number of training epochs: {args.epochs}')
        logger.info(f'    Number of pruning fine-tuning epochs: {args.prune_epochs}')
        logger.info(f'    Number of quant fine-tuning epochs: {args.quant_epochs}')

        # best_metrics 跟踪 full/pruned/qat 以及各量化级 Q8/Q7/Q6 的最优结果
        best_metrics = BestMetricTracker(logger, accelerator, os.path.join(output_dir, 'results'), 
                                        {**{k: ['size', args.eval_metric[0]] for k in ['full', 'pruned', 'qat']},
                                         **{f'Q{quant_level}': ['bpp', args.eval_metric[0]] for quant_level in sorted(args.quant_level, reverse=True)}})
        zeros, total = get_sparsity(model)

        # Main Training loop
        # 阶段 1：主训练（论文默认 300 epochs，随机补丁采样）
        if epoch < args.epochs:
            logger.info(f'Start main training for {args.epochs} epochs.')
        while epoch < args.epochs:
            # Training
            train_step(args, logger, '', epoch, model, train_loader, opt_sch[0], opt_sch[1], train_task, accelerator, False)

            # Evaluation
            if do_eval(args, epoch, args.epochs):
                _, metrics = eval_step(args, logger, '', epoch, model, eval_loader, eval_task, accelerator,
                                       do_log(args, epoch, args.epochs))
                best_metrics.update('full', num_params['All'], metrics)
                checkpoint_manager.save(epoch, metrics)

            epoch += 1


        # Pruning fine-tuning loop
        # 阶段 2：按幅度剪枝（改进的评分 |θ|/P^λ）后微调（论文：15% 剪枝 + 60 epochs）
        if epoch < args.epochs + args.prune_epochs:
            logger.info(f'Start pruning fine-tuning for {args.prune_epochs} epochs.')
            zeros, total = set_pruning(args, logger, model, args.prune_ratio, args.prune_weight)
            logger.info(f'Number of pruned parameters: {zeros}    total: {total}')
            logger.info(f'Sparsity: {zeros / total:.4f}')

        while epoch < args.epochs + args.prune_epochs:
            prune_epoch = epoch - args.epochs
            train_step(args, logger, 'Pruning', prune_epoch, model, train_loader, prune_opt_sch[0], prune_opt_sch[1], train_task, accelerator, False)
            
            # Evaluation
            if (prune_epoch + 1) % args.eval_epochs == 0 or (prune_epoch + 1) == args.prune_epochs:
                _, metrics = eval_step(args, logger, 'Pruning', epoch, model, eval_loader, eval_task, accelerator,
                                       do_log(args, prune_epoch, args.prune_epochs))
                best_metrics.update('pruned', num_params['All'], metrics)
                checkpoint_manager.save(epoch, metrics)

            epoch += 1


        # QAT fine-tuning loop
        # 阶段 3：量化感知训练（Quant-Noise，默认噪声比例 0.9、不用 STE）
        if epoch < args.epochs + args.prune_epochs + args.quant_epochs:
            logger.info(f'Start quant fine-tuning for {args.quant_epochs} epochs.')
            logger.info(f'Quantization level: {sorted(args.quant_level, reverse=True)}')
            logger.info(f'Quantization noise: {args.quant_noise}')

        # Set QAT configs
        # 设置量化参数化：bitwidth=min(quant_level)、noise_ratio=quant_noise、是否 STE
        set_quantization(args, logger, model, min(args.quant_level), args.quant_noise, args.quant_ste)

        while epoch < args.epochs + args.prune_epochs + args.quant_epochs:
            quant_epoch = epoch - args.epochs - args.prune_epochs
            train_step(args, logger, 'QAT', quant_epoch, model, train_loader, quant_opt_sch[0], quant_opt_sch[1], train_task, accelerator, False)

            # Evaluation
            # 每次评估时对每个量化级完整走一遍：压缩→置零→解压→评估→恢复权重
            if do_eval(args, quant_epoch, args.quant_epochs):
                qat_state_dict = copy.deepcopy(model.state_dict())
                for quant_level in sorted(args.quant_level, reverse=True):
                    # Compress bitstream
                    # 权重→整数→算术编码→码流，得到真实字节数与 bpp
                    logger.info(f'Compress model weights into bitstream')
                    logger.info(f'***  Quant level: {quant_level}bits')
                    logger.info(f'***  Sparsity: {zeros / total:.4f}')

                    num_bytes = compress_bitstream(args, logger, accelerator, model, os.path.join(output_dir, 'bitstreams'), quant_level)
                    bits_per_pixel = num_bytes * 8 / np.prod(eval_dataset.video_size)

                    logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
                    logger.info(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')

                    # Set model weights to zero for ensuring the correctness
                    # 先清零权重，确保后续评估用的是解压结果而非内存中的原权重
                    set_zero(model)

                    # Decompress bitstream
                    # 码流→反量化→恢复权重
                    logger.info(f'Decompress model weights from bitstream')
                    decompress(args, logger, accelerator, model, os.path.join(output_dir, 'bitstreams'), quant_level)

                    # Evaluation
                    # 用解压后的（有损）权重评估，得到该 bpp 下的 PSNR/MS-SSIM
                    _, metrics = eval_step(args, logger, f'Q{quant_level}', epoch, model, eval_loader, eval_task, accelerator,
                                           do_log(args, quant_epoch, args.quant_epochs))
                    best_metrics.update(f'Q{quant_level}', bits_per_pixel, metrics)

                    if quant_level == min(args.quant_level): # use the least bitwidth for checkpointing metric
                        checkpoint_manager.save(epoch, metrics)

                    # Restore from the checkpoint
                    # 恢复量化前的浮点权重，继续训练或评估下一个量化级
                    model.load_state_dict(qat_state_dict)

            epoch += 1

        # Complete training
        train_time = datetime.datetime.now() - start_time

        logger.info(f'Training completed in: {train_time}')
        logger.info(f'Output are located in: {output_dir}')

    else:
        # Evaluation
        # 仅评估模式：--eval-only（可配 --bitstream 直接评估码流）
        _, metrics = eval_step(args, logger, None, epoch, model, eval_loader, eval_task, accelerator, True)


if __name__ == '__main__':
    main()