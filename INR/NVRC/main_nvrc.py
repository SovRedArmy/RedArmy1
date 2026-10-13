"""
NVRC main script（NVRC 主脚本：神经视频表示压缩的完整编码流程入口）

整体流程（对应论文 Fig.2 的编解码框架）：
    1. 解析配置 -> 构建 accelerate 加速器/日志/输出目录；
    2. 创建输入视频（训练/评估）与输出视频（重建写回）对象；
    3. 按帧号划分为若干 GoP（intra period），逐组执行：
        a. 创建数据集 / 任务 / DataLoader / NVRC 编解码器 / 优化器；
        b. 训练循环（train_epoch，失真步 + 率失真交替优化）；
        c. model.compress() 将所有网络权重编码成码流；
        d. set_zero() 清零权重 -> model.decompress() 从码流解码回权重（验证码流正确性）；
        e. 用解码后的模型再次评估并统计码率/失真。
"""
from main_utils import *


def main():
    total_start_time = time.time()

    # Set & store args（解析命令行 + YAML 配置，返回统一的 ExperimentConfig）
    args = parse_args()

    # Set accelerator, logger, output dir（创建 accelerate 加速器、logger 与实验输出目录）
    accelerator, logger, output_dir = get_accelerator_logger(args)
    logger.info(f'Output dir: {output_dir}')
    temp_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)  # exist_ok: 支持断点续跑时目录已存在的情况

    # Set seed（固定随机种子以保证可复现）
    logger.info(f'Set seed: {args.seed}')
    accelerate.utils.set_seed(args.seed)

    # Create input/output videos（创建训练/评估用的输入视频对象与重建输出视频对象）
    train_video_in, train_video_out = create_encoding_decoding_videos(temp_dir, args, logger, training=True)
    eval_video_in, eval_video_out = create_encoding_decoding_videos(temp_dir, args, logger, training=False)
    assert all(train_video_in.get_video_size()[d] == eval_video_in.get_video_size()[d] for d in range(3)), \
        f'Input videos have different sizes: {train_video_in.get_video_size()} vs {eval_video_in.get_video_size()}'

    # Create dataset（创建过拟合数据集：整段视频被当作一个数据集，逐 3D patch 读取）
    train_dataset = create_overfit_dataset(args, logger, train_video_in, channel_scale=None, channel_shift=None)
    eval_dataset = create_overfit_dataset(args, logger, eval_video_in, channel_scale=None, channel_shift=None)

    # Create tasks（创建训练/评估任务：定义损失、指标、lambda 与 batch 的解析方式）
    train_task = create_overfit_task(args, logger, train_video_out, channel_scale=None, channel_shift=None,
                                     training=True, device=accelerator.device)
    eval_task = create_overfit_task(args, logger, eval_video_out, channel_scale=None, channel_shift=None,
                                    training=False, device=accelerator.device)

    # Set coding settings（编码设置：起始帧、编码帧数与 GoP 长度 intra_period）
    assert args.start_frame == -1 or \
           (args.start_frame >= 0 and args.start_frame < train_dataset.get_video_size()[0])
    start_frame = args.start_frame if args.start_frame != -1 else 0
    assert args.num_frames == -1 or \
           (args.num_frames > 0 and args.num_frames <= train_dataset.get_video_size()[0] - start_frame)
    num_frames = args.num_frames if args.num_frames != -1 else train_dataset.get_video_size()[0] - start_frame
    assert args.intra_period == -1 or args.intra_period > 0
    intra_period = min(args.intra_period, num_frames) if args.intra_period != -1 else num_frames

    logger.info(f'Encoding settings:')
    logger.info(f'     Start frame - {start_frame}    Num of frames - {num_frames}')
    logger.info(f'     Intra period - {intra_period}')

    intra_idx = 0
    intra_idx_max = (num_frames + intra_period - 1) // intra_period  # GoP 总数（向上取整）

    # Create bitstream（最终完整码流：各组码流前拼接 4 字节长度前缀）
    bs = b''

    # Create buffers（统计各组帧数、码流大小与指标）
    all_num_frames = []
    all_size = []
    all_metrics = []

    while intra_idx < intra_idx_max:
        # 当前 GoP 的起始帧与帧数
        intra_start_frame = start_frame + intra_idx * intra_period
        intra_num_frames = min(intra_period, start_frame + num_frames - intra_start_frame)
        intra_bs = b''

        # Print group info（打印当前组信息）
        logger.info(f'\nStart encoding:')
        logger.info(f'     Group index - {intra_idx}    Group start frame - {intra_start_frame}   Group num of frames - {intra_num_frames}')

        # Set group of frames for coding（把数据集与任务都限定到当前 GoP 的帧范围并重建缓存）
        for dataset in [train_dataset, eval_dataset]:
            dataset.set_frames(intra_start_frame, intra_num_frames)
            dataset.create_cache()

        for task in [train_task, eval_task]:
            task.set_frames(intra_start_frame, intra_num_frames)
            task.create_cache()

        # Create loader（创建当前 GoP 的 DataLoader，batch 即若干 3D patch）
        train_loader = create_overfit_loader(args, logger, accelerator, train_dataset,
                                             start_frame=0, num_frames=intra_num_frames,
                                             training=True)
        eval_loader = create_overfit_loader(args, logger, accelerator, eval_dataset,
                                            start_frame=0, num_frames=intra_num_frames,
                                            training=False)

        # Create codec（构建 NVRC 编解码器：INR 模型 + 分层熵模型，见 compress_models/nvrc.py）
        model = create_codec(output_dir, f'{intra_idx:04d}', args, logger, accelerator,
                                eval_loader, eval_task)

        # Optimizer & scheduler（创建优化器与学习率调度器）
        optimizer, scheduler = create_optimizer_scheduler(output_dir, f'{intra_idx:04d}', args, logger,
                                                          accelerator, model, train_loader)

        # Place model etc. to accelerator（交给 accelerate 设备封装）
        model, train_loader, eval_loader, optimizer, scheduler = \
            accelerator.prepare(model, train_loader, eval_loader, optimizer, scheduler)

        # Profile codec（统计参数量与 MACs）
        num_params, num_params_codec, num_macs, num_macs_rate = \
            profile_codec(output_dir, f'{intra_idx:04d}', args, logger, accelerator, model, eval_loader, eval_task)

        # Restoring training state（创建 checkpoint 管理器；如指定 resume 则恢复训练状态）
        epoch, ckpt_manager = create_ckpt_and_restore(output_dir, f'{intra_idx:04d}', args, logger,
                                                      accelerator, model,
                                                      metric=eval_task.get_metrics())

        # Training loop（训练循环：每个 epoch 先训练、必要时评估并保存 checkpoint）
        start_train_time = time.time()
        if not args.eval_only:
            if epoch < args.epochs:
                logger.info(f'Start training for {args.epochs - epoch} epochs.')
            else:
                logger.info(f'Training is already completed. Skip training.')
            while epoch < args.epochs:
                # Training（单个训练 epoch：失真步 D-step 与率步 R-step 交替）
                train_epoch(output_dir, f'{intra_idx:04d}/train', epoch, args, logger,
                            accelerator, model, train_loader, train_task,
                            optimizer, scheduler, do_log(args, epoch, args.epochs))

                # Evaluation（周期性评估并保存 checkpoint）
                if do_eval(args, epoch, args.epochs):
                    _, metrics = eval_epoch(output_dir, f'{intra_idx:04d}/eval', epoch, args, logger,
                                            accelerator, model, eval_loader, eval_task,
                                            do_log(args, epoch, args.epochs))
                    ckpt_manager.save(epoch, metrics)

                epoch += 1
        else:
            logger.info('Skip training. Evaluation only.')
            # Evaluation（仅评估模式）
            _, metrics = eval_epoch(output_dir, f'{intra_idx:04d}/eval', epoch, args, logger,
                                    accelerator, model, eval_loader, eval_task, True)
            ckpt_manager.save(epoch, metrics)

        train_time = time.time() - start_train_time
        logger.info(f'Training completed in: {train_time:.2f}s')

        # Write bitstream（把训练好的全部权重写入码流：三层分层编码 theta_hat / phi_hat / psi_hat）
        logger.info(f'Encoding model weights to bitstream.')
        start_enc_time = time.time()

        intra_bs = model.compress(intra_bs)  # 压缩：网络参数 + 量化参数 + 熵模型参数
        intra_bs_len = len(intra_bs)
        bs += intra_bs_len.to_bytes(4) + intra_bs  # 长度前缀(4字节) + 组码流
        write_bitstream(output_dir, f'bitstream', bs)
        write_bitstream(output_dir, f'{intra_idx:04d}', intra_bs)

        enc_time = time.time() - start_enc_time
        logger.info(f'Encoding time: {enc_time:.2f}s')
        logger.info(f'Current total bitstream size: {len(bs) * 8}')
        logger.info(f'Group bitstream size: {intra_bs_len * 8}.')

        # Set model weights to zero for ensuring the correctness（清零全部权重，确保随后的结果确实来自码流解码）
        set_zero(model)

        # Decompress bitstream and evaluate（从码流解码回权重，验证编解码一致性）
        logger.info(f'Decoding model weights from bitstream.')
        start_dec_time = time.time()
        
        intra_bs = read_bitstream(output_dir, f'{intra_idx:04d}')
        intra_bs = model.decompress(intra_bs)  # 解压：按与 compress 相反的顺序恢复各层参数

        dec_time = time.time() - start_dec_time
        logger.info(f'Decoding time: {dec_time:.2f}s')
        assert len(intra_bs) == 0, f'Bitstream is not empty after decompress: {len(intra_bs)}'

        # Save decoded model（保存解码后的模型权重）
        save_model(output_dir, f'{intra_idx:04d}/decoded', None, model)

        # Evaluation（用解码后的（量化）模型重新评估，得到最终 RD 性能）
        start_eval_time = time.time()
        logger.info(f'Start evaluating the decoded model.')
        _, metrics = eval_epoch(output_dir, f'{intra_idx:04d}/decoded', None, args, logger,
                                accelerator, model, eval_loader, eval_task, True)
        eval_time = time.time() - start_eval_time
        logger.info(f'Evaluation completed in: {eval_time:.2f}s')

        # Concatenate outputs（把本组重建帧拼接到整段视频输出目录）
        concate_outputs(os.path.join(output_dir, 'outputs', f'{intra_idx:04d}/decoded'),
                        os.path.join(output_dir, 'outputs', 'all', 'decoded'), args.eval_data.dataset,
                        keep_path=False)

        # Get detailed rd performance（统计各部分参数的码率占比）
        profile_model_rate_stat(logger, model, eval_loader, eval_task)

        # Log results（记录本组实际码率 bpp 与各项指标）
        actual_rate = (4 + intra_bs_len) * 8  # 组码流 bits（含 4 字节长度前缀）
        actual_bpp = actual_rate / math.prod((intra_num_frames,) + eval_task.get_video_size()[1:])
        all_num_frames.append(intra_num_frames)
        all_size.append(actual_rate)
        all_metrics.append({**{'params': num_params, 'params_codec': num_params_codec,
                                'macs': num_macs, 'macs_rate': num_macs_rate,
                                'train_time': train_time, 'eval_time': eval_time,
                                'ent_enc_time': enc_time, 'ent_dec_time': dec_time,
                                'bpp': actual_bpp, 'bpp_est': metrics['bpp']},
                            **{k: v for k, v in metrics.items() if k not in ['bpp']}})
        write_results(output_dir, f'{intra_idx:04d}', all_size[-1], all_metrics[-1])

        # End of group（本组编码结束，清理加速器状态并进入下一组）
        logger.info(f'Completed group encoding.\n')
        accelerator.clear()
        intra_idx += 1

    # Log results（汇总所有 GoP：总码流大小与按帧数加权的平均指标）
    total_num_frames = sum(all_num_frames)
    total_size = sum(all_size)
    total_metrics = {k + '_avg': sum([m[k] * n for m, n in zip(all_metrics, all_num_frames)]) / total_num_frames \
                     for k in all_metrics[0].keys()}
    write_results(output_dir, 'all', total_size, total_metrics)

    logger.info(f'Total bitstream size: {total_size}')
    for k, v in total_metrics.items():
        logger.info(f'Average {k}: {v:.4f}')

    # Complete encoding
    total_time = time.time() - total_start_time
    logger.info(f'Encoding completed in: {total_time:.2f}s')
    logger.info(f'Output are located in: {output_dir}')
    accelerator.end_training()
    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()