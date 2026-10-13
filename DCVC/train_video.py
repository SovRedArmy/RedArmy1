# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# DCVC-UF 视频模型训练入口（论文的级联多阶段训练策略，Sec.4.1）：
# stage0：短序列（1+N, 1+2N, 1+4N 帧）训练基础能力（不级联，chunk 间 detach）；
# stage1：17→33 帧级联训练（is_cascaded，chunk 间不 detach）；
# stage2：512 patch + 33→65 帧；stage3：97→129 帧长序列（论文 128 帧训练）。
# I 帧参考由已训练好的帧内模型 DMCI（recon_only）在线生成，仅 P 模型参与梯度更新。

import argparse
import math
import sys
import time
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from src.datasets.video_dataset import VideoFolder
from src.models.image_model import DMCI
from src.utils.common import ModelStructure, save_status, start_train, save_ckpt, \
    get_current_device, get_training_lambdas, get_dataloader, init_train,  cleanup_train, \
    load_existing_weights, get_state_dict, wrap_ddp


def get_training_strategy(training_scheduling, frame_delay, model_structure):
    # 各阶段的训练策略表：每行为 [起始 epoch, 学习率, 每样本帧数, 是否级联,
    # patch 宽, patch 高]。LD（N=1）与 HT（N=8）分别有一套超参
    # epoch is for referencing purpose
    # lr is the learning rate for the current epoch
    if model_structure == ModelStructure.LD:
        if training_scheduling == 'stage0':
            training_strategy = \
                [[0,   1e-4, 2, False, 256, 256]] * 5 + \
                [[5,   1e-4, 3, False, 256, 256]] * 5 + \
                [[10,  1e-4, 6, False, 256, 256]] * 45 + \
                [[55,  1e-4, 6, False, 256, 256]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        elif training_scheduling == 'stage1':
            training_strategy = \
                [[0,   5e-5,  8, True,  256, 256]] * 5 + \
                [[5,   5e-5, 16, True,  256, 256]] * 5 + \
                [[10,  5e-5, 24, True,  256, 256]] * 5 + \
                [[15,  5e-5, 32, True,  256, 256]] * 15 + \
                [[30,  5e-6, 32, True,  256, 256]] * 7 + \
                [[37,  5e-6, 32, True,  256, 256]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        elif training_scheduling == 'stage2':
            training_strategy = \
                [[0,   5e-5, 33, True,  512, 512]] * 14 + \
                [[14,  5e-6, 33, True,  512, 512]] * 4 + \
                [[18,  2e-5, 49, True,  512, 512]] * 7 + \
                [[25,  2e-6, 49, True,  512, 512]] * 2 + \
                [[27,  5e-6, 65, True,  512, 512]] * 7 + \
                [[34,  2e-6, 65, True,  512, 512]] * 6 + \
                [[40,  2e-6, 65, True,  512, 512]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        elif training_scheduling == 'stage3':
            training_strategy = \
                [[0,   2e-6, 97,  True,  512, 512]] * 2 + \
                [[2,   5e-7, 129, True,  512, 512]] * 2 + \
                [[4,   5e-7, 129, True,  512, 512]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        else:
            assert False

        return training_strategy

    if training_scheduling == 'stage0':
        training_strategy = \
            [[0,   1e-4, 1 + 1 * frame_delay, False, 256, 256]] * 5 + \
            [[5,   1e-4, 1 + 2 * frame_delay, False, 256, 256]] * 5 + \
            [[10,  1e-4, 1 + 4 * frame_delay, False, 256, 256]] * 35 + \
            [[45,  1e-4, 1 + 4 * frame_delay, False, 256, 256]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    elif training_scheduling == 'stage1':
        training_strategy = \
            [[0,   5e-5, 17, True,  256, 256]] * 2 + \
            [[2,   5e-5, 25, True,  256, 256]] * 1 + \
            [[3,   5e-5, 33, True,  256, 256]] * 3 + \
            [[6,   5e-6, 33, True,  256, 256]] * 4 + \
            [[10,  5e-6, 33, True,  256, 256]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    elif training_scheduling == 'stage2':
        training_strategy = \
            [[0,   5e-5, 33, True,  512, 512]] * 10 + \
            [[10,  5e-5, 49, True,  512, 512]] * 10 + \
            [[20,  1e-5, 65, True,  512, 512]] * 12 + \
            [[32,  2e-6, 65, True,  512, 512]] * 8 + \
            [[40,  2e-6, 65, True,  512, 512]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    elif training_scheduling == 'stage3':
        training_strategy = \
            [[0,   1e-5, 97,  True,  512, 512]] * 2 + \
            [[2,   2e-6, 129, True,  512, 512]] * 2 + \
            [[4,   2e-6, 129, True,  512, 512]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    else:
        assert False

    return training_strategy


def parse_args(argv):
    # 命令行参数：λ 范围、I 帧模型路径、模型结构、训练阶段、数据集与保存目录等
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('--lambdas', type=float, nargs='+', required=True)
    parser.add_argument('--model_path_i', type=str, required=True)
    parser.add_argument('--model_structure', type=str, default='ld', choices=['htl', 'hts', 'ld'])
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Dataloaders worker per trainer')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save models')
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--training_scheduling', type=str, default='stage0',
                        choices=['stage0', 'stage1', 'stage2', 'stage3'],
                        help='How to schedule the training strategy')

    args = parser.parse_args(argv)
    return args


def train_one_epoch(p_net, i_net, args, dataloader, optimizer, epoch, rank,
                    frame_delay, model_structure):
    # 训练一个 epoch：按策略表设定 lr/序列长度/patch 尺寸；
    # 级联训练时长序列用梯度检查点省显存；
    # 每个 batch：I 帧参考（DMCI 推理）→ 逐 chunk 前向 → loss.backward →
    # 梯度裁剪（max_norm=0.2，非有限范数则跳过该 batch）→ optimizer.step
    p_net_module = p_net.module if isinstance(p_net, DDP) else p_net
    training_strategy = get_training_strategy(args.training_scheduling, frame_delay,
                                              model_structure)
    p_net.train()
    device = next(p_net.parameters()).device

    idx = min(len(training_strategy) - 1, epoch)
    _, lr, train_seq_length, is_cascaded, patch_width, patch_height = training_strategy[idx]

    for g in optimizer.param_groups:
        g['lr'] = lr

    world_size = dist.get_world_size() if rank >= 0 else 1
    use_ckpt = is_cascaded and patch_width > 256  # 512 patch 级联训练时开梯度检查点
    p_net_module.set_use_ckpt(use_ckpt)

    # 动态更新数据集的采样帧数与 patch 尺寸
    dataloader.dataset.set_frame_num(train_seq_length)
    dataloader.dataset.set_patch_size(patch_width, patch_height)

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        batch = [t.to(device) for t in batch]
        batch_size = batch[0].size(0)
        frame_nums = len(batch) - 2  # 去掉末尾的 qp 与 lambda 两个标量
        frame_stop = 2 if is_cascaded else frame_nums  # 级联时 x 为 chunk 列表
        qp = batch[-2]
        curr_lambdas = batch[-1]

        print_loss_info = rank <= 0 and i % (100 if train_seq_length >= 8 else 200) == 0

        # I 帧参考：用冻结的帧内模型在线重建（长序列或级联时才需要）
        ref_frame = batch[0]
        if train_seq_length > 1 + 1 * frame_delay or is_cascaded:
            with torch.inference_mode():
                ref_frame = i_net.forward_one_frame(ref_frame, qp, recon_only=True)

        # 重置 P 模型状态并注入 I 帧参考特征
        p_net_module.clear_dpb()
        p_net_module.add_ref_feature_from_frame(ref_frame, apply_feature_adaptor=False)

        # 逐 chunk（或逐帧）前向 + 反向
        for frame_idx in range(1, frame_stop):
            if is_cascaded:
                cur_frame = batch[1:-2]  # chunk 列表（一次前向多 chunk 级联）
            else:
                cur_frame = batch[frame_idx]
            loss, info = p_net(cur_frame, qp, lambdas=curr_lambdas, get_loss_info=print_loss_info,
                               curr_poc=frame_idx)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪；NaN/Inf 时丢弃该 batch
            total_norm = clip_grad_norm_(p_net.parameters(), max_norm=0.2,
                                         error_if_nonfinite=False).item()
            if math.isnan(total_norm) or math.isinf(total_norm):
                if rank <= 0:
                    print('non-finite norm, skip this batch')
                continue

            optimizer.step()

        if print_loss_info:
            # 周期性打印训练日志（进度、MSE、bpp、loss、lr）
            numer = i * batch_size * world_size
            denom = len(dataloader.dataset)
            t1 = time.time()
            print(
                f'Time: {t1-t0:.3f} seconds, Train epoch {epoch}:'
                f' [{numer} / {denom} ({100. * numer / denom:.0f}%)]'
                f' MSE: {info['mse']} | Bpp y: {info['bpp_y']} | Bpp z: {info['bpp_z']} |'
                f' Losses: {info['losses']} | lr: {optimizer.param_groups[0]['lr']:.1e}'
            )
            t0 = t1


def train(rank, args):
    # 训练主函数（rank=-1 为单进程）：构建数据集与 P 模型（LD 或 HT 结构），
    # 加载 I 帧模型（冻结，仅推理）→ 恢复断点/预训练权重 → DDP 包装 →
    # AdamW 优化器 → 逐 epoch 调 train_one_epoch 并 save_status
    world_size = init_train(rank, args.save_dir)
    model_structure = ModelStructure(args.model_structure)
    if model_structure == ModelStructure.LD:
        from src.models.video_model_ld import DMC, g_frame_delay as frame_delay
    else:
        from src.models.video_model_ht import DMC, g_frame_delay as frame_delay
    train_dataset = VideoFolder(args.train_dataset, 256, 256, DMCI.qp_num(),
                                get_training_lambdas(args.lambdas, DMCI.qp_num()),
                                group_of_pictures=frame_delay)

    # I 帧模型：加载预训练权重并冻结（eval，不参与优化）
    i_state_dict = get_state_dict(args.model_path_i)
    i_net = DMCI()
    i_net.load_state_dict(i_state_dict)
    i_net = i_net.eval()

    # P 模型：LD/HTL/HTS 三种结构
    if model_structure == ModelStructure.LD:
        p_net = DMC()
    else:
        p_net = DMC(model_structure=model_structure)
    begin_epoch, opt_status = load_existing_weights(
        args.save_dir, p_net, rank, args.pretrain_path)

    device = get_current_device(rank)
    if rank >= 0:
        p_net = p_net.cuda(rank)
        i_net = i_net.cuda(rank)
    else:
        p_net = p_net.to(device)
        i_net = i_net.to(device)
    p_net = wrap_ddp(p_net, rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(p_net.parameters(), lr=1e-4)
    if opt_status is not None:
        optimizer.load_state_dict(opt_status)

    dataloader = get_dataloader(train_dataset, rank, world_size, args.batch_size,
                                args.num_workers)
    for epoch in range(begin_epoch, args.epochs):
        if rank >= 0:
            dataloader.sampler.set_epoch(epoch)
        train_one_epoch(p_net, i_net, args, dataloader, optimizer, epoch, rank,
                        frame_delay, model_structure)

        if rank <= 0:
            save_status(args.save_dir, p_net, optimizer, epoch)

    if rank <= 0:
        save_ckpt(args.save_dir, p_net)
    cleanup_train(rank)


def main(argv):
    # 入口：解析参数后经 start_train 启动（多 GPU 时 mp.spawn 分布式）
    args = parse_args(argv)
    start_train(train, args)


if __name__ == '__main__':
    main(sys.argv[1:])
