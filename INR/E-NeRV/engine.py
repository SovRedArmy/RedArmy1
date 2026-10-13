# -*- coding: utf-8 -*-
"""训练 / 评估引擎。

包含两个核心函数：
    train_one_epoch()  单轮训练：遍历 DataLoader，前向 -> 多尺度损失 -> 反向传播 -> 学习率调度 -> 指标统计
    evaluate()         验证：仅前向并统计 PSNR / MS-SSIM（神经表示是逐视频拟合，验证集衡量重建质量）
"""
import math
import os
import sys
import torch
import utils.misc as utils
import torch.nn.functional as F
from datetime import datetime


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    cfg,
    args,
    datasize,
    start_time,
    writer=None,
):
    """训练一个 epoch。

    每个 iteration 的流程：
        1) utils.to_cuda()            把一个 batch 的数据搬到 GPU
        2) model(data)                跳转到模型的 forward()（E_NeRV_Generator.forward / NeRV_Generator.forward）
        3) F.adaptive_avg_pool2d      把 GT 帧下采样到各输出尺度，得到多尺度监督目标 target_list
        4) utils.loss_compute()       逐尺度计算损失（cfg['loss']，默认 Fusion6 = 0.7*L1 + 0.3*(1-SSIM)）
        5) utils.adjust_lr()          按迭代数做 warmup-cosine 学习率调度
        6) losses.backward() + optimizer.step()   反向传播并更新参数
        7) utils.psnr_fn / msssim_fn  统计重建指标
    """
    model.train()  # 切换到训练模式（启用 dropout 等；本模型主要影响 self.training 标志）
    epoch_start_time = datetime.now()
    loss_type = cfg["loss"]  # 损失类型，来自 yaml 配置（默认 Fusion6）

    psnr_list = []    # 收集每个 batch 的 PSNR，形状 (batchsize, num_stage)
    msssim_list = []  # 收集每个 batch 的 MS-SSIM，形状 (batchsize, num_stage)

    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)  # 数据字典 {img_id, img_gt} 搬至 GPU
        # forward pass
        output_list = model(data)  # output is a list for the case that has multiscale（多尺度输出列表，最高分辨率在最末）
        additional_loss_item = {}
        if isinstance(output_list, dict):  # 若模型额外返回了辅助损失（如梯度损失），先取出
            for k, v in output_list.items():
                if "loss" in k:
                    additional_loss_item[k] = v
            output_list = output_list["output_list"]
        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list  # 将 GT 池化到与每个输出尺度相同的大小
        ]
        loss_list = utils.loss_compute(output_list, target_list, loss_type)  # 逐尺度计算损失 -> utils.loss_fn
        losses = sum(loss_list)  # 多尺度损失求和（仅 sin_res=true 时实际上只有最后一个尺度有输出）
        if len(additional_loss_item.values()) > 0:
            losses = losses + sum(additional_loss_item.values())  # 叠加辅助损失（默认配置关闭）

        lr = utils.adjust_lr(optimizer, epoch, cfg["epoch"], i, datasize, cfg)  # 按全局迭代步数调整学习率
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))      # (batchsize, num_stage)
        msssim_list.append(utils.msssim_fn(output_list, target_list))  # (batchsize, num_stage)

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:  # 每隔 print_freq 步（或最后一步）打印一次
            train_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            train_psnr = torch.mean(train_psnr, dim=0)  # (num_stage)  对已遍历 batch 取平均
            train_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            train_msssim = torch.mean(train_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if not hasattr(args, "rank"):  # 独立调用 engine（未经过 main.py 初始化）时的打印分支
                print_str = "[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),   # 张量转成便于打印的字符串
                    utils.RoundTensor(train_msssim, 4, False),
                )
                for k, v in additional_loss_item.items():
                    print_str += f", {k}: {v.item():.6g}"
                print(print_str, flush=True)

            elif args.rank in [0, None]:  # 正常路径：仅主进程打印
                print_str = "[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    args.rank,
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),
                    utils.RoundTensor(train_msssim, 4, False),
                )
                print(print_str, flush=True)

    train_stats = {  # 本 epoch 的最终统计（取最后一个 batch 打印时的累计均值）
        "train_psnr": train_psnr,
        "train_msssim": train_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        train_stats = utils.reduce_dict(train_stats)  # 分布式下跨进程求平均

    # ADD train_PSNR TO TENSORBOARD
    if not hasattr(args, "rank"):  # 独立调用时的 tensorboard 记录分支
        h, w = output_list[-1].shape[-2:]  # 最高分辨率输出的尺寸
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        for k, v in additional_loss_item.items():
            writer.add_scalar(f"Train/{k}", v.item(), epoch + 1)
        for (k, m) in model.named_modules():  # 记录模块的 Lip_c / abssum_max 统计（本仓库模型未定义这些属性，属遗留代码）
            if isinstance(m, torch.nn.Module) and hasattr(m, "Lip_c"):
                writer.add_scalar(f"Stat/{k}_c", m.Lip_c[0].item(), epoch + 1)
                writer.add_scalar(f"Stat/{k}_w", m.abssum_max, epoch + 1)

    elif args.rank in [0, None] and writer is not None:  # 正常路径：仅主进程写 tensorboard
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
    epoch_end_time = datetime.now()
    print(  # 打印本 epoch 耗时与自训练开始以来的平均耗时
        "Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
            (epoch_end_time - epoch_start_time).total_seconds(),
            (epoch_end_time - start_time).total_seconds() / (epoch + 1),
        )
    )

    return train_stats  # 返回给 main() 用于更新最优指标


@torch.no_grad()  # 评估阶段不构建计算图
def evaluate(model, dataloader, device, cfg, args, save_image=False):
    """在验证集上评估重建质量（PSNR / MS-SSIM）。

    与训练的区别：不计算损失、不更新参数，只做前向与指标统计；
    神经视频表示是逐视频过拟合，因此“验证”衡量的是对已见帧的重建精度。
    """
    val_start_time = datetime.now()
    model.eval()  # 切换到评估模式

    psnr_list = []
    msssim_list = []

    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)  # 数据搬至 GPU
        # forward pass
        output_list = model(data)  # output is a list for the case that has multiscale
        if isinstance(output_list, dict):
            output_list = output_list["output_list"]  # ignore the loss in eval（评估时忽略辅助损失）
        torch.cuda.synchronize()  # 同步 CUDA 流，保证后续计时准确
        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list  # GT 下采样到各输出尺度
        ]

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))      # (batchsize, num_stage)
        msssim_list.append(utils.msssim_fn(output_list, target_list))  # (batchsize, num_stage)

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            val_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            val_psnr = torch.mean(val_psnr, dim=0)  # (num_stage)
            val_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            val_msssim = torch.mean(val_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if not hasattr(args, "rank"):  # 独立调用 engine 时的打印分支
                print_str = "[{}], Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                print(print_str, flush=True)

            elif args.rank in [0, None]:  # 正常路径：仅主进程打印
                print_str = "[{}] Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    args.rank,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                print(print_str, flush=True)

    val_stats = {  # 本轮验证的最终统计
        "val_psnr": val_psnr,
        "val_msssim": val_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        val_stats = utils.reduce_dict(val_stats)  # 分布式下跨进程求平均
    val_end_time = datetime.now()
    print(
        "Time on evaluate: \t{:.2f}".format(
            (val_end_time - val_start_time).total_seconds()
        )
    )

    return val_stats  # 返回给 main() 用于更新最优指标
