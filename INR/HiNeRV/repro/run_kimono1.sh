#!/bin/bash
# HiNeRV 论文复现：HEVC ClassB / Kimono1 (240帧 1080p)
# 与 BQTerrace 相同的 HiNeRV-S 完整流水线; base-grid T 按帧数等比缩放 150->60 (240/600)
# v3: $1 可选传入 checkpoint 目录用于断点续训; --cached image 降低RSS
export PATH=/root/anaconda3/envs/hinerv/bin:$PATH
cd /remote-home/gdzj21/VideoCompression/INR/HiNeRV

RESUME_ARGS=""
[ -n "$1" ] && RESUME_ARGS="--resume $1"

CUDA_VISIBLE_DEVICES=6 accelerate launch \
  --num_processes=1 --mixed_precision=fp16 --main_process_port=29562 \
  hinerv_main.py \
  $RESUME_ARGS \
  --dataset /remote-home/share/HEVC_ClassB --dataset-name Kimono1_1920x1080_24 \
  --output /remote-home/gdzj21/VideoCompression/INR/HiNeRV/repro/Kimono1 \
  --loss 0.7 l1 0.3 ms-ssim_5x5 \
  --train-metric psnr --eval-metric psnr ms-ssim \
  --crop-size -1 -1 --input-size 1080 1920 --patch-size 1 120 120 --cached image \
  --epochs 300 --eval-epochs 30 --log-epochs -1 --warmup-epochs 30 \
  --prune-epochs 60 --prune-warmup-epochs 0 --prune-ratio 0.15 --prune-weight 0.5 \
  --quant-epochs 30 --quant-warmup-epochs 0 --quant-lr-scale 0.1 --quant-level 8 7 6 --quant-noise 0.9 \
  --opt adam --lr 2e-3 --warmup-lr 2e-5 --min-lr 2e-5 --max-norm 1.0 --auto-lr-scaling true \
  --model HiNeRV \
  --block-type convnext --channels 280 --channels-reduce 2.0 --depths 3 3 3 1 \
  --exps 4. 4. 4. 1. --stem-kernels 3 --kernels 3 3 3 3 \
  --scales-t 1 1 1 1 --scales-hw 5 3 2 2 --stem-paddings -1 -1 -1 --paddings -1 -1 -1 \
  --base-size -1 -1 -1 --base-grid-size 60 -1 -1 2 --base-grid-level 2 --base-grid-level-scale 2. 1. 1. 0.5 \
  --enc-type normalized+temp_local_grid --enc-grid-size -1 4 --enc-grid-level 3 --enc-grid-level-scale 2. 0.5 --enc-grid-depth-scale 1. 0.5 \
  --upsample-type trilinear --upsample-config matmul-th-w \
  --batch-size 144 --eval-batch-size 1 --grad-accum 1 --log-eval false --seed 0
