#!/bin/bash
# HiNeRV 论文严格复现：UVG ReadySetGo (600帧 1080p)
# 目的：钉死 MyHiNeRV B0/B0r 与论文 31.53dB 的差异来源
# 配置 = cfgs/train/hinerv_1920x1080.txt + cfgs/models/uvg-hinerv-s_1920x1080.txt（逐参数）
# 流程 = 主训练300e(warmup30) -> 剪枝15%+微调60e -> Quant-Noise QAT 30e -> Q8/7/6 码流
# $1 = 可选 GPU 号（默认 4）；$2 = 可选 checkpoint 续训
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
GPU=${1:-4}
RESUME_ARGS=""
[ -n "$2" ] && RESUME_ARGS="--resume $2"
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null
export TMPDIR=/dev/shm/myhinerbv_mp
export TORCH_EXTENSIONS_DIR=/remote-home/gdzj21/tmp/torch_ext
export PYTHONDONTWRITEBYTECODE=1
cd /remote-home/gdzj21/VideoCompression/INR/HiNeRV

CUDA_VISIBLE_DEVICES=$GPU accelerate launch \
  --num_processes=1 --mixed_precision=fp16 --main_process_port=29577 \
  hinerv_main.py \
  $RESUME_ARGS \
  --dataset /remote-home/gdzj21/dataset/UVG/png --dataset-name ReadySetGo \
  --output /remote-home/gdzj21/VideoCompression/INR/HiNeRV/repro/ReadySetGo \
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
  --base-size -1 -1 -1 --base-grid-size 150 -1 -1 2 --base-grid-level 2 --base-grid-level-scale 2. 1. 1. 0.5 \
  --enc-type normalized+temp_local_grid --enc-grid-size -1 4 --enc-grid-level 3 --enc-grid-level-scale 2. 0.5 --enc-grid-depth-scale 1. 0.5 \
  --upsample-type trilinear --upsample-config matmul-th-w \
  --batch-size 144 --eval-batch-size 1 --grad-accum 1 --log-eval false --seed 0 \
  2>&1 | tee -a /remote-home/gdzj21/VideoCompression/INR/HiNeRV/repro/ReadySetGo_run.log
