#!/bin/bash
# E1: CANeRV 增强（《DSA/DFA/HSA》应用于 NVRC 两级架构）
set -e
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null
export TMPDIR=/dev/shm/myhinerbv_mp TORCH_EXTENSIONS_DIR=/remote-home/gdzj21/tmp/torch_ext PYTHONDONTWRITEBYTECODE=1
cd /remote-home/gdzj21/VideoCompression/MyNVRC
GPU=${GPU:-3}
OUT=output/mynvrc
mkdir -p $OUT/logs

# 检查 base 已完成
BASE_PATH=${OUT}/mynvrc-n3-base-s1/outputs/0000/decoded/None
if [ ! -d "${BASE_PATH}" ]; then
    echo "E1: base 未完成 (${BASE_PATH})"; exit 1
fi
echo "E1: base 已就绪"

# E1-S1: clip 段（CANeRV 初始化）
CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29670} \
  main_nvrc.py \
  --exp-config scripts/configs/nvrc/overfit/e1_canerv_s1.yaml \
  --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s1.yaml \
  --model-config scripts/configs/nvrc/models/uvg_hinerv-v2-clip160_1920x1080.yaml \
  --train-dataset-dir /remote-home/gdzj21/dataset/UVG/png --train-dataset ReadySetGo \
  --eval-dataset-dir /remote-home/gdzj21/dataset/UVG/png --eval-dataset ReadySetGo \
  --output ${OUT} \
  --train-data-config scripts/configs/data/video/png/uvg.yaml --eval-data-config scripts/configs/data/video/png/uvg.yaml \
  --train-task-config scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml --eval-task-config scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml \
  --train-patch-size 1 120 120 --eval-patch-size 1 -1 -1 \
  --lamb 32 --grad-accum 1 --rate-steps 8 --train-batch-size 144 --eval-batch-size 1 \
  --start-frame -1 --num-frames -1 --workers 0 --train-enable-log false --eval-enable-log false \
  --exp-name e1-canerv-s1 2>&1 | tee -a $OUT/logs/e1-canerv-s1.log | tail -3

# E1-S2: 硬量化精调
CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29670} \
  main_nvrc.py \
  --exp-config scripts/configs/nvrc/overfit/s2-30e.yaml \
  --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s2.yaml \
  --model-config scripts/configs/nvrc/models/uvg_hinerv-v2-clip160_1920x1080.yaml \
  --train-dataset-dir /remote-home/gdzj21/dataset/UVG/png --train-dataset ReadySetGo \
  --eval-dataset-dir /remote-home/gdzj21/dataset/UVG/png --eval-dataset ReadySetGo \
  --output ${OUT} \
  --train-data-config scripts/configs/data/video/png/uvg.yaml --eval-data-config scripts/configs/data/video/png/uvg.yaml \
  --train-task-config scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml --eval-task-config scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml \
  --train-patch-size 1 120 120 --eval-patch-size 1 -1 -1 \
  --lamb 32 --grad-accum 1 --rate-steps 8 --train-batch-size 144 --eval-batch-size 1 \
  --start-frame -1 --num-frames -1 --workers 0 --train-enable-log false --eval-enable-log false \
  --resume ${OUT}/e1-canerv-s1 --resume-model-only true \
  --exp-name e1-canerv-s2 2>&1 | tee -a $OUT/logs/e1-canerv-s2.log | tail -3

echo "==== E1 done ===="
