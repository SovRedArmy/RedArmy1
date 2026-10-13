#!/bin/bash
# N1 消融矩阵：N1a(仅采样)×3 + N1b(仅损失)×3 + 已有 N1(完整)
set -e
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null
export TMPDIR=/dev/shm/myhinerbv_mp TORCH_EXTENSIONS_DIR=/remote-home/gdzj21/tmp/torch_ext PYTHONDONTWRITEBYTECODE=1
cd /remote-home/gdzj21/VideoCompression/MyNVRC
GPU=${GPU:-3}
OUT=output/mynvrc
DATASET_DIR=/remote-home/gdzj21/dataset/UVG/png
DATA_CFG=scripts/configs/data/video/png/uvg.yaml
TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
mkdir -p $OUT/logs

COMMON="--train-dataset-dir ${DATASET_DIR} --train-dataset ReadySetGo \
 --eval-dataset-dir ${DATASET_DIR} --eval-dataset ReadySetGo --output ${OUT} \
 --train-data-config ${DATA_CFG} --eval-data-config ${DATA_CFG} \
 --train-task-config ${TASK_CFG} --eval-task-config ${TASK_CFG} \
 --train-patch-size 1 120 120 --eval-patch-size 1 -1 -1 \
 --lamb 32 --grad-accum 1 --rate-steps 8 --train-batch-size 144 --eval-batch-size 1 \
 --start-frame -1 --num-frames -1 --workers 2 --train-enable-log false --eval-enable-log false"

MODEL_CFG=scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml

for V in "$@"; do
  S1CFG=scripts/configs/nvrc/overfit/${V}_s1.yaml
  [ -f "$S1CFG" ] || { echo "no $S1CFG"; continue; }
  echo "==== $(date '+%F %T') GPU${GPU} $V ===="
  CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29600} \
    main_nvrc.py --exp-config ${S1CFG} --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s1.yaml \
    --model-config ${MODEL_CFG} ${COMMON} --exp-name ${V}-s1 2>&1 | tee -a $OUT/logs/${V}-s1.log | tail -3
  CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29600} \
    main_nvrc.py --exp-config scripts/configs/nvrc/overfit/s2-30e.yaml \
    --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s2.yaml \
    --model-config ${MODEL_CFG} ${COMMON} --resume ${OUT}/${V}-s1 --resume-model-only true \
    --exp-name ${V}-s2 2>&1 | tee -a $OUT/logs/${V}-s2.log | tail -3
done
echo "==== N1 ablation done ===="
