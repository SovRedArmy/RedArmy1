#!/bin/bash
# N1 消融多视频验证：在多部 UVG 视频上跑 n0/n1a/n1b/n1 四组对照
# 用法: bash run_n1_multivideo.sh <video_name> <gpu_id>
set -e
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null
export TMPDIR=/dev/shm/myhinerbv_mp TORCH_EXTENSIONS_DIR=/remote-home/gdzj21/tmp/torch_ext PYTHONDONTWRITEBYTECODE=1
cd /remote-home/gdzj21/VideoCompression/MyNVRC
VIDEO=${1:-Beauty}
GPU=${2:-2}
OUT=output/mynvrc
DATA=scripts/configs/data/video/png/uvg.yaml
TASK=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
MODEL=scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml
mkdir -p $OUT/logs

COMMON="--train-dataset-dir /remote-home/gdzj21/dataset/UVG/png --train-dataset ${VIDEO} \
 --eval-dataset-dir /remote-home/gdzj21/dataset/UVG/png --eval-dataset ${VIDEO} --output ${OUT} \
 --train-data-config ${DATA} --eval-data-config ${DATA} \
 --train-task-config ${TASK} --eval-task-config ${TASK} \
 --train-patch-size 1 120 120 --eval-patch-size 1 -1 -1 \
 --lamb 32 --grad-accum 1 --rate-steps 8 --train-batch-size 144 --eval-batch-size 1 \
 --start-frame -1 --num-frames -1 --workers 2 --train-enable-log false --eval-enable-log false"

run_variant () {  # $1=exp_name  $2=s1_config
  local EXP=$1 S1=$2
  echo "==== [$(date '+%F %T')] GPU${GPU} ${EXP} (${VIDEO}) ===="
  # S1
  CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29680} \
    main_nvrc.py --exp-config ${S1} \
    --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s1.yaml \
    --model-config ${MODEL} ${COMMON} --exp-name ${EXP}-s1 2>&1 | tee -a $OUT/logs/${EXP}-s1.log | tail -3
  # S2
  CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29680} \
    main_nvrc.py --exp-config scripts/configs/nvrc/overfit/s2-30e.yaml \
    --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s2.yaml \
    --model-config ${MODEL} ${COMMON} \
    --resume ${OUT}/${EXP}-s1 --resume-model-only true --exp-name ${EXP}-s2 2>&1 | tee -a $OUT/logs/${EXP}-s2.log | tail -3
}

# 四组对照（复用现有配置文件）
run_variant "mv-${VIDEO}-n0"   scripts/configs/nvrc/overfit/s1-360e.yaml
run_variant "mv-${VIDEO}-n1a"  scripts/configs/nvrc/overfit/n1a_s1.yaml
run_variant "mv-${VIDEO}-n1b"  scripts/configs/nvrc/overfit/n1b_s1.yaml
run_variant "mv-${VIDEO}-n1"   scripts/configs/nvrc/overfit/mynvrc_dyn_s1.yaml

echo "==== ${VIDEO} N1 ablation done ===="
