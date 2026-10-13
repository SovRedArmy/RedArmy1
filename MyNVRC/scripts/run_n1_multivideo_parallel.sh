#!/bin/bash
# N1 消融多视频验证（并行版）：4 组对照同时跑在 4 张 GPU 上
# 用法: bash run_n1_multivideo_parallel.sh <video> <gpu_n0> <gpu_n1a> <gpu_n1b> <gpu_n1>
set -e
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null
export TMPDIR=/dev/shm/myhinerbv_mp TORCH_EXTENSIONS_DIR=/remote-home/gdzj21/tmp/torch_ext PYTHONDONTWRITEBYTECODE=1
cd /remote-home/gdzj21/VideoCompression/MyNVRC
VIDEO=${1:-Beauty}
G_N0=${2:-0}; G_N1A=${3:-1}; G_N1B=${4:-2}; G_N1=${5:-3}
OUT=output/mynvrc
DATA=scripts/configs/data/video/png/uvg.yaml
TASK=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
MODEL=scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml
mkdir -p $OUT/logs

run_one () {  # $1=gpu  $2=exp_name  $3=s1_config  $4=port
  local G=$1 EXP=$2 S1=$3 P=$4
  export OMP_NUM_THREADS=8
  # S1
  CUDA_VISIBLE_DEVICES=$G accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=$P \
    main_nvrc.py --exp-config ${S1} \
    --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s1.yaml \
    --model-config ${MODEL} \
    --train-dataset-dir /remote-home/gdzj21/dataset/UVG/png --train-dataset ${VIDEO} \
    --eval-dataset-dir /remote-home/gdzj21/dataset/UVG/png --eval-dataset ${VIDEO} \
    --output ${OUT} \
    --train-data-config ${DATA} --eval-data-config ${DATA} \
    --train-task-config ${TASK} --eval-task-config ${TASK} \
    --train-patch-size 1 120 120 --eval-patch-size 1 -1 -1 \
    --lamb 32 --grad-accum 1 --rate-steps 8 --train-batch-size 144 --eval-batch-size 1 \
    --start-frame -1 --num-frames -1 --workers 2 --train-enable-log false --eval-enable-log false \
    --exp-name ${EXP}-s1 > /dev/null 2>&1 &
  local S1_PID=$!
  
  # S1 完成后自动接 S2
  (
    wait $S1_PID
    CUDA_VISIBLE_DEVICES=$G accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=$P \
      main_nvrc.py --exp-config scripts/configs/nvrc/overfit/s2-30e.yaml \
      --compress-model-config scripts/configs/nvrc/compress_models/nvrc_s2.yaml \
      --model-config ${MODEL} \
      --train-dataset-dir /remote-home/gdzj21/dataset/UVG/png --train-dataset ${VIDEO} \
      --eval-dataset-dir /remote-home/gdzj21/dataset/UVG/png --eval-dataset ${VIDEO} \
      --output ${OUT} \
      --train-data-config ${DATA} --eval-data-config ${DATA} \
      --train-task-config ${TASK} --eval-task-config ${TASK} \
      --train-patch-size 1 120 120 --eval-patch-size 1 -1 -1 \
      --lamb 32 --grad-accum 1 --rate-steps 8 --train-batch-size 144 --eval-batch-size 1 \
      --start-frame -1 --num-frames -1 --workers 2 --train-enable-log false --eval-enable-log false \
      --resume ${OUT}/${EXP}-s1 --resume-model-only true \
      --exp-name ${EXP}-s2 >> $OUT/logs/${EXP}-s2.log 2>&1
    echo "[$(date '+%F %T')] ${EXP} S2 done" >> $OUT/logs/multivideo_queue.log
  ) &
}

echo "[$(date '+%F %T')] ${VIDEO}: 启动 4 组并行 (GPU $G_N0/$G_N1A/$G_N1B/$G_N1)" >> $OUT/logs/multivideo_queue.log

# 4 组同时启动（各自独立 GPU）
run_one $G_N0  "mv-${VIDEO}-n0"  scripts/configs/nvrc/overfit/s1-360e.yaml      29700
run_one $G_N1A "mv-${VIDEO}-n1a" scripts/configs/nvrc/overfit/n1a_s1.yaml       29701
run_one $G_N1B "mv-${VIDEO}-n1b" scripts/configs/nvrc/overfit/n1b_s1.yaml       29702
run_one $G_N1  "mv-${VIDEO}-n1"  scripts/configs/nvrc/overfit/mynvrc_dyn_s1.yaml 29703

wait
echo "[$(date '+%F %T')] ${VIDEO}: 全部完成" >> $OUT/logs/multivideo_queue.log
