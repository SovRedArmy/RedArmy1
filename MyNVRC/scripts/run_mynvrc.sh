#!/bin/bash
# MyNVRC 实验矩阵（GPU 空闲时执行；用法 bash scripts/run_mynvrc.sh N0 N1 N2 N3 [GPU]）
#   N0 = NVRC 原版基线（原生三层熵模型，无迁移模块）——同机对照
#   N1 = N0 + 模块A（动态加权采样+保守损失权重+余弦窗口）
#   N2 = N0 但熵模型换最简 QEC（模块B 对照：简单逐张量 vs 原生上下文 ar/d-axis）
#   N3 = 模块C（宽 base 整段训练 → clip 级 II-init + 残差参考，intra_period 60）
# 数据：ReadySetGo（与 MyHiNeRV B/C 系列同视频同预算可比）
set -e
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null
export TMPDIR=/dev/shm/myhinerbv_mp
export TORCH_EXTENSIONS_DIR=/remote-home/gdzj21/tmp/torch_ext
export PYTHONDONTWRITEBYTECODE=1
cd /remote-home/gdzj21/VideoCompression/MyNVRC
GPU=${GPU:-0}
DATASET_DIR=/remote-home/gdzj21/dataset/UVG/png
VIDEO=ReadySetGo
OUT=output/mynvrc
DATA_CFG=scripts/configs/data/video/png/uvg.yaml
TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
mkdir -p $OUT/logs

COMMON_ARGS="--train-dataset-dir ${DATASET_DIR} --train-dataset ${VIDEO} --eval-dataset-dir ${DATASET_DIR} --eval-dataset ${VIDEO} --output ${OUT} \
 --train-data-config ${DATA_CFG} --eval-data-config ${DATA_CFG} \
 --train-task-config ${TASK_CFG} --eval-task-config ${TASK_CFG} \
 --train-patch-size 1 120 120 --eval-patch-size 1 -1 -1 \
 --lamb 32 --grad-accum 1 --rate-steps 8 --train-batch-size 144 --eval-batch-size 1 \
 --start-frame -1 --num-frames -1 --workers 2 --train-enable-log false --eval-enable-log false"

run_s1s2 () {  # $1=exp  $2=s1cfg  $3=compress_s1  $4=compress_s2  额外参数透传
  local EXP=$1 S1=$2 C1=$3 C2=$4; shift 4
  CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29590} \
    main_nvrc.py --exp-config ${S1} --compress-model-config ${C1} --model-config ${MODEL_CFG} \
    ${COMMON_ARGS} --exp-name ${EXP}-s1 "$@" 2>&1 | tee -a $OUT/logs/${EXP}-s1.log | tail -3
  CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=${PORT:-29590} \
    main_nvrc.py --exp-config scripts/configs/nvrc/overfit/s2-30e.yaml \
    --compress-model-config ${C2} --model-config ${MODEL_CFG} \
    ${COMMON_ARGS} --resume ${OUT}/${EXP}-s1 --resume-model-only true \
    --exp-name ${EXP}-s2 "$@" 2>&1 | tee -a $OUT/logs/${EXP}-s2.log | tail -3
}

for V in "$@"; do
case $V in
N0) MODEL_CFG=scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml
    run_s1s2 mynvrc-n0 scripts/configs/nvrc/overfit/s1-360e.yaml \
      scripts/configs/nvrc/compress_models/nvrc_s1.yaml scripts/configs/nvrc/compress_models/nvrc_s2.yaml ;;
N1) MODEL_CFG=scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml
    run_s1s2 mynvrc-n1 scripts/configs/nvrc/overfit/mynvrc_dyn_s1.yaml \
      scripts/configs/nvrc/compress_models/nvrc_s1.yaml scripts/configs/nvrc/compress_models/nvrc_s2.yaml ;;
N2) MODEL_CFG=scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml
    run_s1s2 mynvrc-n2 scripts/configs/nvrc/overfit/s1-360e.yaml \
      scripts/configs/nvrc/compress_models/nvrc_qec_s1.yaml scripts/configs/nvrc/compress_models/nvrc_qec_s2.yaml ;;
N3) # 模块 C 两段：宽 base 整段训练（一次）→ clip 级 II-init（base_model_path 注入）
    MODEL_CFG=scripts/configs/nvrc/models/uvg_hinerv-v2-base_1920x1080.yaml
    if [ ! -d "${OUT}/mynvrc-n3-base-s2/0000/decoded" ]; then
      run_s1s2 mynvrc-n3-base scripts/configs/nvrc/overfit/mynvrc_base.yaml \
        scripts/configs/nvrc/compress_models/nvrc_s1.yaml scripts/configs/nvrc/compress_models/nvrc_s2.yaml
    fi
    MODEL_CFG=scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml
    BASE_PATH=${OUT}/mynvrc-n3-base-s2/0000/decoded
    # base_model_path 是 ExperimentConfig 字段（非 CLI 参数）：运行时烤进专属 exp-config
    CLIPS_CFG=/tmp/mynvrc_clips_$$.yaml
    cp scripts/configs/nvrc/overfit/mynvrc_clips.yaml ${CLIPS_CFG}
    echo "base_model_path: ${BASE_PATH}" >> ${CLIPS_CFG}
    run_s1s2 mynvrc-n3 ${CLIPS_CFG} \
      scripts/configs/nvrc/compress_models/nvrc_s1.yaml scripts/configs/nvrc/compress_models/nvrc_s2.yaml ;;
esac
done
echo "==== MyNVRC matrix done ===="
