#!/bin/bash
# NVRC 复现脚本（JVET Class B, /remote-home/share/HEVC_ClassB）
# 用法: bash overfitting_jvetb_nvrc.sh <GPU_ID> <VIDEO_DIR_NAME> <LAMB> <SCALE(s|m|l)> <LR_S1> <LR_S2> <GRAD_ACCUM> <BATCH_SIZE> [EPOCHS_S1] [EPOCHS_S2] [EXP_TAG]
# 缺省 EPOCHS_S1/EPOCHS_S2 为论文官方设置 360/30；可传入缩减预算（退火曲线按比例保持）
if [ "$#" -lt 8 ]; then
    echo "Incorrect number of parameters"
    exit 1
fi
GPU_ID=$1
VID=$2
LAMB=$3
SCALE=$4
LR_S1=$5
LR_S2=$6
GRAD_ACCUM=$7
BATCH_SIZE=$8
EPOCHS_S1=${9:-360}
EPOCHS_S2=${10:-30}
EXP_TAG=${11:-paper}

shift $#

echo "GPU_ID: ${GPU_ID}"
echo "VID: ${VID}"
echo "LAMB: ${LAMB}"
echo "SCALE: ${SCALE}"
echo "LR_S1: ${LR_S1}  LR_S2: ${LR_S2}"
echo "GRAD_ACCUM: ${GRAD_ACCUM}  BATCH_SIZE: ${BATCH_SIZE}"
echo "EPOCHS_S1: ${EPOCHS_S1}  EPOCHS_S2: ${EPOCHS_S2}"

WORK_DIR=/remote-home/gdzj21
ROOT=${WORK_DIR}/VideoCompression/INR
REPO=${ROOT}/NVRC
# 根盘很小：把所有编译/临时缓存重定向到大磁盘，避免磁盘满导致崩溃
export TMPDIR=${WORK_DIR}/tmp
export TORCHINDUCTOR_CACHE_DIR=${WORK_DIR}/tmp/inductor
export TRITON_CACHE_DIR=${WORK_DIR}/tmp/triton
export XDG_CACHE_HOME=${WORK_DIR}/tmp/xdg_cache
mkdir -p ${TMPDIR} ${TORCHINDUCTOR_CACHE_DIR} ${TRITON_CACHE_DIR} ${XDG_CACHE_HOME}
TRAIN_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
EVAL_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim.yaml
COMPRESS_MODEL_CFG_S1=scripts/configs/nvrc/compress_models/nvrc_s1.yaml
COMPRESS_MODEL_CFG_S2=scripts/configs/nvrc/compress_models/nvrc_s2.yaml
MODEL_CFG_S1=scripts/configs/nvrc/models/uvg_hinerv-v2-${SCALE}_1920x1080.yaml
MODEL_CFG_S2=${MODEL_CFG_S1}
DATASET_DIR=/remote-home/share/HEVC_ClassB
DATASET=${VID}
START_FRAME=-1
NUM_FRAMES=-1
INTRA_PERIOD=-1
FMT=png
T=-1
H=-1
W=-1
T_PATCH=1
H_PATCH=120
W_PATCH=120
NUM_PROC=1
WORKERS=2
GRAD_ACCUM=${GRAD_ACCUM}
TRAIN_BATCH_SIZE=${BATCH_SIZE}
EVAL_BATCH_SIZE=1
MODEL_NAME=nvrc_jvetb_hinerv-v2-${SCALE}_1920x1080_x${GRAD_ACCUM}x${TRAIN_BATCH_SIZE}

OUTPUT=${WORK_DIR}/Outputs/NVRC/${DATASET}
EXP_NAME_S1=${MODEL_NAME}_lamb-${LAMB}_lr-${LR_S1}_${EXP_TAG}_s1
EXP_NAME_S2=${MODEL_NAME}_lamb-${LAMB}_lr-${LR_S2}_${EXP_TAG}_s2

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Start running the script with experiment:"
echo "    Dataset: ${DATASET}"
echo "    Output: ${OUTPUT}"
echo "    Stage 1: ${EXP_NAME_S1}"
echo "    Stage 2: ${EXP_NAME_S2}"

# 为缩减 epoch 预算生成派生配置（warmup/eval 周期按官方比例缩放）；
# 全预算（360/30）时直接使用官方原版配置文件，与论文设置完全一致
make_exp_cfg () {
    if [ "$1" = "360" ] && [ "$2" = "1" ]; then
        echo scripts/configs/nvrc/overfit/s1-360e.yaml
        return
    fi
    if [ "$1" = "30" ] && [ "$2" = "2" ]; then
        echo scripts/configs/nvrc/overfit/s2-30e.yaml
        return
    fi
    local EPOCHS=$1
    local WARMUP=$(( EPOCHS / 12 + 1 ))
    local EVAL_E=$(( EPOCHS / 12 + 1 ))
    local CFG=${WORK_DIR}/tmp/nvrc_exp_${EXP_TAG}_s${2}.yaml
    cat > ${CFG} << EOF
epochs: ${EPOCHS}
eval_epochs: ${EVAL_E}
warmup_epochs: ${WARMUP}
rate_steps: 8
log_epochs: -2
opt: adam
lr: ${3}
warmup_lr: 1e-5
min_lr: ${4}
max_norm: 1.0
weight_decay: ${5}
weight_decay_scaling: true
auto_lr_scaling: true
EOF
    echo ${CFG}
}
EXP_CFG_S1=$(make_exp_cfg ${EPOCHS_S1} 1 ${LR_S1} 1e-4 1e-6)
EXP_CFG_S2=$(make_exp_cfg ${EPOCHS_S2} 2 ${LR_S2} 1e-5 0.)
echo "EXP_CFG_S1: ${EXP_CFG_S1}"
echo "EXP_CFG_S2: ${EXP_CFG_S2}"

# S1 training
if [ ! -d ${OUTPUT}/${EXP_NAME_S1} ]; then
    source /remote-home/gdzj21/miniconda3/etc/profile.d/conda.sh
    conda activate NVRC
    cd ${REPO} && \
    accelerate launch --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
                      --gpu_ids=${GPU_ID} --num_processes=${NUM_PROC} --mixed_precision=fp16 --dynamo_backend=inductor \
    main_nvrc.py --exp-config ${EXP_CFG_S1} \
                 --output ${OUTPUT} --exp-name ${EXP_NAME_S1} \
                 --train-task-config ${TRAIN_TASK_CFG} --eval-task-config ${EVAL_TASK_CFG} \
                 --compress-model-config ${COMPRESS_MODEL_CFG_S1} --model-config ${MODEL_CFG_S1} \
                 --train-dataset-dir ${DATASET_DIR} --train-dataset ${DATASET} --train-fmt ${FMT} \
                 --lamb ${LAMB} \
                 --start-frame ${START_FRAME} --num-frames ${NUM_FRAMES} --intra-period ${INTRA_PERIOD} \
                 --train-video-size ${T} ${H} ${W} --eval-video-size ${T} ${H} ${W} \
                 --train-patch-size ${T_PATCH} ${H_PATCH} ${W_PATCH} --eval-patch-size 1 -1 -1 \
                 --grad-accum ${GRAD_ACCUM} --rate-steps 8 \
                 --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} \
                 --train-enable-log false --eval-enable-log false --log-epochs -2 \
                 --opt adam --sched cosine \
                 --lr ${LR_S1} --warmup-lr 1e-5 --min-lr 1e-4 --auto-lr-scaling true --max-norm 1.0 \
                 --workers ${WORKERS} --prefetch-factor 2
else
    echo "Experiment ${EXP_NAME_S1} already exists, skipping S1 training."
fi

# S1 断点自动续跑：目录存在但未完成（如被 OOM 杀掉）则从 checkpoint 恢复，最多重试 40 次
RETRY=0
while [[ ! -f "${OUTPUT}/${EXP_NAME_S1}/results/all.txt" && ${RETRY} -lt 40 ]]; do
    source /remote-home/gdzj21/miniconda3/etc/profile.d/conda.sh
    conda activate NVRC
    cd ${REPO} && \
    accelerate launch --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
                      --gpu_ids=${GPU_ID} --num_processes=${NUM_PROC} --mixed_precision=fp16 --dynamo_backend=inductor \
    main_nvrc.py --exp-config ${EXP_CFG_S1} \
                 --output ${OUTPUT} --exp-name ${EXP_NAME_S1} \
                 --train-task-config ${TRAIN_TASK_CFG} --eval-task-config ${EVAL_TASK_CFG} \
                 --compress-model-config ${COMPRESS_MODEL_CFG_S1} --model-config ${MODEL_CFG_S1} \
                 --train-dataset-dir ${DATASET_DIR} --train-dataset ${DATASET} --train-fmt ${FMT} \
                 --lamb ${LAMB} \
                 --start-frame ${START_FRAME} --num-frames ${NUM_FRAMES} --intra-period ${INTRA_PERIOD} \
                 --train-video-size ${T} ${H} ${W} --eval-video-size ${T} ${H} ${W} \
                 --train-patch-size ${T_PATCH} ${H_PATCH} ${W_PATCH} --eval-patch-size 1 -1 -1 \
                 --grad-accum ${GRAD_ACCUM} --rate-steps 8 \
                 --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} \
                 --train-enable-log false --eval-enable-log false --log-epochs -2 \
                 --opt adam --sched cosine \
                 --lr ${LR_S1} --warmup-lr 1e-5 --min-lr 1e-4 --auto-lr-scaling true --max-norm 1.0 \
                 --workers ${WORKERS} --prefetch-factor 2 \
                 --resume ${OUTPUT}/${EXP_NAME_S1}
    RETRY=$((RETRY+1))
    echo "S1 retry round ${RETRY} finished; results exist: $([[ -f ${OUTPUT}/${EXP_NAME_S1}/results/all.txt ]] && echo yes || echo no)"
    sleep 30
done

if [[ ! -f "${OUTPUT}/${EXP_NAME_S1}/results/all.txt" ]]; then
    echo "S1 training not completed, please check the logs."
    exit 1
fi

# S2 training
if [ ! -d ${OUTPUT}/${EXP_NAME_S2} ]; then
    source /remote-home/gdzj21/miniconda3/etc/profile.d/conda.sh
    conda activate NVRC
    cd ${REPO} && \
    accelerate launch --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
                      --gpu_ids=${GPU_ID} --num_processes=${NUM_PROC} --mixed_precision=fp16 --dynamo_backend=inductor \
    main_nvrc.py --exp-config ${EXP_CFG_S2} \
                 --output ${OUTPUT} --exp-name ${EXP_NAME_S2} \
                 --train-task-config ${TRAIN_TASK_CFG} --eval-task-config ${EVAL_TASK_CFG} \
                 --compress-model-config ${COMPRESS_MODEL_CFG_S2} --model-config ${MODEL_CFG_S2} \
                 --train-dataset-dir ${DATASET_DIR} --train-dataset ${DATASET} --train-fmt ${FMT} \
                 --lamb ${LAMB} \
                 --start-frame ${START_FRAME} --num-frames ${NUM_FRAMES} --intra-period ${INTRA_PERIOD} \
                 --train-video-size ${T} ${H} ${W} --eval-video-size ${T} ${H} ${W} \
                 --train-patch-size ${T_PATCH} ${H_PATCH} ${W_PATCH} --eval-patch-size 1 -1 -1 \
                 --grad-accum ${GRAD_ACCUM} --rate-steps 8 \
                 --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} \
                 --train-enable-log false --eval-enable-log true --log-epochs -2 \
                 --opt adam --sched cosine \
                 --lr ${LR_S2} --warmup-lr 1e-5 --min-lr 1e-5 --auto-lr-scaling true --max-norm 1.0 \
                 --workers 4 --prefetch-factor 4 \
                 --resume ${OUTPUT}/${EXP_NAME_S1} --resume-model-only true
else
    echo "Experiment ${EXP_NAME_S2} already exists, skipping S2 training."
fi

if [ ! -f "${OUTPUT}/${EXP_NAME_S2}/results/all.txt" ]; then
    echo "S2 training not completed, please check the logs."
    exit 1
fi

# S2 断点自动续跑：未完成则恢复——已有 S2 checkpoint 时恢复完整训练状态；
# 尚无 checkpoint 时从 S1 权重重新热启动。最多重试 40 次
RETRY=0
while [[ ! -f "${OUTPUT}/${EXP_NAME_S2}/results/all.txt" && ${RETRY} -lt 40 ]]; do
    if [ -d "${OUTPUT}/${EXP_NAME_S2}/checkpoints/0000" ]; then
        RESUME_ARGS="--resume ${OUTPUT}/${EXP_NAME_S2}"
    else
        RESUME_ARGS="--resume ${OUTPUT}/${EXP_NAME_S1} --resume-model-only true"
    fi
    source /remote-home/gdzj21/miniconda3/etc/profile.d/conda.sh
    conda activate NVRC
    cd ${REPO} && \
    accelerate launch --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
                      --gpu_ids=${GPU_ID} --num_processes=${NUM_PROC} --mixed_precision=fp16 --dynamo_backend=inductor \
    main_nvrc.py --exp-config ${EXP_CFG_S2} \
                 --output ${OUTPUT} --exp-name ${EXP_NAME_S2} \
                 --train-task-config ${TRAIN_TASK_CFG} --eval-task-config ${EVAL_TASK_CFG} \
                 --compress-model-config ${COMPRESS_MODEL_CFG_S2} --model-config ${MODEL_CFG_S2} \
                 --train-dataset-dir ${DATASET_DIR} --train-dataset ${DATASET} --train-fmt ${FMT} \
                 --lamb ${LAMB} \
                 --start-frame ${START_FRAME} --num-frames ${NUM_FRAMES} --intra-period ${INTRA_PERIOD} \
                 --train-video-size ${T} ${H} ${W} --eval-video-size ${T} ${H} ${W} \
                 --train-patch-size ${T_PATCH} ${H_PATCH} ${W_PATCH} --eval-patch-size 1 -1 -1 \
                 --grad-accum ${GRAD_ACCUM} --rate-steps 8 \
                 --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} \
                 --train-enable-log false --eval-enable-log true --log-epochs -2 \
                 --opt adam --sched cosine \
                 --lr ${LR_S2} --warmup-lr 1e-5 --min-lr 1e-5 --auto-lr-scaling true --max-norm 1.0 \
                 --workers ${WORKERS} --prefetch-factor 2 \
                 ${RESUME_ARGS}
    RETRY=$((RETRY+1))
    echo "S2 retry round ${RETRY} finished; results exist: $([[ -f ${OUTPUT}/${EXP_NAME_S2}/results/all.txt ]] && echo yes || echo no)"
    sleep 30
done

if [ ! -f "${OUTPUT}/${EXP_NAME_S2}/results/all.txt" ]; then
    echo "S2 training not completed after retries, please check the logs."
    exit 1
fi
echo "=== JVET-B NVRC RUN COMPLETED: ${EXP_NAME_S2} ==="
cat ${OUTPUT}/${EXP_NAME_S2}/results/all.txt
