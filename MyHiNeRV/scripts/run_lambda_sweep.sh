#!/bin/bash
# λ（率失真权重）扫描：base 宽模型 λ 小、GOP 小模型 λ 大，多组对照找最佳 RD 曲线。
# 用法：GPU=4 SEQ=ReadySetGo QUICK=1 MODE=hyper bash scripts/run_lambda_sweep.sh
#   MODE: hyper | qec
#   扫描组合：GOP λ ∈ {0.5, 1.0, 2.0, 4.0} × base λ ∈ {0.25, 0.5, 1.0}
set -e
cd "$(dirname "$0")/.."

export TMPDIR=/remote-home/gdzj21/tmp/myhinerbv_tmp
mkdir -p "$TMPDIR"

GPU=${GPU:-0}
SEQ=${SEQ:-ReadySetGo}
QUICK=${QUICK:-1}
MODE=${MODE:-hyper}
DATASET=${DATASET:-/remote-home/gdzj21/dataset/UVG/png}
OUTROOT=${OUTROOT:-output/lambda_sweep}
PYTHON=${PYTHON:-python}

if [ "$QUICK" = "1" ]; then
  EP=120; PREP=24; QE=12; PE=120; WARM=15; EVE=15
else
  EP=300; PREP=60; QE=30; PE=300; WARM=30; EVE=30
fi

COMMON="--dataset ${DATASET} --dataset-name ${SEQ} --output ${OUTROOT} \
 --crop-size -1 -1 --input-size 1080 1920 --patch-size 1 120 120 --cached image \
 --batch-size 144 --eval-batch-size 1 --workers 0 --seed 0 --log-eval false \
 --epochs ${EP} --eval-epochs ${EVE} --log-epochs -1 --warmup-epochs ${WARM} \
 --prune-epochs ${PREP} --prune-warmup-epochs 0 --prune-ratio 0.15 --prune-weight 0.5 \
 --quant-epochs ${QE} --quant-warmup-epochs 0 --quant-lr-scale 0.1 --quant-level 8 \
 --opt adam --lr 2e-3 --warmup-lr 2e-5 --min-lr 2e-5 --max-norm 1.0 --auto-lr-scaling true \
 --loss 0.7 l1 0.3 ms-ssim_5x5 --train-metric psnr --eval-metric psnr ms-ssim"

MODEL_SMALL="--model MyHiNeRV --block-type convnext --channels 120 --channels-reduce 2.0 --depths 2 2 1 1 \
 --exps 4. 4. 1. 1. --stem-kernels 3 --kernels 3 3 3 3 \
 --scales-t 1 1 1 1 --scales-hw 5 3 2 2 --stem-paddings -1 -1 -1 --paddings -1 -1 -1 \
 --base-size -1 -1 -1 --base-grid-size -1 -1 -1 2 --base-grid-level 2 --base-grid-level-scale 2. 1. 1. 0.5 \
 --enc-type normalized+temp_local_grid --enc-grid-size -1 4 --enc-grid-level 3 --enc-grid-level-scale 2. 0.5 --enc-grid-depth-scale 1. 0.5 \
 --upsample-type trilinear --upsample-config matmul-th-w"

TWO_LEVEL="--base-channels 128 --base-depths 3 3 3 1 --base-exps 4. 4. 4. 1. --base-grid-size 150 -1 -1 2 \
 --p-epochs ${PE} --p-quant-epochs ${QE} --clip-len 16 --min-clip-len 12 --max-clip-len 20 --num-clips -1 --seg-mode adaptive"

MODULES_ON="--crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
 --w-fft 0.1 --w-scale 0.2 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.01 --aux-ramp-start 0.3 --aux-ramp-len 0.3 --hf-start 0.5 --hf-beta 1.0 \
 --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 --dyn-temperature 0.5 --patch-drop-ratio 0.25"

mkdir -p "${OUTROOT}/logs"
for WR in 0.5 1.0 2.0 4.0; do
  for BWR in 0.25 0.5 1.0; do
    NAME="lambda_${MODE}_w${WR}_b${BWR}"
    echo "==== [$(date '+%F %T')] GPU${GPU} ${NAME} ===="
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} myhinerbv_main.py ${COMMON} ${MODEL_SMALL} ${MODULES_ON} ${TWO_LEVEL} \
      --quant-mode ${MODE} --w-rate ${WR} --base-w-rate ${BWR} --use-flow true \
      --exp-name "${NAME}" 2>&1 | tee "${OUTROOT}/logs/${NAME}.log" | tail -2
  done
done
echo "==== [$(date '+%F %T')] lambda sweep done; summaries: ===="
for f in ${OUTROOT}/*/summary.txt; do echo "--- $f"; cat "$f"; done
