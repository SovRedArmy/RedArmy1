#!/bin/bash
# MyHiNeRV 消融实验运行器
# 用法：GPU=4 SEQ=ReadySetGo QUICK=1 bash scripts/run_ablation.sh A0 A1 A2 ...
#   QUICK=1  快速档（约 1/2.5 训练量，用于快速验证模块贡献）
#   QUICK=0  完整档（300 epochs，论文级配置）
# 可选变体：
#   A0  HiNeRV 等价基线（单模型，全部新模块关闭，noise 量化）
#   A1  完整方法（两级 + 全部损失 + 动态采样/patch丢弃 + 自适应GOP，noise 量化）
#   A2  = A1 但 QEC 量化（λ: base 0.5 / GOP 1.5）
#   A3  = A1 但 Hyper 量化（DCVC-RT 式逐元素 μ/σ，λ: base 0.5 / GOP 1.5）
#   A4  完整方法 - 两级（单模型 + 全部损失 + 动态采样，noise）→ 隔离两级收益
#   A5  = A1 但固定等长 GOP（--seg-mode fixed）→ 隔离自适应切分收益
#   A6  = A1 但关闭 Scene Matters 四件套（FFT/流/对比/CRSPE）→ 隔离损失收益
#   A7  = A1 但关闭动态采样与 patch 丢弃 → 隔离 dynamicAL 收益
set -e
cd "$(dirname "$0")/.."

# 多进程/DataLoader 临时目录走本地 /dev/shm：NFS 上 pymp 目录清理会因 silly-rename
# 报 EBUSY 杀死主进程（2026-09-22 C1 连环崩溃根因）。根分区长期 93% 满不可用。
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null
export TMPDIR=/dev/shm/myhinerbv_mp

GPU=${GPU:-0}
SEQ=${SEQ:-ReadySetGo}
QUICK=${QUICK:-1}
DATASET=${DATASET:-/remote-home/gdzj21/dataset/UVG/png}
OUTROOT=${OUTROOT:-output/ablation}
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
 --quant-epochs ${QE} --quant-warmup-epochs 0 --quant-lr-scale 0.1 --quant-level 8 7 6 --quant-noise 0.9 \
 --opt adam --lr 2e-3 --warmup-lr 2e-5 --min-lr 2e-5 --max-norm 1.0 --auto-lr-scaling true \
 --loss 0.7 l1 0.3 ms-ssim_5x5 --train-metric psnr --eval-metric psnr ms-ssim"

# GOP 小模型 ≈0.41M（UAR-NVC 式小模型）；A0/A4 基线仍用 HiNeRV-S 等价结构（280ch，3.19M）
MODEL_SMALL="--model MyHiNeRV --block-type convnext --channels 120 --channels-reduce 2.0 --depths 2 2 1 1 \
 --exps 4. 4. 1. 1. --stem-kernels 3 --kernels 3 3 3 3 \
 --scales-t 1 1 1 1 --scales-hw 5 3 2 2 --stem-paddings -1 -1 -1 --paddings -1 -1 -1 \
 --base-size -1 -1 -1 --base-grid-size -1 -1 -1 2 --base-grid-level 2 --base-grid-level-scale 2. 1. 1. 0.5 \
 --enc-type normalized+temp_local_grid --enc-grid-size -1 4 --enc-grid-level 3 --enc-grid-level-scale 2. 0.5 --enc-grid-depth-scale 1. 0.5 \
 --upsample-type trilinear --upsample-config matmul-th-w"
# GOP 小模型 ≈0.41M；base-grid T 用 -1 跟随 GOP 长度（时间网格与 clip 对齐）
MODEL_HIENERV="--model MyHiNeRV --block-type convnext --channels 280 --channels-reduce 2.0 --depths 3 3 3 1 \
 --exps 4. 4. 4. 1. --stem-kernels 3 --kernels 3 3 3 3 \
 --scales-t 1 1 1 1 --scales-hw 5 3 2 2 --stem-paddings -1 -1 -1 --paddings -1 -1 -1 \
 --base-size -1 -1 -1 --base-grid-size 150 -1 -1 2 --base-grid-level 2 --base-grid-level-scale 2. 1. 1. 0.5 \
 --enc-type normalized+temp_local_grid --enc-grid-size -1 4 --enc-grid-level 3 --enc-grid-level-scale 2. 0.5 --enc-grid-depth-scale 1. 0.5 \
 --upsample-type trilinear --upsample-config matmul-th-w"

# 完整方法的两级配置
TWO_LEVEL="--base-channels 128 --base-depths 3 3 3 1 --base-exps 4. 4. 4. 1. \
 --p-epochs ${PE} --p-quant-epochs ${QE} --main-extra-epochs 180 --clip-len 16 --min-clip-len 12 --max-clip-len 20 --num-clips -1 --seg-mode adaptive --gop-mode residual"
# 全部新模块开启
MODULES_ON="--crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
 --w-fft 0.1 --w-scale 0.5 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.01 --hf-start 2. --hf-beta 1.0 \
 --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 --dyn-temperature 0.5 --patch-drop-ratio 0.25"
# 全部新模块关闭（HiNeRV 等价）
MODULES_OFF="--crspe-channels 0 --multiscale false --flow-head false \
 --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
 --dyn-enabled false --patch-drop-ratio 0."

run_variant() {
  local name=$1; shift
  local flags="$*"
  # OMP_THREADS=8 时限制 torch intra-op 线程数（CPU 超订下减少自旋浪费）
  # 注意：${VAR:+...} 展开的词不会被 bash 当作环境变量前缀，必须经 env 显式注入
  local omp=""
  [ -n "${OMP_THREADS:-}" ] && omp="env OMP_NUM_THREADS=${OMP_THREADS}"
  echo "==== [$(date '+%F %T')] GPU${GPU} ${name} ===="
  if echo "$flags" | grep -q "two-level-off"; then
    # 单模型入口（无两级）：去掉 two-level 标记
    flags=$(echo "$flags" | sed 's/ two-level-off//')
    CUDA_VISIBLE_DEVICES=${GPU} ${omp} ${PYTHON} hinerv_main.py ${COMMON} ${MODEL_SMALL} ${flags} \
      --exp-name "${name}" 2>&1 | tee "${OUTROOT}/logs/${name}.log" | grep -E "(Eval|Compressed model|BPP|Error|Traceback)" | tail -5
  else
    CUDA_VISIBLE_DEVICES=${GPU} ${omp} ${PYTHON} myhinerbv_main.py ${COMMON} ${MODEL_SMALL} ${flags} \
      --exp-name "${name}" 2>&1 | tee "${OUTROOT}/logs/${name}.log" | grep -E "(Eval|Compressed model|BPP|\*\*\*|Error|Traceback)" | tail -5
  fi
}

mkdir -p "${OUTROOT}/logs"

for V in "$@"; do
case $V in
A0) run_variant "A0-baseline" ${MODEL_HIENERV} ${MODULES_OFF} --quant-mode noise two-level-off ;;
A1) run_variant "A1-full-noise" ${MODULES_ON} ${TWO_LEVEL} --quant-mode noise --use-flow true ;;
A3)
    # 修复版两级架构（UAR-NVC 诊断落地）：
    #   1) base 用 B0（280ch/31.59dB 全视频模型）——替换 A1 的 26.1dB 弱 base，分支只需学微修正；
    #   2) 固定 60 帧 clips（10 个 GOP，回到 UAR-NVC 消融的甜点区，非 38 个）；
    #   3) 分支加宽 160ch（缓解 0.41M 容量饥饿，≤280 保证 derive 可映射）；
    #   4) GOP 训练改 QEC joint（率损失 = 残差熵，训练主动压残差——A1 纯失真训练的根治）。
    SRC_A3="${OUTROOT}/A3-fixed-2level"
    if [ -f "${SRC_A3}/base/decoded_state.pt" ] && [ -f "${SRC_A3}/clips.yaml" ]; then
        echo "A3: base=B0 snapshot + fixed-60 clips 就绪"
    else
        echo "A3: 材料未找到，拒绝启动"; exit 1
    fi
    EP=420 PREP=0 QE=12 OMP_THREADS=8 run_variant "A3-fixed-2level" ${MODULES_OFF} ${TWO_LEVEL} \
      --base-channels 280 --gop-channels 160 --p-epochs 120 \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 --prune-epochs 24 --quant-epochs 12 --epochs 420 --quant-mode qec --w-rate 1.5 --joint-rate true --rate-warmup-epochs 10 \
      --gop-init-mode interp \
      --use-flow false --clips-file "${SRC_A3}/clips.yaml" --workers 2 ;;
A1v2)
    # 两级架构修复版：base_ref 注册泄漏修复（A1 解码 10.8dB 的根因）+ UAR-NVC II-init 融合
    #（θ' = ε·derive(base) + (1−ε)·前一 GOP 解码权重，ε 由相邻 GOP 边界帧 MSE 自适应）。
    # 复用 A1 已完成的 base 与 GOP 切分，只重跑 GOP 链。
    SRC_A1="${OUTROOT}/A1-full-noise"
    DST_B="${OUTROOT}/A1v2-full-iinit/base"
    if [ -f "${SRC_A1}/base/decoded_state.pt" ] && [ -f "${SRC_A1}/base/patch_dynamics.npy" ]; then
        mkdir -p "${DST_B}"
        cp "${SRC_A1}/base/decoded_state.pt" "${SRC_A1}/base/patch_dynamics.npy" "${DST_B}/"
        cp -r "${SRC_A1}/base/bitstreams" "${DST_B}/" 2>/dev/null
        echo "A1v2: reused A1 base (decoded_state + dynamics + bitstreams)"
    else
        echo "A1v2: A1 base 未找到，拒绝启动"; exit 1
    fi
    run_variant "A1v2-full-iinit" ${MODULES_ON} ${TWO_LEVEL} \
      --quant-mode noise --use-flow true --gop-init-mode interp \
      --clips-file "${SRC_A1}/clips.yaml" ;;
A2) run_variant "A2-full-qec" ${MODULES_ON} ${TWO_LEVEL} --prune-epochs 0 --quant-epochs 12 --epochs 420 --quant-mode qec --w-rate 1.5 --base-w-rate 0.5 --use-flow true ;;
A3) run_variant "A3-full-hyper" ${MODULES_ON} ${TWO_LEVEL} --quant-mode hyper --w-rate 1.5 --base-w-rate 0.5 --use-flow true ;;
A4) run_variant "A4-single-full" ${MODEL_HIENERV} ${MODULES_ON} --quant-mode noise two-level-off ;;
B0) EP=420 PREP=24 QE=12 run_variant "B0-baseline-420" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled false \
      --quant-mode noise --use-flow false --eval-epochs 10 --compression-interval 60 --epochs 420 two-level-off ;;
B1) EP=420 PREP=24 QE=12 run_variant "B1-samp-only-420" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --quant-mode noise --use-flow false --eval-epochs 10 --compression-interval 60 --epochs 420 two-level-off ;;
B2) EP=420 PREP=24 QE=12 OMP_THREADS=8 run_variant "B2-samp-loss-420" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 --compression-interval 60 --epochs 420 --workers 2 two-level-off ;;
B3) EP=420 PREP=24 QE=12 run_variant "B3-aggressive-420" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.5 --dyn-sampling-w-min 0.3 --dyn-sampling-w-max 3.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 --compression-interval 60 --epochs 420 two-level-off ;;
B4) EP=420 PREP=24 QE=12 OMP_THREADS=8 run_variant "B4-cosine-samp-420" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 \
      --sampling-window 0.2,0.35,0.65,0.85 \
      --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 --compression-interval 60 --epochs 420 --workers 2 two-level-off ;;
B0r)
    # 加强版压缩：从已完成的 B0 复制浮点权重，60ep prune + 30ep QAT
    if [ -f "${OUTROOT}/B0-baseline-420/state_after_main.pt" ]; then
        mkdir -p "${OUTROOT}/B0r-baseline-enhanced"
        cp "${OUTROOT}/B0-baseline-420/state_after_main.pt" "${OUTROOT}/B0r-baseline-enhanced/"
        echo "B0r: copied state_after_main.pt from B0 (root)"
    else
        echo "B0r: B0 尚未完成（无 state_after_main.pt），拒绝启动"; exit 1
    fi
    OMP_THREADS=8 run_variant "B0r-baseline-enhanced" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled false \
      --quant-mode noise --use-flow false --eval-epochs 10 \
      --prune-epochs 60 --quant-epochs 30 --epochs 420 --workers 2 two-level-off ;;
B1r)
    if [ -f "${OUTROOT}/B1-samp-only-420/state_after_main.pt" ]; then
        mkdir -p "${OUTROOT}/B1r-samp-enhanced"
        cp "${OUTROOT}/B1-samp-only-420/state_after_main.pt" "${OUTROOT}/B1r-samp-enhanced/"
        echo "B1r: copied state_after_main.pt from B1 (root)"
    else
        echo "B1r: B1 尚未完成（无 state_after_main.pt），拒绝启动"; exit 1
    fi
    OMP_THREADS=8 run_variant "B1r-samp-enhanced" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --quant-mode noise --use-flow false --eval-epochs 10 \
      --prune-epochs 60 --quant-epochs 30 --epochs 420 --workers 2 two-level-off ;;
B2r)
    if [ -f "${OUTROOT}/B2-samp-loss-420/state_after_main.pt" ]; then
        mkdir -p "${OUTROOT}/B2r-samp-loss-enhanced"
        cp "${OUTROOT}/B2-samp-loss-420/state_after_main.pt" "${OUTROOT}/B2r-samp-loss-enhanced/"
        echo "B2r: copied state_after_main.pt from B2 (root)"
    else
        echo "B2r: B2 尚未完成（无 state_after_main.pt），拒绝启动"; exit 1
    fi
    OMP_THREADS=8 run_variant "B2r-samp-loss-enhanced" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 \
      --prune-epochs 60 --quant-epochs 30 --epochs 420 --workers 2 two-level-off ;;
B3r)
    if [ -f "${OUTROOT}/B3-aggressive-420/state_after_main.pt" ]; then
        mkdir -p "${OUTROOT}/B3r-aggressive-enhanced"
        cp "${OUTROOT}/B3-aggressive-420/state_after_main.pt" "${OUTROOT}/B3r-aggressive-enhanced/"
        echo "B3r: copied state_after_main.pt from B3 (root)"
    else
        echo "B3r: B3 尚未完成（无 state_after_main.pt），拒绝启动"; exit 1
    fi
    OMP_THREADS=8 run_variant "B3r-aggressive-enhanced" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.5 --dyn-sampling-w-min 0.3 --dyn-sampling-w-max 3.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 \
      --prune-epochs 60 --quant-epochs 30 --epochs 420 --workers 2 two-level-off ;;
B4r)
    if [ -f "${OUTROOT}/B4-cosine-samp-420/state_after_main.pt" ]; then
        mkdir -p "${OUTROOT}/B4r-cosine-enhanced"
        cp "${OUTROOT}/B4-cosine-samp-420/state_after_main.pt" "${OUTROOT}/B4r-cosine-enhanced/"
        echo "B4r: copied state_after_main.pt from B4 (root)"
    else
        echo "B4r: B4 尚未完成（无 state_after_main.pt），拒绝启动"; exit 1
    fi
    OMP_THREADS=8 run_variant "B4r-cosine-enhanced" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 \
      --sampling-window 0.2,0.35,0.65,0.85 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 \
      --prune-epochs 60 --quant-epochs 30 --epochs 420 --workers 2 two-level-off ;;
C1)
    # 联合率失真训练（UAR-NVC QEC 式）：B1 赢家配置 + 量化/率损失从主训练 epoch 0 激活。
    # 无剪枝（PREP=0，NVRC 同款：码率控制交给可学习 q_step + 熵模型）；
    # 保留 12ep 低 lr 终段（QE=12）作硬量化精调（对应 NVRC S2）。
    EP=420 PREP=0 QE=12 OMP_THREADS=8 run_variant "C1-joint-qec-420" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 --quant-mode qec --w-rate 1.5 --joint-rate true --rate-warmup-epochs 15 \
      --use-flow false --eval-epochs 10 --compression-interval 60 \
      --epochs 420 --workers 2 two-level-off ;;
C2)
    # C 系列第二 RD 点：更强率约束（w_rate=8）把 q_step 推大、码率锚进 ~0.013bpp
    # 区间，与 B 系列曲线同台 BD-RATE。其余与 C1 完全一致（B1 采样配置）。
    EP=420 PREP=0 QE=12 OMP_THREADS=8 run_variant "C2-joint-qec-w8" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 --quant-mode qec --w-rate 8.0 --joint-rate true --rate-warmup-epochs 15 \
      --use-flow false --eval-epochs 10 --compression-interval 60 \
      --epochs 420 --workers 2 two-level-off ;;
C3)
    # 两个赢家叠加：B4 余弦窗口采样（数据侧）× joint QEC（权重侧）。w_rate 与 C1 一致
    # （同码率锚下隔离采样系统的贡献：C3 vs C1）。
    EP=420 PREP=0 QE=12 OMP_THREADS=8 run_variant "C3-cosine-joint" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 \
      --sampling-window 0.2,0.35,0.65,0.85 --dyn-loss-alpha 0.15 \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 --quant-mode qec --w-rate 1.5 --joint-rate true --rate-warmup-epochs 15 \
      --use-flow false --eval-epochs 10 --compression-interval 60 \
      --epochs 420 --workers 2 two-level-off ;;
M1)
    # M 档单模型对照（400ch ≈ 6.5M）：A3 两级方案的参数/码率量级对位 baseline。
    # 与 C1 同管线（qec joint + 无剪枝），仅模型加大——隔离"多模型特化 vs 直接加大单模型"。
    EP=420 OMP_THREADS=8 run_variant "M1-qec-m400" ${MODEL_HIENERV} \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 --channels 400 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --quant-mode qec --w-rate 1.5 --joint-rate true --rate-warmup-epochs 15 \
      --use-flow false --eval-epochs 10 --compression-interval 60 --workers 2 two-level-off ;;
C2b)
    # C2 修正版：w_rate=8 + 真无剪枝（C2 被旧配置 bug 带进了 prune24 混合管线而崩：
    # float 30.69 → pruned 25.43，率感知训练的模型不宜再走重剪枝——这本身是发现）。
    EP=420 OMP_THREADS=8 run_variant "C2b-joint-w8-noprune" ${MODEL_HIENERV} \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --quant-mode qec --w-rate 8.0 --joint-rate true --rate-warmup-epochs 15 \
      --use-flow false --eval-epochs 10 --compression-interval 60 --workers 2 two-level-off ;;
H1)
    # 杂交方案：B4 传统浮点主训练（已完成）+ QEC 率感知 QAT 尾段（无剪枝）。
    # 融合两端优势：B4 的浮点质量（31.79）+ C1 式压缩转化效率（损耗 0.53dB@12ep）。
    SRC_H="${OUTROOT}/H1-b4-qec-qat"
    if [ -f "${OUTROOT}/B4-cosine-samp-420/state_after_main.pt" ]; then
        mkdir -p "${SRC_H}"
        cp "${OUTROOT}/B4-cosine-samp-420/state_after_main.pt" "${SRC_H}/"
        echo "H1: B4 snapshot 就绪"
    else
        echo "H1: B4 snapshot 未找到，拒绝启动"; exit 1
    fi
    OMP_THREADS=8 run_variant "H1-b4-qec-qat" ${MODEL_HIENERV} \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 \
      --sampling-window 0.2,0.35,0.65,0.85 --dyn-loss-alpha 0.15 \
      --quant-mode qec --w-rate 1.5 \
      --use-flow false --eval-epochs 10 --workers 2 two-level-off ;;
C4)
    # NVRC 机制反向移植：soft-round-noise 软量化退火（0.5→0.3）+ D/R 交替（K=8）。
    # 其余与 C1 完全一致（B1 采样, w_rate=1.5, joint, 无剪枝）——隔离训练机器的贡献。
    EP=420 OMP_THREADS=8 run_variant "C4-nvrc-mech" ${MODEL_HIENERV} \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --quant-mode qec --w-rate 1.5 --joint-rate true --rate-warmup-epochs 15 \
      --quant-soft-anneal "0.5,0.3" --dr-alternate 8 \
      --use-flow false --eval-epochs 10 --compression-interval 60 --workers 2 two-level-off ;;
B0s)
    # Bs 超强压缩：120ep 剪枝恢复 + 60ep QAT（检验 B2/B4 压缩预算扩大后的差距趋势）
    SRC="${OUTROOT}/B0-baseline-420"
    [ -f "${SRC}/state_after_main.pt" ] || { echo "B0s: snapshot missing"; exit 1; }
    mkdir -p "${OUTROOT}/B0s-super-120p60q"; cp "${SRC}/state_after_main.pt" "${OUTROOT}/B0s-super-120p60q/"
    OMP_THREADS=8 run_variant "B0s-super-120p60q" ${MODEL_HIENERV} \
      --prune-epochs 120 --quant-epochs 60 --epochs 420 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled false --quant-mode noise --use-flow false --eval-epochs 10 --workers 2 two-level-off ;;
B2s)
    SRC="${OUTROOT}/B2-samp-loss-420"
    [ -f "${SRC}/state_after_main.pt" ] || { echo "B2s: snapshot missing"; exit 1; }
    mkdir -p "${OUTROOT}/B2s-super-120p60q"; cp "${SRC}/state_after_main.pt" "${OUTROOT}/B2s-super-120p60q/"
    OMP_THREADS=8 run_variant "B2s-super-120p60q" ${MODEL_HIENERV} \
      --prune-epochs 120 --quant-epochs 60 --epochs 420 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 --workers 2 two-level-off ;;
B4s)
    SRC="${OUTROOT}/B4-cosine-samp-420"
    [ -f "${SRC}/state_after_main.pt" ] || { echo "B4s: snapshot missing"; exit 1; }
    mkdir -p "${OUTROOT}/B4s-super-120p60q"; cp "${SRC}/state_after_main.pt" "${OUTROOT}/B4s-super-120p60q/"
    OMP_THREADS=8 run_variant "B4s-super-120p60q" ${MODEL_HIENERV} \
      --prune-epochs 120 --quant-epochs 60 --epochs 420 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 \
      --sampling-window 0.2,0.35,0.65,0.85 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow false --eval-epochs 10 --workers 2 two-level-off ;;
D1)
    # D 系列：列级结构化稀疏 + 率感知联合训练（FaNeRV+TACO 启发）
    # 列级剪枝（30% 输出通道置零）比逐元素剪枝更抗耦合；
    # 在 joint 训练中渐进推进，模型在率约束下主动适应稀疏。
    EP=420 OMP_THREADS=8 run_variant "D1-colprune-joint" ${MODEL_HIENERV} \
      --prune-epochs 0 --quant-epochs 12 --epochs 420 \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --quant-mode qec --w-rate 1.5 --joint-rate true --rate-warmup-epochs 15 \
      --quant-soft-anneal "0.5,0.3" --dr-alternate 8 \
      --col-prune-target 0.30 --col-prune-start 60 --col-prune-end 300 --col-prune-every 20 \
      --use-flow false --eval-epochs 10 --compression-interval 60 --workers 2 two-level-off ;;
B5)
    # B5: Tree-NeRV 式层级采样 vs 平面采样（隔离采样策略的贡献）
    # 使用帧级权重（时间变化驱动）+ patch 级权重（μ+λσ LossGrad）的混合采样
    EP=420 PREP=60 QE=30 OMP_THREADS=8 run_variant "B5-treesamp-420" ${MODEL_HIENERV} \
      --crspe-channels 0 --multiscale false --flow-head false \
      --w-fft 0. --w-scale 0. --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.15 \
      --sampling-window 0.2,0.35,0.65,0.85 \
      --sampling-mode hierarchical --tree-ratio 0.5 \
      --quant-mode noise --use-flow false --eval-epochs 10 --workers 2 two-level-off ;;
A4R2a) run_variant "A4R2a-samp-only" ${MODEL_HIENERV} \
      --crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
      --w-fft 0.1 --w-scale 0.2 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.005 \
      --aux-window 0.2,0.35,0.65,0.85 --hf-start 0.5 --hf-end 0.8 \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.0 \
      --quant-mode noise --use-flow true --eval-epochs 10 two-level-off ;;
A4R2b) run_variant "A4R2b-moderate" ${MODEL_HIENERV} \
      --crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
      --w-fft 0.1 --w-scale 0.2 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.005 \
      --aux-window 0.2,0.35,0.65,0.85 --hf-start 0.5 --hf-end 0.8 \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow true --eval-epochs 10 two-level-off ;;
A4R2c) run_variant "A4R2c-strong-loss" ${MODEL_HIENERV} \
      --crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
      --w-fft 0.1 --w-scale 0.2 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.005 \
      --aux-window 0.2,0.35,0.65,0.85 --hf-start 0.5 --hf-end 0.8 \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.0 --dyn-sampling-w-min 0.5 --dyn-sampling-w-max 2.0 --dyn-loss-alpha 0.30 \
      --quant-mode noise --use-flow true --eval-epochs 10 two-level-off ;;
A4R2d) run_variant "A4R2d-aggressive" ${MODEL_HIENERV} \
      --crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
      --w-fft 0.1 --w-scale 0.2 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.005 \
      --aux-window 0.2,0.35,0.65,0.85 --hf-start 0.5 --hf-end 0.8 \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 1.5 --dyn-sampling-w-min 0.3 --dyn-sampling-w-max 3.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow true --eval-epochs 10 two-level-off ;;
A4R2e) run_variant "A4R2e-loss-only" ${MODEL_HIENERV} \
      --crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
      --w-fft 0.1 --w-scale 0.2 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.005 \
      --aux-window 0.2,0.35,0.65,0.85 --hf-start 0.5 --hf-end 0.8 \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 \
      --dyn-sampling-temp 0.0 --dyn-sampling-w-min 1.0 --dyn-sampling-w-max 1.0 --dyn-loss-alpha 0.15 \
      --quant-mode noise --use-flow true --eval-epochs 10 two-level-off ;;
A4R) run_variant "A4R-single-improved" ${MODEL_HIENERV} \
      --crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
      --w-fft 0.1 --w-scale 0.2 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.005 \
      --aux-window 0.2,0.35,0.65,0.85 --hf-start 0.5 --hf-end 0.8 \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 --dyn-temperature 0.5 \
      --patch-drop-ratio 0.15 --patch-drop-start 0.3 --patch-drop-end 0.7 \
      --quant-mode noise --use-flow true two-level-off ;;
A5) run_variant "A5-fixedgop" ${MODULES_ON} ${TWO_LEVEL} --seg-mode fixed --quant-mode noise --use-flow true ;;
A6) run_variant "A6-no-scenematters" ${MODEL_SMALL} \
      --crspe-channels 0 --multiscale true --flow-head false \
      --w-fft 0. --w-scale 0.5 --w-flow 0. --w-flow-ent 0. --w-tcl 0. --hf-start 2. --hf-beta 1.0 \
      --dyn-enabled true --dyn-update-epochs 5 --dyn-warmup-epochs 10 --dyn-temperature 0.5 --patch-drop-ratio 0.25 \
      ${TWO_LEVEL} --quant-mode noise --use-flow false ;;
A7) run_variant "A7-no-dyn" ${MODEL_SMALL} --crspe-channels 32 --crspe-content-init true --multiscale true --flow-head true \
      --w-fft 0.1 --w-scale 0.5 --w-flow 0.01 --w-flow-ent 0.001 --w-tcl 0.01 --hf-start 2. --hf-beta 1.0 \
      --dyn-enabled false --patch-drop-ratio 0. \
      ${TWO_LEVEL} --quant-mode noise --use-flow true ;;
*) echo "unknown variant $V";;
esac
done
echo "==== [$(date '+%F %T')] all requested variants done (GPU${GPU}) ===="
