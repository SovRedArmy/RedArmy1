#!/bin/bash
# 消融第二波自动接力：A0 完成释放 GPU4 → 跑 A4、A5；A1 完成释放 GPU5 → 跑 A6、A7。
cd /remote-home/gdzj21/VideoCompression/MyHiNeRV
ROOT=output/ablation
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
PY=/remote-home/gdzj21/miniconda3/envs/NVRC/bin/python

wait_for() {  # 等待某变体 summary 出现（兼容单模型入口）
  while [ ! -f "$ROOT/$1/summary.txt" ] && [ ! -f "$ROOT/$1/summary.yaml" ] && [ ! -f "$ROOT/$1/results/single_summary.txt" ]; do sleep 300; done
}

wait_for A0-baseline
echo "[queue] A0 done -> GPU4: A4 A5 ($(date '+%F %T'))"
GPU=4 QUICK=1 SEQ=ReadySetGo PYTHON=$PY bash scripts/run_ablation.sh A4 >> $ROOT/logs/A4.driver.log 2>&1
GPU=4 QUICK=1 SEQ=ReadySetGo PYTHON=$PY bash scripts/run_ablation.sh A5 >> $ROOT/logs/A5.driver.log 2>&1
echo "[queue] GPU4 queue done ($(date '+%F %T'))"

wait_for A1-full-noise
echo "[queue] A1 done -> GPU5: A6 A7 ($(date '+%F %T'))"
GPU=5 QUICK=1 SEQ=ReadySetGo PYTHON=$PY bash scripts/run_ablation.sh A6 >> $ROOT/logs/A6.driver.log 2>&1
GPU=5 QUICK=1 SEQ=ReadySetGo PYTHON=$PY bash scripts/run_ablation.sh A7 >> $ROOT/logs/A7.driver.log 2>&1
echo "[queue] GPU5 queue done ($(date '+%F %T'))"
