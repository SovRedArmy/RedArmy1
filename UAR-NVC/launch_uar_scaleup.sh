#!/bin/bash
# UAR-NVC 放大复现: 2 个模型规模 × 5 个独立组并行 (GOM=5, 组间 ref 链重启已确认)
# 预算 I=60/P=24 代码 epoch (≈ 现有匹配运行的 3 倍)
cd /remote-home/gdzj21/VideoCompression/UAR-NVC
export PATH=/remote-home/gdzj21/envs/nerv/bin:/root/anaconda3/envs/hinerv/bin:$PATH
export TMPDIR=/remote-home/gdzj21/VideoCompression/MyINR/tmp
PY=/remote-home/gdzj21/envs/nerv/bin/python
mkdir -p logs_scaleup

# ms=0.9 → GPUs 4,5,6,7,0
for g in 0 1 2 3 4; do
  gpu=$(( (g + 4) % 8 ))          # 4,5,6,7,0
  env CUDA_VISIBLE_DEVICES=$gpu $PY -u main.py --model_name HiNeRV --dataset Bunny \
      --data_path ./data/bunny --gop 6 --gom 5 --modelsize 0.9 \
      --epochs 60 --local_epochs 24 --lr 0.002 --group-only $g \
      > logs_scaleup/ms09_g$g.log 2>&1 &
done
# ms=1.8 → GPUs 1,2,3,4,5
for g in 0 1 2 3 4; do
  gpu=$(( g == 0 ? 1 : (g == 1 ? 2 : (g == 2 ? 3 : (g == 3 ? 4 : 5))) ))
  env CUDA_VISIBLE_DEVICES=$gpu $PY -u main.py --model_name HiNeRV --dataset Bunny \
      --data_path ./data/bunny --gop 6 --gom 5 --modelsize 1.8 \
      --epochs 60 --local_epochs 24 --lr 0.002 --group-only $g \
      > logs_scaleup/ms18_g$g.log 2>&1 &
done
echo "launched 10 procs"
