#!/bin/bash
# 补充实验：梯度下降 NeRV 基线（论文 Fig.1/7 中被加速 10^4 倍的对照组）
# 对 3 个 K400 val 片段（与 NeRV-Enc 测试相同的 1.2s 预处理片段）逐个过拟合 NeRV：
# 8 帧、256px、1500 epoch、batch 2、Adam lr 1e-3 cosine —— NeRV 原始训练设置。
cd /remote-home/gdzj21/VideoCompression/INR/FastNeRV/projects/nerv_baseline
PY=/remote-home/gdzj21/miniconda3/envs/fastnerv/bin/python
TRIM=/remote-home/gdzj21/VideoCompression/INR/FastNeRV/data/processed_trim/val

run_one () {  # $1=video_path $2=vid_name
  CUDA_VISIBLE_DEVICES=1 $PY train_hypoconv.py -e 1500 \
    --vid_path "$1" --vid "$2" \
    --crop_size 256 --hid_dim 32 --strds 8 4 4 2 --ks 1 3 \
    --pe_dim 64 --pe_sigma 1024 --act gelu --out_bias tanh \
    -b 2 --lr 0.001 --eval_freq 500 --frames 8 --not_resume --overwrite \
    --outf repro_$2
}

run_one $TRIM/00cwEcZZcu4_000003_000013.mp4 k400val_01
run_one $TRIM/00U9x58IraQ_000007_000017.mp4 k400val_02
run_one $TRIM/018EClOtVTM_000035_000045.mp4 k400val_03
