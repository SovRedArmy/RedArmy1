#!/bin/bash
# DCVC-RT 速度复现脚本: 在本机 RTX A6000 (GPU1) 上运行论文的速度实验。
# 用法: bash run_repro.sh <mode>
#   paper  : 论文原始配置 (融合 CUDA kernel, 无 force_zero_thres)
#   readme : README 推荐配置 (融合 CUDA kernel, force_zero_thres=0.12)
#   nofuse : 纯 PyTorch 回退 (禁用融合 kernel, 用于量化 kernel 融合的贡献)
set -e
MODE=${1:-paper}
PY=/remote-home/gdzj21/miniconda3/envs/dcvcrt/bin/python
export CUDA_VISIBLE_DEVICES=1
cd /remote-home/gdzj21/VideoCompression/DCVC/DCVC-family/DCVC-RT

FZ_ARGS=()
if [ "$MODE" = "readme" ]; then
    FZ_ARGS=(--force_zero_thres 0.12)
fi
if [ "$MODE" = "nofuse" ]; then
    # 用影子模块屏蔽真实 CUDA 扩展, 强制走纯 PyTorch 回退路径
    export PYTHONPATH=/remote-home/gdzj21/tmp/fuse_off_shim
fi

$PY test_video.py \
    --model_path_i ./checkpoints/repro_random_image.pth.tar \
    --model_path_p ./checkpoints/repro_random_video.pth.tar \
    --rate_num 4 \
    --test_config ./dataset_config_repro_uvg.json \
    --cuda 1 -w 1 --write_stream 1 \
    --force_intra_period -1 --reset_interval 64 \
    --force_frame_num 100 \
    --check_existing 0 --verbose 1 --verbose_json 0 \
    --stream_path /remote-home/gdzj21/tmp/dcvcrt_bins \
    --output_path /remote-home/gdzj21/tmp/dcvcrt_out/repro_${MODE}.json \
    "${FZ_ARGS[@]}" 2>&1 | grep -vE "^ Frame|it/s|it\]" | tail -30
