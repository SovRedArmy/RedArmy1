#!/bin/bash
# NeRV 论文 UVG 复现: 7 个 UVG 1080p 序列 × NeRV-S, 论文消融协议(150 epoch, warmup 30)
# 数据: /remote-home/gdzj21/dataset/UVG (官方 ultravideo.fi 下载, YUV→PNG, 全帧 3900 帧)
PY=/remote-home/gdzj21/envs/nerv/bin/python
cd /remote-home/gdzj21/VideoCompression/INR/NeRV
export TMPDIR=/remote-home/gdzj21/tmp/nervtmp

COMMON="-e 150 --lower-width 96 --num-blocks 1 --frame_gap 1 --outf repro_uvg \
  --embed 1.25_40 --stem_dim_num 512_1 --reduction 2 --expansion 1 --single_res \
  --loss Fusion6 --warmup 0.2 --lr_type cosine --strides 5 3 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish --fc_hw_dim 9_16_26"

# 7 个序列各占一张 GPU(与 root 任务共卡)
CUDA_VISIBLE_DEVICES=0 $PY train_nerv.py $COMMON --dataset beauty      > logs_uvg_beauty.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY train_nerv.py $COMMON --dataset bosphorus   > logs_uvg_bosphorus.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY train_nerv.py $COMMON --dataset honeybee    > logs_uvg_honeybee.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY train_nerv.py $COMMON --dataset jockey      > logs_uvg_jockey.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY train_nerv.py $COMMON --dataset readysetgo  > logs_uvg_readysetgo.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY train_nerv.py $COMMON --dataset shakendry   > logs_uvg_shakendry.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY train_nerv.py $COMMON --dataset yachtride   > logs_uvg_yachtride.log 2>&1 &

echo "已启动 $(jobs -p | wc -l) 个 UVG 训练任务: $(jobs -p | tr '\n' ' ')"
wait
echo "=== UVG 全部训练完成 ==="
