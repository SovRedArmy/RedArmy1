#!/bin/bash
# NeRV 论文复现训练脚本: 1080p HEVC Class B
# 协议: 论文消融实验设定(150 epochs, warmup 0.2 -> 30 epochs), strides 5 3 2 2 2 为 1080p 设定
PY=/remote-home/gdzj21/envs/nerv/bin/python
cd /remote-home/gdzj21/VideoCompression/INR/NeRV
export TMPDIR=/remote-home/gdzj21/tmp/nervtmp

COMMON="-e 150 --lower-width 96 --num-blocks 1 --frame_gap 1 --outf repro_classb \
  --embed 1.25_40 --stem_dim_num 512_1 --reduction 2 --expansion 1 --single_res \
  --loss Fusion6 --warmup 0.2 --lr_type cosine --strides 5 3 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish"

# NeRV-S (fc 9_16_26) × 4 个序列
CUDA_VISIBLE_DEVICES=0 $PY train_nerv.py $COMMON --dataset cactus          --fc_hw_dim 9_16_26  > logs_cactus_S.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY train_nerv.py $COMMON --dataset basketballdrive --fc_hw_dim 9_16_26  > logs_basketballdrive_S.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY train_nerv.py $COMMON --dataset parkscene       --fc_hw_dim 9_16_26  > logs_parkscene_S.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY train_nerv.py $COMMON --dataset kimono          --fc_hw_dim 9_16_26  > logs_kimono_S.log 2>&1 &

# NeRV-M (9_16_58) 和 NeRV-L (9_16_112) 在 kimono 上做模型大小对比(论文 Table: 3.2M/6.3M/12.5M)
CUDA_VISIBLE_DEVICES=5 $PY train_nerv.py $COMMON --dataset kimono --fc_hw_dim 9_16_58  > logs_kimono_M.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY train_nerv.py $COMMON --dataset kimono --fc_hw_dim 9_16_112 > logs_kimono_L.log 2>&1 &

echo "已启动 $(jobs -p | wc -l) 个训练任务, PID: $(jobs -p | tr '\n' ' ')"
wait
echo "=== 全部训练完成 ==="
