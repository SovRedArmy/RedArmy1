#!/bin/bash
# UVG 复现: 等待全部训练完成后自动评估(FPS + 8/10bit量化 + Huffman)
PY=/remote-home/gdzj21/envs/nerv/bin/python
cd /remote-home/gdzj21/VideoCompression/INR/NeRV
export TMPDIR=/remote-home/gdzj21/tmp/nervtmp

# 训练→评估 的 (序列, GPU) 对应关系
PAIRS=("beauty:0" "bosphorus:1" "honeybee:2" "jockey:3" "readysetgo:4" "shakendry:5" "yachtride:6")
COMMON="-e 150 --lower-width 96 --num-blocks 1 --frame_gap 1 --outf repro_uvg \
  --embed 1.25_40 --stem_dim_num 512_1 --reduction 2 --expansion 1 --single_res \
  --loss Fusion6 --warmup 0.2 --lr_type cosine --strides 5 3 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish --fc_hw_dim 9_16_26"

# 1) 等待 7 个训练全部完成(最多等 8 小时)
echo "$(date +%H:%M) 等待训练完成..."
for i in $(seq 1 960); do
  done_cnt=0
  for pair in "${PAIRS[@]}"; do ds=${pair%%:*}; grep -q "Training complete" logs_uvg_${ds}.log 2>/dev/null && done_cnt=$((done_cnt+1)); done
  [ $done_cnt -eq 7 ] && break
  sleep 30
done
echo "$(date +%H:%M) 训练全部完成($done_cnt/7), 开始评估"

# 2) 逐个评估(每序列: FPS 测试 + 8bit/10bit 量化)
for pair in "${PAIRS[@]}"; do
  ds=${pair%%:*}; gpu=${pair##*:}
  CKPT=$(ls -d output/repro_uvg/$ds/*fc_9_16_26*/model_latest.pth 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "[$ds] 无 checkpoint, 跳过"; continue; fi
  echo "[$ds] 评估 $CKPT"
  CUDA_VISIBLE_DEVICES=$gpu $PY train_nerv.py $COMMON --dataset $ds --eval_only --eval_fps --weight "$CKPT" \
    > eval_uvg_${ds}_fps.log 2>&1
  for QB in 8 10; do
    CUDA_VISIBLE_DEVICES=$gpu $PY train_nerv.py $COMMON --dataset $ds --eval_only --quant_bit $QB --quant_axis 0 --weight "$CKPT" \
      > eval_uvg_${ds}_q${QB}.log 2>&1
  done
  echo "[$ds] 完成: $(grep 'FPS' eval_uvg_${ds}_fps.log | tail -1)"
  echo "[$ds] q8: $(grep -E 'PSNR/ms_ssim' eval_uvg_${ds}_q8.log | tail -1); $(grep Entropy eval_uvg_${ds}_q8.log)"
done
echo "=== UVG 全部评估完成 ==="
