#!/bin/bash
# NeRV 复现结果评估(顺序执行, 完整日志): 解码FPS + 量化(8/10bit)+Huffman熵编码
# 用法: ./eval_repro.sh <dataset> <fc_hw_dim> <gpu_id> [额外参数]
PY=/remote-home/gdzj21/envs/nerv/bin/python
cd /remote-home/gdzj21/VideoCompression/INR/NeRV
export TMPDIR=/remote-home/gdzj21/tmp/nervtmp

DS=$1; FC=$2; GPU=${3:-0}; EXTRA=${4:-}
COMMON="-e 150 --lower-width 96 --num-blocks 1 --frame_gap 1 --outf repro_classb \
  --embed 1.25_40 --stem_dim_num 512_1 --reduction 2 --expansion 1 --single_res \
  --loss Fusion6 --warmup 0.2 --lr_type cosine --strides 5 3 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish --dataset $DS --fc_hw_dim $FC"

CKPT=$(ls -d output/repro_classb/$DS/*fc_${FC}*/model_latest.pth 2>/dev/null | head -1)
TAG=${DS}_$(echo $FC | tr '_' '-')
if [ -z "$CKPT" ]; then echo "[$TAG] 找不到 checkpoint!"; exit 1; fi
echo "[$TAG] 评估 ckpt=$CKPT"

# 1) 解码速度测试 (每帧前向10次)
CUDA_VISIBLE_DEVICES=$GPU $PY train_nerv.py $COMMON --eval_only --eval_fps --weight "$CKPT" $EXTRA \
  > eval_${TAG}_fps.log 2>&1

# 2) 量化 + Huffman: 8bit / 10bit 两个码率点
for QB in 8 10; do
  CUDA_VISIBLE_DEVICES=$GPU $PY train_nerv.py $COMMON --eval_only --quant_bit $QB --quant_axis 0 --weight "$CKPT" $EXTRA \
  > eval_${TAG}_q${QB}.log 2>&1
done

echo "[$TAG] === 评估完成, 结果: ==="
grep -h -E "FPS" eval_${TAG}_fps.log | tail -2
grep -h -E "Entropy|PSNR/ms_ssim" eval_${TAG}_q8.log eval_${TAG}_q10.log
