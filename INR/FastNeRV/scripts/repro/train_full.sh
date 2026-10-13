#!/bin/bash
# Fast-NeRV 复现：完整训练（论文 Table 2 主实验，8 帧 / 256px / 150 epoch）
# - 训练集: K400 400 类 x 每类 25 个 = 10k 视频（cls_vid_num 400_25，与论文一致）
# - 测试集: K400-val 全量 19,877 + UCF101 官方测试集 3,783
# - batch 32(全局), AdamW 1e-4, step 衰减, MSE 损失 —— 均按论文附录 0.A.5
cd /remote-home/gdzj21/VideoCompression/INR/FastNeRV
PY=/remote-home/gdzj21/miniconda3/envs/fastnerv/bin/python
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY run_trainer.py --cfg cfgs/nerv_enc_repro.yaml --replace \
  --csv_file k400_train_paper.csv --out_path outputs/repro_full \
  --frame_num 8 --t_duration 1.0 --input_size 256 \
  -b 32 --tag repro8f -p 5300 -j 64 \
  --opts max_epoch 150 eval_epoch 50 vis_epoch 50 save_epoch 50
