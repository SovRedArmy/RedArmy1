#!/bin/bash
# Fast-NeRV 复现：单卡完整训练（规避与他人任务的 DDP 同步争抢；batch 32 与论文一致）
# - 数据: 预处理为 短边256/10fps 并预剪 1.2s 片段（等价仓库 LMDB 路径的固定窗口行为）
# - resize False：数据已是短边 256，跳过冗余 Resize（论文管线中 Resize 的等价效果
#   已在预处理时完成），CenterCrop/RandomCrop 语义不变
# - 训练集: K400 400 类 x 每类 25 = 10k；测试集: K400-val 19,877 + UCF101 3,783
cd /remote-home/gdzj21/VideoCompression/INR/FastNeRV
PY=/remote-home/gdzj21/miniconda3/envs/fastnerv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY run_trainer.py --cfg cfgs/nerv_enc_repro.yaml \
  --csv_file k400_train_trim.csv --out_path outputs/repro_full \
  --frame_num 8 --t_duration 1.0 --input_size 256 \
  -b 32 --tag repro8f1g -p 5310 -j 12 \
  --opts train_dataset.args.resize False \
    test_dataset.args.resize False \
    test_dataset.csv_paths.k400 k400_val_trim.csv \
    test_dataset.csv_paths.ucf101 ucf101_test_trim.csv \
    max_epoch 150 eval_epoch 50 vis_epoch 50 save_epoch 50
