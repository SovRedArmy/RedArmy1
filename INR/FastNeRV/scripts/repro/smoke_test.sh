#!/bin/bash
# Fast-NeRV 复现：smoke test（2 卡 DDP、~40 个训练视频、2 epoch，验证完整管线）
# 注：z_norm=bn_bn 使用 SyncBatchNorm，需要 DDP 进程组，故至少 2 卡。
cd /remote-home/gdzj21/VideoCompression/INR/FastNeRV
PY=/remote-home/gdzj21/miniconda3/envs/fastnerv/bin/python
CUDA_VISIBLE_DEVICES=0,1 $PY run_trainer.py --cfg cfgs/nerv_enc_repro.yaml --replace \
  --csv_file k400_train_paper.csv --out_path outputs/repro_smoke \
  --frame_num 8 --t_duration 1.0 --input_size 256 \
  -b 8 --tag smoke -p 5250 -j 8 \
  --opts train_dataset.args.cls_vid_num 5_8 \
    max_epoch 2 eval_epoch 2 vis_epoch 100 save_epoch 100 \
    test_dataset.csv_paths.k400 k400_val_tiny.csv \
    test_dataset.csv_paths.ucf101 ucf101_tiny.csv \
    test_dataset.args.cls_vid_num 400_1000000
