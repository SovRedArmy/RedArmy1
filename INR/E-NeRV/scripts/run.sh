#!/bin/bash
# 训练启动脚本。用法:
#   bash scripts/run.sh <配置文件路径> <实验名EXP_NAME> <分布式端口>
# 例如:
#   bash scripts/run.sh ./cfgs/E-NeRV-bunny.yaml exp1 29500
set -x  # 打印执行的每条命令，便于排查

CFG_PATH=$1   # yaml 配置路径（如 ./cfgs/E-NeRV-bunny.yaml）
DIR=$2        # 实验名，决定输出目录 ./outputs/$DIR
PORT=$3       # 分布式训练通信端口

mkdir -p ./outputs/$DIR          # 创建输出目录（日志/checkpoint/tensorboard 都在这里）
TIME=$(date +"%Y%m%d_%H%M%S")    # 时间戳，用于 tensorboard 目录与日志文件命名

python -u main.py \              # -u 关闭输出缓冲，日志实时写入
    --cfg_path $CFG_PATH \       # 实验配置
    --output_dir ./outputs/$DIR \  # 输出目录
    --time_str $TIME \           # tensorboard 目录名后缀
    --port ${PORT} \             # 分布式端口（单机模式不生效）
    2>&1 | tee ./outputs/$DIR/$TIME.log  # stdout+stderr 同步存到日志文件
