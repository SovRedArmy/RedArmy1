#!/bin/bash
# 看护：ReadySetGo 复现的 Q6 码流结果落盘后，自动接力启动 YachtRide 复现（GPU0）
WLOG=/remote-home/gdzj21/VideoCompression/INR/HiNeRV/repro/yacht_watcher.log
R=/remote-home/gdzj21/VideoCompression/INR/HiNeRV/repro/ReadySetGo/ReadySetGo-HiNeRV-20260925-070818-a3b0898f/results
echo "[$(date '+%F %T')] 看护启动：等待 $R/Q6.txt" >> "$WLOG"
while true; do
  if [ -f "$R/Q6.txt" ]; then
    echo "[$(date '+%F %T')] ReadySetGo Q6 已落盘，3 分钟后启动 YachtRide" >> "$WLOG"
    sleep 180
    bash /remote-home/gdzj21/VideoCompression/INR/HiNeRV/repro/run_YachtRide.sh 0 >> "$WLOG" 2>&1 &
    echo "[$(date '+%F %T')] YachtRide 复现已启动（GPU0）" >> "$WLOG"
    exit 0
  fi
  sleep 300
done
