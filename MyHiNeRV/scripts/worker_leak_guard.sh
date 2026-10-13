#!/bin/bash
# worker_leak_guard.sh — DataLoader 泄漏 worker 的内存保护清理器
# 背景：dynamics 每 5ep 重建 DataLoader（persistent_workers=True），旧 loader 被
# accelerate 注册表持有不释放 → 每次泄漏 2 个空闲 worker。平时无害（0.1% CPU，
# COW 共享内存），但数量随训练线性增长。
# 策略：仅当系统可用内存 < GUARD_GB 时，对每个运行中的 exp-name 任务保留
# 主进程 + 最新 2 个 worker，杀掉更旧的空闲 worker。
GUARD_GB=${GUARD_GB:-45}
LOG=/remote-home/gdzj21/VideoCompression/MyHiNeRV/output/ablation/logs/leak_guard.log
while true; do
  AVAIL=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$AVAIL" -lt "$GUARD_GB" ]; then
    echo "[$(date '+%F %T')] 可用内存 ${AVAIL}GB < ${GUARD_GB}GB，开始清理泄漏 worker" >> "$LOG"
    for NAME in $(ps -eo cmd | grep -oE "exp-name [A-Za-z0-9_-]+" | awk '{print $2}' | sort -u); do
      PIDS=$(pgrep -f "exp-name ${NAME}$" 2>/dev/null; pgrep -f "exp-name ${NAME} " 2>/dev/null)
      PIDS=$(ps -eo pid,cmd | grep "exp-name ${NAME}" | grep -v grep | awk '{print $1}' | sort -n)
      N=$(echo "$PIDS" | wc -l)
      if [ "$N" -gt 3 ]; then
        MAIN=$(echo "$PIDS" | head -1)          # 最老的是主进程
        KEEP2=$(echo "$PIDS" | tail -2)          # 最新 2 个 = 当前活跃 worker
        VICTIMS=$(echo "$PIDS" | sed 1d | grep -vxF -e $(echo "$KEEP2" | head -1) -e $(echo "$KEEP2" | tail -1) || true)
        for v in $VICTIMS; do kill -9 "$v" 2>/dev/null; done
        echo "[$(date '+%F %T')] ${NAME}: ${N} 个进程，保留主进程 ${MAIN} + 最新 2 worker，清理 $(echo "$VICTIMS" | wc -l) 个" >> "$LOG"
      fi
    done
    sleep 300
  fi
  sleep 120
done
