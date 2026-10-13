#!/bin/bash
# 消融任务监视器：全部 summary 出现 → 退出码 0（触发分析）；
# 任一任务日志 mtime 连续两次轮询（10 分钟）不变且无 summary → 退出码 2（异常）；
# 50 小时超时 → 退出码 3。
# 注意：共享 FS 的文件时间戳存在时钟偏差，不能与墙钟比较，只用相对变化。
ROOT=/remote-home/gdzj21/VideoCompression/MyHiNeRV/output/ablation
VARIANTS="A0-baseline A1-full-noise A2-full-qec A3-full-hyper"

mtime_of() { stat -c %Y "$1" 2>/dev/null || echo 0; }

declare -A PREV
for v in $VARIANTS; do PREV[$v]=$(mtime_of "${ROOT}/logs/${v}.log"); done

for i in $(seq 1 300); do
  sleep 600  # 先等待再比较（首轮 PREV 与当前必然相同，不能立即判停滞）
  n=0
  for v in $VARIANTS; do
    if [ -f "${ROOT}/${v}/summary.txt" ] || [ -f "${ROOT}/${v}/summary.yaml" ] || [ -f "${ROOT}/${v}/results/single_summary.txt" ]; then n=$((n+1)); fi
  done
  if [ "$n" -ge 4 ]; then
    echo "ALL_DONE ($(date '+%F %T'))"
    exit 0
  fi
  for v in $VARIANTS; do
    done_flag=$([ -f "${ROOT}/${v}/summary.txt" ] || [ -f "${ROOT}/${v}/summary.yaml" ] || [ -f "${ROOT}/${v}/results/single_summary.txt" ] && echo 1 || echo 0)
    cur=$(mtime_of "${ROOT}/logs/${v}.log")
    if [ "$done_flag" = "0" ] && [ "${cur}" = "${PREV[$v]}" ] && [ "${cur}" != "0" ]; then
      echo "STALLED: ${v} 日志两次轮询无更新 ($(date '+%F %T'))"
      exit 2
    fi
    PREV[$v]=$cur
  done
done
echo "TIMEOUT after 50h ($(date '+%F %T'))"
exit 3
