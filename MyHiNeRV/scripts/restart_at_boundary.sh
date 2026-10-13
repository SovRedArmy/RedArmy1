#!/bin/bash
# restart_at_boundary.sh <VARIANT-KEY> <EXP-NAME> <CKPT> <NEXT-EPOCH> [屏蔽任务...]
#
# 在 checkpoint 边界安全重启（零 epoch 损失）：
#   1. 等 checkpoints/<CKPT> 落盘 且 日志出现 'Epoch <NEXT>:'（保证 checkpoint 完整写完，
#      若该边界有压缩探针，此行也在探针结束之后才出现）
#   2. 临时屏蔽队列中其他 WAIT 任务（attempts=99），防止它们抢本任务释放的 GPU
#   3. kill 训练进程 → 监督器检测到 EXIT 后用 run_ablation.sh 的【新配置】重新拉起
#      （hinerv_main 自动从最大 checkpoint_NNN 续训，epoch/optimizer/scheduler 全恢复）
#   4. 确认本任务重新运行后解除屏蔽
V=$1; NAME=$2; CKPT=$3; NEXT=$4; shift 4; BLOCKED="$@"
ROOT=/remote-home/gdzj21/VideoCompression/MyHiNeRV/output/ablation
DIR=$ROOT/$NAME; LOG=$DIR/rank_0.txt; WLOG=$ROOT/logs/restart_${V}.log
STATE=/remote-home/gdzj21/tmp/myhinerbv_queue.state
ATTEMPTS=/remote-home/gdzj21/tmp/myhinerbv_queue.attempts
logw() { echo "[$(date '+%F %T')] $*" >> "$WLOG"; }

logw "看护启动：等 $DIR/checkpoints/$CKPT 与 'Epoch ${NEXT}:' 日志行（将屏蔽: ${BLOCKED:-无}）"
while true; do
  if [ -d "$DIR/checkpoints/$CKPT" ] && grep -q "Train - Epoch ${NEXT}:" "$LOG" 2>/dev/null; then
    break
  fi
  sleep 60
done
logw "边界到达（checkpoint + Epoch${NEXT} 日志均已出现）。30s 落盘缓冲..."
sleep 30

# 2) 屏蔽其他 WAIT 任务
for b in $BLOCKED; do echo "$b 99" >> "$ATTEMPTS"; done
[ -n "$BLOCKED" ] && logw "已屏蔽: $BLOCKED"

# 记录 kill 前的旧 pgid（STATE 行），用于之后判断"新拉起"
OLDROW=$(awk -v v="$V" '$1==v{print $2}' "$STATE" | tail -1)
logw "旧进程组 pgid=${OLDROW:-未知}"

# 3) kill 目标 python（cmdline 含 --exp-name <NAME>；同管道的 tee 随 EOF 自然退出）
PID=$(pgrep -f "exp-name ${NAME}" | head -1)
logw "kill TERM pid=${PID}"
[ -n "$PID" ] && kill -TERM "$PID"
for i in $(seq 1 20); do pgrep -f "exp-name ${NAME}" >/dev/null || break; sleep 3; done
if pgrep -f "exp-name ${NAME}" >/dev/null; then
  pkill -KILL -f "exp-name ${NAME}"; logw "kill KILL 补刀"
fi

# 4) 等监督器重新拉起：STATE 中本任务的行必须是【新 pgid】且存活
#    （旧 wrapper 会在 python 死后存活数秒，kill -0 对旧 pid 的成功是假阳性）
logw "等待监督器用新配置拉起 ${V} ..."
ok=0
for i in $(seq 1 60); do
  NEWPID=$(awk -v v="$V" '$1==v{print $2}' "$STATE" | tail -1)
  if [ -n "$NEWPID" ] && [ "$NEWPID" != "$OLDROW" ] && kill -0 "$NEWPID" 2>/dev/null; then
    logw "${V} 已重新运行，pgid=${NEWPID}"; ok=1; break
  fi
  sleep 30
done

# 5) 解除屏蔽并验证新配置
awk '$2 != 99' "$ATTEMPTS" > "${ATTEMPTS}.tmp" && mv "${ATTEMPTS}.tmp" "$ATTEMPTS"
logw "已解除屏蔽"
if [ "$ok" = "1" ]; then
  sleep 240   # 等初始化完成
  NP=$(pgrep -f "exp-name ${NAME}" | head -1)
  if [ -n "$NP" ]; then
    OMPV=$(tr '\0' '\n' < /proc/$NP/environ 2>/dev/null | grep '^OMP_NUM_THREADS=' || echo "OMP未设置")
    WV=$(tr '\0' ' ' < /proc/$NP/cmdline 2>/dev/null | grep -oE '\-\-workers [0-9]+' || echo "--workers 未覆盖")
    logw "新进程 pid=${NP}：${OMPV}；${WV}"
    logw "重启流程完成 ✓"
  else
    logw "警告：新进程未找到，请人工检查"
  fi
else
  logw "警告：30 分钟内未观察到 ${V} 被拉起，请人工检查监督器"
fi
