#!/bin/bash
# GPU 空闲监视 + 自动重启队列：扫描所有 GPU，发现整卡空闲（显存<5GB 且未被本队列占用）
# 就按优先级启动下一个消融任务（支持断点续跑，A1/A2/A5 会自动跳过已完成的 base）。
# 崩溃自动重试最多 2 次。队列全部完成或 48 小时后退出。
cd /remote-home/gdzj21/VideoCompression/MyHiNeRV
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
PY=/remote-home/gdzj21/miniconda3/envs/NVRC/bin/python
ROOT=output/ablation
STATE=/tmp/myhinerbv_queue.state
ATTEMPTS=/tmp/myhinerbv_queue.attempts
QUEUE="A1 A2 A3 A5 A6 A7"
MAX_ATTEMPTS=2

touch $STATE $ATTEMPTS

variant_dir() { ls -d $ROOT/$1-* 2>/dev/null | head -1; }

is_done() {
  local d=$(variant_dir $1)
  [ -n "$d" ] && ls $d/summary.txt $d/summary.yaml $d/results/single_summary.txt 2>/dev/null | grep -q .
}

state_has_variant() { awk -v v="$1" '$1==v{f=1} END{exit !f}' $STATE; }
state_has_gpu() { awk -v g="$1" '$3==g{f=1} END{exit !f}' $STATE; }
attempts() { awk -v v="$1" '$1==v{print $2}' $ATTEMPTS; }

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 5000 {print $1}'
}

launch() {  # launch <variant> <gpu>
  local v=$1 g=$2
  echo "[$(date '+%F %T')] launch ${v} on GPU${g} (attempt $(($(attempts $v)+1)))"
  GPU=$g QUICK=1 SEQ=ReadySetGo PYTHON=$PY nohup bash /remote-home/gdzj21/VideoCompression/MyHiNeRV/scripts/run_ablation.sh $v \
    >> /remote-home/gdzj21/VideoCompression/MyHiNeRV/$ROOT/logs/${v}.driverq.log 2>&1 &
  local pid=$!
  echo "$v $pid $g" >> $STATE
  awk -v v="$v" '$1!=v' $ATTEMPTS > $ATTEMPTS.tmp && mv $ATTEMPTS.tmp $ATTEMPTS
  echo "$v $(($(attempts $v)+1))" >> $ATTEMPTS
  sleep 180  # 等待初始化占住显存，避免同一张卡被重复分配
}

for i in $(seq 1 576); do  # 576 × 5min = 48h
  # 清理已退出的记录
  tmp=""
  while read -r v pid g; do
    [ -n "$pid" ] && kill -0 $pid 2>/dev/null && tmp="${tmp}${v} ${pid} ${g}\n"
  done < $STATE
  printf "$tmp" > $STATE

  for v in $QUEUE; do
    is_done $v && continue
    state_has_variant $v && continue
    [ "$(attempts $v)" \>= $MAX_ATTEMPTS ] 2>/dev/null && { echo "[$(date '+%F %T')] $v exceeded max attempts, skip"; continue; }
    for g in $(free_gpus); do
      if ! state_has_gpu $g; then
        launch $v $g
        break
      fi
    done
  done

  all_done=1
  for v in $QUEUE; do is_done $v || all_done=0; done
  [ "$all_done" = "1" ] && { echo "[$(date '+%F %T')] all variants done"; exit 0; }
  sleep 300
done
echo "[$(date '+%F %T')] timeout after 48h"
exit 3
