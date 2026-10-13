#!/bin/bash
# 看护脚本 v3：管理 4 个 HiNeRV 复现任务（Kimono1 已完成，不在此列）
#   - 任务未运行且内存充足时启动；崩溃后自动从最新 checkpoint 续训
#   - 每个任务的 stdout 追加到 <输出根目录>/run.log；出现 "Training completed" 后不再重启
#   - 全部完成后自动退出
REP=/remote-home/gdzj21/VideoCompression/INR/HiNeRV/repro

avail_gb() { free -g | awk '/^Mem:/{print $7}'; }
latest_ckpt() { ls -dt "$1"/*/checkpoints/checkpoint_[0-9]* 2>/dev/null | head -1; }
job_running() { pgrep -f "dataset-name $1" > /dev/null 2>&1; }
job_done() { grep -q "Training completed" "$1/run.log" 2>/dev/null; }

# 参数: $1=数据集名(pgrep用) $2=启动脚本 $3=运行日志(当前实例的stdout) $4=checkpoint根目录 $5=最少可用内存GB
# 注: BasketballDrive/Cactus/ParkScene 三序列因脚本生成时的sed问题, 输出目录带分辨率后缀,
#     与初始 run.log 目录不同, 故分开指定; 完成标记在任一日志中出现即视为完成。
watch_one() {
  local NAME=$1 SCRIPT=$2 RUNLOG=$3 ROOT=$4 MINMEM=$5
  (grep -q "Training completed" "$RUNLOG" "$ROOT/run.log" 2>/dev/null) && return 0
  job_running "$NAME" && return 0

  local MEM; MEM=$(avail_gb)
  if [ "$MEM" -lt "$MINMEM" ]; then
    echo "[$(date '+%F %T')] $NAME 未运行但内存不足 (${MEM}GB < ${MINMEM}GB)，等待"
    return 0
  fi

  local CKPT; CKPT=$(latest_ckpt "$ROOT")
  mkdir -p "$ROOT"   # 输出目录可能尚未创建（首次运行时 run.log 重定向需要）
  if [ -n "$CKPT" ]; then
    echo "[$(date '+%F %T')] $NAME 从 $CKPT 续训 (可用内存 ${MEM}GB)"
    nohup bash "$SCRIPT" "$CKPT" >> "$ROOT/run.log" 2>&1 &
  else
    echo "[$(date '+%F %T')] $NAME 全新启动 (可用内存 ${MEM}GB)"
    nohup bash "$SCRIPT" >> "$ROOT/run.log" 2>&1 &
  fi
  sleep 45   # 启动间隔，等内存占用稳定后再考虑下一个任务
}

while true; do
  watch_one "BQTerrace_1920x1080_60"       "$REP/run_bqterr_ace.sh"  "$REP/BQTerrace/run.log"              "$REP/BQTerrace"                 30
  watch_one "BasketballDrive_1920x1080_50"  "$REP/run_BasketballDrive.sh" "$REP/BasketballDrive/run.log"  "$REP/BasketballDrive_1920x1080_50" 30
  watch_one "Cactus_1920x1080_50"          "$REP/run_Cactus.sh"      "$REP/Cactus/run.log"                 "$REP/Cactus_1920x1080_50"       30
  watch_one "ParkScene_1920x1080_24"       "$REP/run_ParkScene.sh"   "$REP/ParkScene/run.log"              "$REP/ParkScene_1920x1080_24"    20

  if (grep -q "Training completed" "$REP/BQTerrace/run.log" 2>/dev/null) && \
     (grep -q "Training completed" "$REP/BasketballDrive/run.log" "$REP/BasketballDrive_1920x1080_50/run.log" 2>/dev/null) && \
     (grep -q "Training completed" "$REP/Cactus/run.log" "$REP/Cactus_1920x1080_50/run.log" 2>/dev/null) && \
     (grep -q "Training completed" "$REP/ParkScene/run.log" 2>/dev/null); then
    echo "[$(date '+%F %T')] 全部任务完成，看护退出"
    break
  fi
  sleep 120
done
