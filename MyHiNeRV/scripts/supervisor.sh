#!/bin/bash
# MyHiNeRV 自愈型实验监督器：
#   - 每 2 分钟巡检一次：进程存活 + 日志推进（连续 7 次无更新≈14 分钟视为停滞）
#   - 崩溃/停滞 → 自动清理并重启（断点续跑：已完成的 base/GOP 自动跳过）
#   - GPU 空闲 → 按优先级启动下一个任务（每卡一任务）
#   - 事件写入 INCIDENTS.log；每次巡检刷新 STATUS.txt 并打印状态板（tmux 可实时观看）
# 运行：tmux new-session -d -s myhinerbv 'bash scripts/supervisor.sh'
#   观看：tmux attach -t myhinerbv（Ctrl-b d 退出观看，不影响运行）
cd /remote-home/gdzj21/VideoCompression/MyHiNeRV
export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
PY=/remote-home/gdzj21/miniconda3/envs/NVRC/bin/python
ROOT=output/ablation
mkdir -p /dev/shm/myhinerbv_mp 2>/dev/null; export TMPDIR=/dev/shm/myhinerbv_mp   # 本地 shm：NFS 上 pymp 清理会 EBUSY 崩溃
export TMUX_TMPDIR=/remote-home/gdzj21/tmp/myhinerbv_tmp  # tmux socket 不再依赖 /tmp
mkdir -p "$TMPDIR"
STATE=/remote-home/gdzj21/tmp/myhinerbv_queue.state
ATTEMPTS=/remote-home/gdzj21/tmp/myhinerbv_queue.attempts
STALL=/remote-home/gdzj21/tmp/myhinerbv_queue.stall
INCIDENTS=$ROOT/INCIDENTS.log
STATUS=$ROOT/STATUS.txt
QUEUE="A1 B0 B1 B2 B3 B4 B0r B1r B2r B3r B4r C1 C2 C3 A1v2"
FREE_MB=${FREE_MB:-5000}       # 视为"空闲卡"的显存阈值（共卡模式调高，如 18000）
MAX_JOBS=${MAX_JOBS:-6}        # 最多同时运行的任务数
MIN_RAM_GB=${MIN_RAM_GB:-18}   # 启动新任务要求的系统可用内存（GB），防止 OOM 连环杀
MAX_ATTEMPTS=4
STALL_LIMIT=7        # 连续无更新巡检次数（×2min ≈ 14 分钟）
POLL=120

mkdir -p "$ROOT"
touch $STATE $ATTEMPTS $STALL $INCIDENTS

# 单实例锁：防止多个 supervisor 实例并发写状态文件产生竞态
exec 9>/remote-home/gdzj21/tmp/myhinerbv_supervisor_v2.lock
flock -n 9 || { echo "another supervisor running, exit"; exit 0; }

# 任务运行环境：多进程/DataLoader 临时目录走本地 /dev/shm（NFS 上 pymp 目录清理会
# 因 silly-rename 报 EBUSY 杀死主进程——2026-09-22 C1 连环崩溃根因）；
# torch 扩展/编译缓存与 tmux socket 仍走 NFS（根分区长期 93% 满）。
export TMPDIR=/dev/shm/myhinerbv_mp
export TORCH_EXTENSIONS_DIR=/remote-home/gdzj21/tmp/torch_ext
export TORCHINDUCTOR_CACHE_DIR=/remote-home/gdzj21/tmp/inductor
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $TMPDIR $TORCH_EXTENSIONS_DIR $TORCHINDUCTOR_CACHE_DIR

log_incident() { echo "[$(date '+%F %T')] $*" >> $INCIDENTS; }

variant_dir()  { ls -d $ROOT/$1-* 2>/dev/null | grep -v oldsize | head -1; }
variant_name() { basename "$(variant_dir $1)" 2>/dev/null; }

is_done() {
  local d=$(variant_dir $1)
  [ -n "$d" ] && ls $d/summary.txt $d/summary.yaml $d/results/single_summary.txt 2>/dev/null | grep -q .
}
has_variant() { awk -v v="$1" '$1==v{f=1} END{exit !f}' $STATE; }
has_gpu()     { awk -v g="$1" '$3==g{f=1} END{exit !f}' $STATE; }
attempts()    { awk -v v="$1" '$1==v{if ($2+0>m) m=$2+0} END{print m+0}' $ATTEMPTS; }

last_log() {  # 变体最近一行训练日志（截断）
  local d=$(variant_dir $1)
  [ -n "$d" ] && [ -f "$ROOT/logs/$(basename $d).log" ] && \
    tail -1 "$ROOT/logs/$(basename $d).log" | cut -c1-90
}

free_gpus() {  # 空闲显存 ≥ FREE_MB 的卡
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' -v t="$FREE_MB" '$3 - $2 >= t {print $1}' | sort -t, -k1n
}
n_running() { wc -l < $STATE; }
ram_ok() { [ "$(free -g | awk '/^Mem:/{print $7}')" -ge "$MIN_RAM_GB" ]; }
last_hold=0

launch() {  # launch <variant> <gpu> <reason>
  local v=$1 g=$2 reason=$3
  local n=$(($(attempts $v) + 1))
  log_incident "LAUNCH ${v} on GPU${g} (attempt ${n}; ${reason})"
  GPU=$g QUICK=1 SEQ=ReadySetGo PYTHON=$PY setsid nohup bash 9>&- \
    /remote-home/gdzj21/VideoCompression/MyHiNeRV/scripts/run_ablation.sh $v \
    >> "$ROOT/logs/${v}.supervisor.log" 2>&1 &
  local pid=$!   # setsid: pid 即进程组组长，kill -- -pid 可整组消灭（含 python/tee）
  echo "$v $pid $g" >> $STATE
  awk -v v="$v" '$1!=v' $ATTEMPTS > $ATTEMPTS.tmp && mv $ATTEMPTS.tmp $ATTEMPTS
  echo "$v $n" >> $ATTEMPTS
  awk -v v="$v" '$1!=v' $STALL > $STALL.tmp && mv $STALL.tmp $STALL
  echo "$v 0" >> $STALL
  sleep 150   # 等初始化占住显存
}

render_status() {
  local gpus=$(timeout 30 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  {
    echo "==== MyHiNeRV 实验监督器  $(date '+%F %T')  (刷新 ${POLL}s) ===="
    echo "GPU(卡号:显存MB:利用率%): $gpus"
    echo "------------------------------------------------------------------"
    printf "%-6s %-9s %-4s %-7s %s\n" "任务" "状态" "尝试" "GPU" "最近日志"
    for v in $QUEUE; do
      local d=$(variant_dir $v)
      local st="WAIT"
      local gpu="-"
      if is_done $v; then st="DONE"
      elif has_variant $v; then
        gpu=$(awk -v v="$v" '$1==v{print $3}' $STATE)
        st="RUN(gpu${gpu})"
      elif [ "$(attempts $v)" \>= $MAX_ATTEMPTS ] 2>/dev/null; then st="FAILED"
      fi
      printf "%-6s %-9s %-4s %-7s %s\n" "$v" "$st" "$(attempts $v)" "$gpu" "$(last_log $v)"
    done
    echo "------------------------------------------------------------------"
    echo "事件（最近 6 条）："; tail -6 $INCIDENTS 2>/dev/null
    echo "观看: tmux attach -t myhinerbv | 事件全记录: $INCIDENTS"
  } > $STATUS
  clear
  cat $STATUS
}

for cycle in $(seq 1 7200); do   # 7200×2min = 10 天
  touch $STATE $ATTEMPTS $STALL   # 状态文件可能被外部清理，逐轮补建
  # 1) 清理已退出的运行记录；记录仍在跑的任务的日志推进情况
  tmp=""
  while read -r v pid g; do
    [ -n "$pid" ] || continue
    if kill -0 $pid 2>/dev/null; then
      tmp="${tmp}${v} ${pid} ${g}\n"
    else
      log_incident "EXIT ${v} (pid ${pid} on GPU${g} 已退出)"
      awk -v v="$v" '$1!=v' $STALL > $STALL.tmp && mv $STALL.tmp $STALL
    fi
    # 兜底：该组若已死但仍有组内残留进程（极端竞态），按 PGID 清扫
    kill -0 $pid 2>/dev/null || kill -9 -- -$pid 2>/dev/null
  done < $STATE
  printf "$tmp" > $STATE

  # 2) 停滞检测：运行中的任务日志连续 STALL_LIMIT 次未更新 → 杀掉重启
  while read -r v pid g; do
    [ -n "$pid" ] || continue
    d=$(variant_dir $v); logf="$ROOT/logs/$(basename $d).log"
    cur=0
    [ -f "$logf" ] && cur=$(stat -c %Y "$logf")
    cpu=$(ps -o times= -p $pid 2>/dev/null | tr -d ' '); [ -z "$cpu" ] && cpu=0
    prev_log=$(awk -v v="$v" '$1==v{print $2}' $STALL); [ -z "$prev_log" ] && prev_log=0
    prev_cpu=$(awk -v v="$v" '$1==v{print $4}' $STALL); [ -z "$prev_cpu" ] && prev_cpu=0
    # 真停滞 = 日志未推进 且 进程 CPU 时间未增长（共卡下首个日志打印可能 >14min，CPU 在涨就不算停滞）
    if [ "$cur" = "$prev_log" ] && [ "$cur" != "0" ] && [ "$cpu" -le "$prev_cpu" ]; then
      n=$(awk -v v="$v" '$1==v{print $3+1}' $STALL); [ -z "$n" ] && n=1
      awk -v v="$v" -v c="$cur" -v n="$n" -v t="$cpu" '$1!=v; $1==v{print $1, c, n, t}' $STALL > $STALL.tmp && mv $STALL.tmp $STALL
      if [ "$n" -ge $STALL_LIMIT ]; then
        log_incident "STALL ${v} 日志与CPU ${n} 次巡检均未推进 → kill & restart"
        kill -9 $pid 2>/dev/null
        pkill -9 -P $pid 2>/dev/null
        for cpid in $(ps aux | grep "main.py --dataset" | grep -v grep | awk '{print $2}'); do
          ps -o ppid= -p $cpid | grep -qw "$pid" && kill -9 $cpid 2>/dev/null
        done
      fi
    else
      awk -v v="$v" -v c="$cur" -v t="$cpu" '$1!=v; $1==v{print $1, c, 0, t}' $STALL > $STALL.tmp && mv $STALL.tmp $STALL
    fi
  done < $STATE

  # 2.5) 孤儿清扫：任何不属于存活组的 myhinerbv python（PPID=1 或组不在状态表）直接清除
  for opid in $(ps -o pid=,ppid=,cmd= -C python 2>/dev/null | grep "myhinerbv_mai[n]" | awk '{print $1}'); do
    opgid=$(ps -o pgid= -p $opid | tr -d ' ')
    keep=0
    while read -r v pid g; do [ "$pid" = "$opgid" ] && keep=1; done < $STATE
    if [ "$keep" = "0" ]; then
      log_incident "SWEEP orphan python pid=${opid} (pgid=${opgid}) killed"
      kill -9 $opid 2>/dev/null
    fi
  done

  # 3) GPU 空闲 → 按队列启动
  for v in $QUEUE; do
    is_done $v && continue
    has_variant $v && continue
    (( $(attempts $v) >= MAX_ATTEMPTS )) && continue
    (( $(n_running) >= MAX_JOBS )) && break
    ram_ok || {
      now=$(date +%s)
      if [ $((now - last_hold)) -gt 1800 ]; then log_incident "HOLD: 系统可用内存 < ${MIN_RAM_GB}GB，暂缓启动新任务"; last_hold=$now; fi
      break
    }
    for g in $(free_gpus); do
      if ! has_gpu $g; then
        if [ ! -d "$(variant_dir $v)" ] || [ -z "$(variant_dir $v)" ]; then
          launch $v $g "new"
        else
          launch $v $g "resume"
        fi
        break
      fi
    done
  done

  render_status

  all_done=1
  for v in $QUEUE; do is_done $v || all_done=0; done
  [ "$all_done" = "1" ] && { log_incident "ALL DONE"; exit 0; }
  sleep $POLL
done
