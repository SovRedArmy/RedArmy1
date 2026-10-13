# ==========================================================================================
# v3 任务看门狗: 每 5 分钟
#   1. 记录 3 个任务的最新进度到 watch_status.txt
#   2. 若调度器进程死掉且任务未全部完成 → 自动重启调度器(其自带跳过已完成 + 断点续跑)
#   3. 若系统可用内存 < 12GB, 不做任何事(调度器自身的 RAM 闸门会管新任务)
# 用法: nohup python watch_v3.py > logs/watch_v3.log 2>&1 &
# ==========================================================================================
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_FILE = os.path.join(ROOT, 'jobs/v3_wave3.txt')
LOG_DIR = os.path.join(ROOT, 'logs/v3wave3')
STATUS = os.path.join(ROOT, 'watch_status.txt')
SCHED_LOG = os.path.join(ROOT, 'logs/v3wave3_sched.log')
PY = '/remote-home/gdzj21/envs/nerv/bin/python'


def jobs_done():
    """按 results.json 判断 jobs 文件里的任务是否全部完成。"""
    done = 0
    total = 0
    with open(JOBS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            total += 1
            m = re.search(r'--data (\S+).*?--tag (\S+)', line)
            o = re.search(r'--outf (\S+)', line)
            if m and o:
                rp = os.path.join(ROOT, o.group(1),
                                  f"{os.path.basename(m.group(1))}_{m.group(2)}", 'results.json')
                if os.path.isfile(rp):
                    done += 1
    return done, total


def sched_alive():
    try:
        out = subprocess.check_output(['pgrep', '-f', 'scheduler.py jobs/v3_wave3.txt'])
        return bool(out.decode().strip())
    except subprocess.CalledProcessError:
        return False


def last_progress():
    """抓每个 job log 的最新 epoch 行。"""
    parts = []
    for i in range(3):
        lf = os.path.join(LOG_DIR, f'job{i:03d}.log')
        if not os.path.isfile(lf):
            parts.append(f'job{i:03d}:无日志')
            continue
        try:
            out = subprocess.run(['tail', '-3', lf], capture_output=True, text=True, timeout=10).stdout
            m = re.findall(r'ep(\d+)/(\d+)', out)
            if m:
                parts.append(f'job{i:03d}:ep{m[-1][0]}/{m[-1][1]}')
            elif 'results' in out or 'psnr' in out.lower():
                parts.append(f'job{i:03d}:评估中')
            else:
                parts.append(f'job{i:03d}:?')
        except Exception as e:
            parts.append(f'job{i:03d}:err')
    return ' '.join(parts)


while True:
    try:
        done, total = jobs_done()
        sched = sched_alive()
        prog = last_progress() if done < total else 'ALL_DONE'
        with open(STATUS, 'a') as f:
            f.write(f'{time.strftime("%H:%M:%S")} done={done}/{total} sched={1 if sched else 0} {prog}\n')
        if done < total and not sched:
            with open(SCHED_LOG, 'a') as f:
                f.write(f'\n[watchdog {time.strftime("%F %T")}] 调度器已死, 重启\n')
            env = dict(os.environ, TMPDIR=os.path.join(ROOT, 'tmp'))
            subprocess.Popen(
                [PY, os.path.join(ROOT, 'scheduler.py'), JOBS_FILE,
                 '--gpus', '0,1,2,3,4,5,6,7', '--jpu', '1',
                 '--min-ram-gb', '18', '--retries', '3', '--log-dir', LOG_DIR],
                stdout=open(SCHED_LOG, 'a'), stderr=subprocess.STDOUT,
                cwd=ROOT, env=env, start_new_session=True)
    except Exception as e:
        with open(STATUS, 'a') as f:
            f.write(f'{time.strftime("%H:%M:%S")} watchdog error: {e}\n')
    time.sleep(300)
