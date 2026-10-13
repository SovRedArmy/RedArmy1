# ==========================================================================================
# 简单 GPU 任务调度器: 从 jobs 文件逐行读取 shell 命令, 轮询分派到指定 GPU 列表,
# 每个 GPU 同时最多 jobs_per_gpu 个任务(与他人任务共享 GPU, 控制并发与显存)。
# 用法: python scheduler.py jobs.txt --gpus 0,1,2,3,4,5,6,7 --jpu 2 --log-dir logs/jobs
# jobs 文件支持 "# 注释" 行; 命令中的 %GPU% 占位符会被替换为实际 GPU 号。
# ==========================================================================================
import argparse
import itertools
import os
import subprocess
import time


def gpu_free_mb(gpu):
    """查询某 GPU 的空闲显存(MB), 失败返回大值。"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--id=%d' % gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return 1 << 30


def ram_avail_gb():
    """系统可用内存(GB) — 低于阈值时暂缓发新任务, 避免成为 OOM-killer 目标。"""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) / 1048576
    except Exception:
        pass
    return 1 << 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs_file')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--jpu', type=int, default=2, help='并发任务数/GPU')
    ap.add_argument('--log-dir', default='logs/jobs')
    ap.add_argument('--poll', type=float, default=20)
    ap.add_argument('--min-ram-gb', type=float, default=18, help='系统可用内存低于此值时暂停发新任务')
    ap.add_argument('--retries', type=int, default=2, help='每个任务失败后最多重试次数(训练支持断点续跑)')
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(',')]
    jobs = []
    with open(args.jobs_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                jobs.append(line)

    # 跳过已完成的作业(按输出目录 results.json 判断)
    import re
    skipped = 0
    todo = []
    for cmd in jobs:
        m = re.search(r'--data (\S+).*?--tag (\S+)', cmd)
        outf = re.search(r'--outf (\S+)', cmd)
        if m and outf and os.path.isfile(os.path.join(outf.group(1),
                                       f"{os.path.basename(m.group(1))}_{m.group(2)}", 'results.json')):
            skipped += 1
        else:
            todo.append(cmd)
    jobs = todo
    print(f'skipped {skipped} finished jobs; {len(jobs)} to run')

    slots = {g: args.jpu for g in gpus}       # 每 GPU 剩余槽位
    running = []                              # (proc, gpu, cmd, logfile)
    queue = [(idx, cmd, 0) for idx, cmd in enumerate(jobs)]   # (idx, cmd, 已重试次数)
    done, failed = [], []
    print(f'{len(jobs)} jobs on GPUs {gpus} ({args.jpu}/gpu)')

    while queue or running:
        # 回收完成的任务(失败且还有重试额度 → 重新入队; 训练支持断点续跑, 重跑代价低)
        for r in list(running):
            proc, g, (idx, cmd, tries), lf = r
            if proc.poll() is not None:
                running.remove(r)
                slots[g] += 1
                if proc.returncode == 0:
                    done.append((cmd, 0))
                    print(f'[done {len(done)}/{len(jobs)}] {cmd[:90]}', flush=True)
                elif tries < args.retries:
                    queue.append((idx, cmd, tries + 1))
                    print(f'[retry {tries + 1}/{args.retries}] rc={proc.returncode} {cmd[:90]}', flush=True)
                else:
                    failed.append((cmd, proc.returncode))
                    print(f'[FAILED rc={proc.returncode}] {cmd[:90]}', flush=True)
        # 分派新任务(显存不足 10GB 的 GPU 暂不分派; 系统可用内存不足也暂缓, 避免被 OOM-killer 击杀)
        while queue and any(v > 0 for v in slots.values()):
            if ram_avail_gb() < args.min_ram_gb:
                break
            avail = [g for g in gpus if slots[g] > 0 and gpu_free_mb(g) > 10000]
            if not avail:
                break
            g = avail[0]
            idx, cmd, tries = queue.pop(0)
            real = cmd.replace('%GPU%', str(g))
            lf = os.path.join(args.log_dir, f'job{idx:03d}' + (f'_r{tries}' if tries else '') + '.log')
            env = dict(os.environ)
            env.setdefault('TMPDIR', os.path.abspath('./tmp'))   # /tmp 可能被别人占满, 用本地 tmp
            os.makedirs(env['TMPDIR'], exist_ok=True)
            with open(lf, 'w') as logf:
                proc = subprocess.Popen(real, shell=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
            slots[g] -= 1
            running.append((proc, g, (idx, cmd, tries), lf))
            print(f'[start gpu{g}] {cmd[:90]}', flush=True)
        time.sleep(args.poll)

    print(f'ALL DONE: {len(done)} ok, {len(failed)} failed')
    for cmd, rc in failed:
        print(f'  FAILED rc={rc}: {cmd}')


if __name__ == '__main__':
    main()
