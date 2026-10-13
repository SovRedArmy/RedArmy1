"""汇总复现结果：解析训练日志中的 PSNR/SSIM（train/k400/ucf101）并对比论文 Table 2。"""
import re, glob, sys

LOGS = [
    '/remote-home/gdzj21/VideoCompression/INR/FastNeRV/outputs/train_full_1gpu.log',
    '/remote-home/gdzj21/VideoCompression/INR/FastNeRV/outputs/train_resume2.log',
]

# 论文 Table 2 (8 帧, 150 epoch): Train 25.8 / K400 25.8 / UCF101 25.2 PSNR; SSIM 0.732/0.727/0.723
PAPER = {'train': (25.8, 0.732), 'k400': (25.8, 0.727), 'ucf101': (25.2, 0.723)}

rows = []
for log in LOGS:
    try:
        text = open(log, errors='ignore').read()
    except FileNotFoundError:
        continue
    for m in re.finditer(
            r'\[([-\d :]+)\] Epoch (\d+), (train|eval k400|eval ucf101): '
            r'.*?psnr_pred=([\d.]+) ssim_pred=([\d.]+)', text):
        rows.append((m.group(1), int(m.group(2)), m.group(3).replace('eval ', ''),
                     float(m.group(4)), float(m.group(5))))

# 每个 split 取最大 epoch 的记录（同一 epoch 多条时取最后一条）
latest = {}
for t, ep, split, psnr, ssim in rows:
    key = split
    if key not in latest or (ep, t) >= (latest[key][0], latest[key][1]):
        latest[key] = (ep, t, psnr, ssim)

print(f"{'split':<8}{'epoch':>6}{'PSNR':>8}{'SSIM':>8}{'   论文PSNR':>10}{'ΔPSNR':>8}")
for split in ['train', 'k400', 'ucf101']:
    if split in latest:
        ep, t, psnr, ssim = latest[split]
        pp, ps = PAPER[split]
        print(f"{split:<8}{ep:>6}{psnr:>8.2f}{ssim:>8.3f}{pp:>10.1f}{psnr-pp:>+8.2f}")

# 轨迹（每 10 epoch 的 train PSNR）
traj = {}
for t, ep, split, psnr, ssim in rows:
    if split == 'train' and ep % 10 == 0:
        traj[ep] = psnr
print('\ntrain PSNR 轨迹:', {k: traj[k] for k in sorted(traj)})
