# ==========================================================================================
# make_tables.py —— 从 runs/ 生成最终报告所需的全部表格(markdown)
#   1. 主实验 RD 点表(bunny + UVG, base vs ours)
#   2. 每序列 BD-rate / 平均 BD-rate
#   3. 消融表(bunny: A1..A10 vs base)
#   4. 多 clip 明细(I/P 每段 psnr/bits)
# 输出: tables.md
# ==========================================================================================
import glob
import json
import os
from collections import defaultdict

from bd_rate import bd_psnr, bd_rate


def load(runs='runs'):
    d = defaultdict(dict)     # video -> tag -> results
    for p in glob.glob(os.path.join(runs, '*', 'results.json')):
        r = json.load(open(p))
        d[r['video']][r['tag']] = r
    return d


def pts(vid_tags, method_prefix, presets=('S', 'M', 'L')):
    out = []
    for p in presets:
        t = f'{method_prefix}_{p}'
        if t in vid_tags:
            out.append((vid_tags[t]['psnr'], vid_tags[t]['bpp']))
    return sorted(out, key=lambda x: x[1])


def main():
    data = load()
    lines = []

    # ---- 1. 主实验 RD 点 ----
    lines.append('## 1. Rate points (PSNR dB @ bpp)\n')
    lines.append('| video | method | S | M | L |')
    lines.append('|---|---|---|---|---|')
    for v in sorted(data):
        for m in ['base', 'ours']:
            ps = pts(data[v], m)
            if ps:
                cells = ' | '.join(f'{p:.2f}@{b*1000:.2f}e-3' for p, b in ps)
                lines.append(f'| {v} | {m} | {cells} |')

    # ---- 2. BD-rate ----
    lines.append('\n## 2. BD-rate vs base (negative = savings)\n')
    methods = ['ours', 'ourspar', 'basedense'] + sorted(
        {t.rsplit('_', 1)[0] for v in data for t in data[v]
         if t.rsplit('_', 1)[0].startswith('A')})
    agg = defaultdict(list)
    lines.append('| video | ' + ' | '.join(methods) + ' |')
    lines.append('|---' * (len(methods) + 1) + '|')
    for v in sorted(data):
        row = [v]
        ap = pts(data[v], 'base')
        for m in methods:
            tp = pts(data[v], m)
            if len(ap) >= 3 and len(tp) >= 3:
                try:
                    br = bd_rate(ap, tp)
                    agg[m].append(br)
                    row.append(f'{br:+.1f}%')
                    continue
                except Exception:
                    pass
            row.append('-')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('| **avg** | ' + ' | '.join(
        (f'**{sum(a)/len(a):+.1f}%**' if a else '-') for m in methods) + ' |')

    # ---- 3. 消融明细(bunny) ----
    lines.append('\n## 3. Ablations on bunny (per rate point)\n')
    lines.append('| tag | PSNR | bpp | msssim | fps |')
    lines.append('|---|---|---|---|---|')
    for t in sorted(data.get('bunny', {})):
        r = data['bunny'][t]
        lines.append(f"| {t} | {r['psnr']:.2f} | {r['bpp']*1000:.2f}e-3 | "
                     f"{r['msssim']:.4f} | {r.get('fps') or 0:.0f} |")

    # ---- 4. 多 clip 明细(ours) ----
    lines.append('\n## 4. Multi-clip breakdown (ours)\n')
    for v in sorted(data):
        for t in sorted(data[v]):
            r = data[v][t]
            if r.get('K', 1) > 1:
                lines.append(f"**{v}_{t}** (K={r['K']}, bounds={r['bounds']}):")
                for c in r['clips']:
                    lines.append(f"  - clip{c['k']} [{c['lo']}:{c['hi']}] {c['role']}: "
                                 f"{c['psnr']:.2f}dB @ {c['bits']/1e6:.2f}Mb")

    out = '\n'.join(lines)
    open('tables.md', 'w').write(out + '\n')
    print(out)


if __name__ == '__main__':
    main()
