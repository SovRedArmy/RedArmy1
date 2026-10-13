#!/usr/bin/env python
"""
消融实验结果分析：汇总各变体的 RD 点、训练曲线与结构信息，
计算相对基线（A0）的 BD-rate / BD-PSNR，输出 markdown 报告。

用法：python scripts/analyze_ablation.py [--root output/ablation] [--out report.md] [--ref A0-baseline]
"""
import os
import re
import sys
import glob
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_summary(exp_dir):
    """读取 summary.yaml（两级管线）或 results/single_summary.txt（单模型）。"""
    yml = os.path.join(exp_dir, 'summary.yaml')
    txt = os.path.join(exp_dir, 'results', 'single_summary.txt')
    if os.path.isfile(yml):
        with open(yml) as f:
            return yaml.safe_load(f), 'two-level'
    if os.path.isfile(txt):
        pts = []
        with open(txt) as f:
            for line in f:
                m = re.match(r'Q(\d+): bpp=([\d.]+) bytes=(\d+)(.*)', line.strip())
                if m:
                    metas = dict(re.findall(r"(\w+)=([\d.]+)", m.group(4)))
                    pts.append({'q': int(m.group(1)), 'bpp': float(m.group(2)),
                                'bytes': int(m.group(3)), 'metrics': {k: float(v) for k, v in metas.items()}})
        return {'quant': {str(p['q']): p for p in pts}}, 'single'
    return None, None


def bd_rate(r1, p1, r2, p2):
    """经典 BD-rate（%），正 = 第二组更费码率。"""
    from scripts.bd_rate import bd_rate as _br, bd_psnr as _bp
    return _br(np.array(r1), np.array(p1), np.array(r2), np.array(p2)), _bp(np.array(r1), np.array(p1), np.array(r2), np.array(p2))


def read_log_curves(log_path):
    """从训练日志提取 (epoch, psnr) 轨迹与最终 epoch。"""
    if not os.path.isfile(log_path):
        return []
    curves = []
    pat = re.compile(r'Train(?: \((\w+)\))? - Epoch (\d+) \[\d+/\d+\].*?psnr: ([\d.]+)')
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                curves.append((m.group(1) or 'main', int(m.group(2)), float(m.group(3))))
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='output/ablation')
    ap.add_argument('--out', default='output/ablation/REPORT.md')
    ap.add_argument('--ref', default='A0-baseline')
    args = ap.parse_args()

    exps = sorted([d for d in glob.glob(os.path.join(args.root, 'A*-*')) if os.path.isdir(d)])
    if not exps:
        print('no experiments found')
        return

    # 参考基线的 RD 点
    ref_data, _ = load_summary(os.path.join(args.root, args.ref))
    ref_pts = []
    if ref_data:
        for q, r in sorted(ref_data.get('quant', {}).items(), key=lambda kv: float(kv[0])):
            if 'psnr' in r.get('metrics', {}):
                ref_pts.append((r['bpp'], r['metrics']['psnr']))

    lines = ['# MyHiNeRV 消融实验报告', '',
             f'- 数据集根目录：`{args.root}`',
             f'- 基线：`{args.ref}`（BD-rate 正值 = 比基线费码率，负值 = 更好）', '']

    lines += ['## RD 点汇总', '',
              '| 变体 | 量化级 | bpp | PSNR | MS-SSIM | 总字节 | base 字节 | GOP 数 |',
              '|---|---|---|---|---|---|---|---|']
    for d in exps:
        name = os.path.basename(d)
        data, kind = load_summary(d)
        if data is None:
            lines.append(f'| {name} | （未完成） | | | | | | |')
            continue
        n_gop = len(data.get('clips', [])) if kind == 'two-level' else 1
        for q, r in sorted(data.get('quant', {}).items(), key=lambda kv: float(kv[0])):
            met = r.get('metrics', {})
            lines.append(f"| {name} | Q{q} | {r['bpp']:.4f} | {met.get('psnr', float('nan')):.3f} | "
                         f"{met.get('ms-ssim', float('nan')):.4f} | {r.get('bytes', 0)} | "
                         f"{r.get('base_bytes', '-')} | {n_gop} |")
    lines.append('')

    # BD-rate vs 基线（要求 >=3 个重叠 RD 点）
    lines += ['## 相对基线的 BD-rate', '', '| 变体 | RD 点数 | BD-rate | BD-PSNR |', '|---|---|---|---|']
    for d in exps:
        name = os.path.basename(d)
        data, _ = load_summary(d)
        if data is None or name == args.ref:
            continue
        pts = [(r['bpp'], r['metrics']['psnr']) for q, r in sorted(data.get('quant', {}).items(), key=lambda kv: float(kv[0]))
               if 'psnr' in r.get('metrics', {})]
        if len(pts) >= 3 and len(ref_pts) >= 3:
            try:
                br, bp = bd_rate([p[0] for p in ref_pts], [p[1] for p in ref_pts],
                                 [p[0] for p in pts], [p[1] for p in pts])
                lines.append(f'| {name} | {len(pts)} | {br:+.2f}% | {bp:+.4f} dB |')
                continue
            except Exception as e:
                lines.append(f'| {name} | {len(pts)} | 失败({e}) | |')
                continue
        lines.append(f'| {name} | {len(pts)} | （点数不足） | |')
    lines.append('')

    # GOP 结构与训练进度
    lines += ['## 结构与进度', '']
    for d in exps:
        name = os.path.basename(d)
        data, kind = load_summary(d)
        log = os.path.join(args.root, 'logs', f'{name}.log')
        curves = read_log_curves(log)
        last = curves[-1] if curves else None
        clips = data.get('clips') if data else None
        info = f"- **{name}**（{kind}）："
        if clips:
            info += f"{len(clips)} 个 GOP，边界 {clips}"
        if last:
            info += f"；日志最后：{last[0]} epoch {last[1]}，PSNR {last[2]:.2f}"
        if data and last is None:
            info += '；已完成'
        lines.append(info)
    lines.append('')

    report = '\n'.join(lines)
    with open(args.out, 'w') as f:
        f.write(report)
    print(report)
    print(f'\n报告已写入 {args.out}')


if __name__ == '__main__':
    main()
