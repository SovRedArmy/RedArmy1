# ==========================================================================================
# 结果汇总: 扫描 runs/<video>_<method>_<preset>/results.json,
#  - 按 视频 × 方法 汇总各码率点 (psnr, bpp)
#  - 计算每个方法相对 base(baseline) 的 BD-rate / BD-PSNR (PCHIP)
# 用法: python collect.py [--runs runs] [--anchor base]
# ==========================================================================================
import argparse
import glob
import json
import os
from collections import defaultdict

from bd_rate import bd_psnr, bd_rate


def load_all(runs_dir):
    """{(video, method, preset): results_dict}"""
    out = {}
    for p in glob.glob(os.path.join(runs_dir, '*', 'results.json')):
        r = json.load(open(p))
        tag = r['tag']
        if '_' not in tag:
            continue
        method, preset = tag.rsplit('_', 1)
        out[(r['video'], method, preset)] = r
    return out


def rate_points(res_list):
    return sorted([(r['psnr'], r['bpp']) for r in res_list], key=lambda p: p[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--anchor', default='base')
    ap.add_argument('--videos', default='')
    args = ap.parse_args()

    all_res = load_all(args.runs)
    vids = defaultdict(lambda: defaultdict(dict))
    for (v, m, p), r in all_res.items():
        vids[v][m][p] = r
    sel = set(args.videos.split(',')) if args.videos else None

    print('=' * 105)
    print('RATE POINTS (psnr dB @ bpp)')
    for v in sorted(vids):
        if sel and v not in sel:
            continue
        for m in sorted(vids[v]):
            ps = vids[v][m]
            s = ' | '.join(f"{p}:{ps[p]['psnr']:.2f}@{ps[p]['bpp']*1000:.2f}e-3"
                           for p in sorted(ps, key=lambda x: ps[x]['bpp']))
            print(f'{v:14s} {m:16s} {s}')

    print('=' * 105)
    print(f'BD-RATE vs {args.anchor} (negative = 省码率)')
    methods = sorted({m for v in vids for m in vids[v]} - {args.anchor})
    per_method = defaultdict(list)
    for v in sorted(vids):
        if sel and v not in sel:
            continue
        a = vids[v].get(args.anchor, {})
        if len(a) < 3:
            print(f'{v:14s} anchor 点数不足({len(a)})')
            continue
        apts = rate_points(list(a.values()))
        for m in methods:
            pts = vids[v].get(m, {})
            if len(pts) < 3:
                continue
            try:
                br = bd_rate(apts, rate_points(list(pts.values())))
                bp = bd_psnr(apts, rate_points(list(pts.values())))
                per_method[m].append(br)
                print(f'{v:14s} {m:16s} BD-rate={br:+7.2f}%   BD-PSNR={bp:+.3f}dB')
            except Exception as e:
                print(f'{v:14s} {m:16s} 计算失败: {e}')

    print('=' * 105)
    print('AVG BD-RATE across videos')
    for m in methods:
        vals = per_method.get(m, [])
        if vals:
            print(f'{m:16s} avg={sum(vals)/len(vals):+7.2f}%  (n={len(vals)}, '
                  f'min={min(vals):+.2f}, max={max(vals):+.2f})')


if __name__ == '__main__':
    main()
