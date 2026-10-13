#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 NeRV 复现实验结果: 从 output/repro_classb 的日志和 eval.txt 解析
PSNR / MS-SSIM / FPS / 熵编码效率, 并计算 bpp 与 MPix 参数量."""
import os, re, glob, json

ROOT = '/remote-home/gdzj21/VideoCompression/INR/NeRV'
OUT = os.path.join(ROOT, 'output', 'repro_classb')
FRAMES = {'kimono': 240, 'parkscene': 240, 'basketballdrive': 500, 'cactus': 500, 'bqterrace': 600}
SIZE = {'9_16_26': 'NeRV-S', '9_16_58': 'NeRV-M', '9_16_112': 'NeRV-L'}
W, H = 1920, 1080

def parse_train_log(logfile):
    """从训练日志提取参数量与最终 PSNR"""
    if not os.path.exists(logfile): return None
    txt = open(logfile, errors='ignore').read()
    params = re.findall(r'Model Params: ([\d.]+)M', txt)
    psnr = re.findall(r'Eval best_PSNR at epoch(\d+):.*?best: ([\d.]+)\s+best_msssim: ([\d.]+)', txt)
    m = re.findall(r'1080p: current: ([\d.]+)\s+best: ([\d.]+)\s+best_msssim: ([\d.]+)', txt)
    return dict(params_m=float(params[0]) if params else None,
                last_psnr=float(m[-1][0]) if m else None,
                best_psnr=float(m[-1][1]) if m else None,
                best_msssim=float(m[-1][2]) if m else None)

def parse_eval(logfile):
    """从 eval 输出提取熵编码效率与量化后 PSNR"""
    if not os.path.exists(logfile): return []
    txt = open(logfile, errors='ignore').read()
    out = []
    for blk in txt.split('Results for checkpoint')[1:]:
        eff = re.findall(r'Entropy encoding efficiency for bit (\d+): ([\d.]+)', blk)
        psnr = re.findall(r'PSNR/ms_ssim on validate set for bit (-?\d+) with axis (-?\d+): ([\d.]+)/(?-1:)([\d.]+)', blk) \
               or re.findall(r'PSNR/ms_ssim on validate set for bit (-?\d+) with axis (-?\d+): ([\d.]+)/([\d.]+)', blk)
        fps = re.findall(r'FPS: ([\d.]+)', blk)
        out.append(dict(eff={int(b): float(v) for b, v in eff},
                        psnr_quant={int(b): float(p) for b, a, p, m in psnr},
                        fps=float(fps[-1]) if fps else None))
    return out

def bpp(params_m, quant_bit, eff):
    """bpp = 参数量 × (1-稀疏度=0) × 平均比特(=bit×效率) / 像素数"""
    avg_bits = quant_bit * eff
    return params_m * 1e6 * avg_bits / (W * H * FRAMES[ds])

if __name__ == '__main__':
    print(f"{'数据集':<15}{'模型':<8}{'参数量':>8}{'PSNR':>8}{'MS-SSIM':>9}")
    for logfile in sorted(glob.glob(os.path.join(ROOT, 'logs_*.log'))):
        name = os.path.basename(logfile)[5:-4]          # kimono_S
        ds, size = name.rsplit('_', 1)
        r = parse_train_log(logfile)
        if not r: continue
        print(f"{ds:<15}{size:<8}{r['params_m']:>7.2f}M{r['best_psnr'] or 0:>8.2f}{r['best_msssim'] or 0:>9.4f}")
