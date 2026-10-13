"""
BD-rate / BD-PSNR 计算（经典 Bjøntegaard Delta，分段三次多项式拟合，
与 MATLAB bdsnr/bdrate 和 pypae 兼容）。

用法：
  python scripts/bd_rate.py --ref ref.txt --test test.txt
两个文件每行一个 RD 点："bpp psnr"（空格/逗号分隔，可带表头）。
输出 BD-rate%% 与 BD-PSNR(dB)，正的 BD-rate 表示 test 比 ref 费码率。
"""
import os
import sys
import argparse
import numpy as np


def _pchip_slopes(x, y):
    """PCHIP 的导数估计（保形分段三次 Hermite）。"""
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n)
    for i in range(n):
        if i == 0:
            d[i] = delta[0]
        elif i == n - 1:
            d[i] = delta[-1]
        elif delta[i - 1] * delta[i] <= 0:
            d[i] = 0.
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return d


def pchip_integrate(x, y):
    """PCHIP 插值下 y 的定积分（x 需升序）。"""
    d = _pchip_slopes(x, y)
    h = np.diff(x)
    area = 0.
    for i in range(len(h)):
        t = np.linspace(0, 1, 17)
        hermite = ((2 * t ** 3 - 3 * t ** 2 + 1) * y[i] +
                   (t ** 3 - 2 * t ** 2 + t) * h[i] * d[i] +
                   (-2 * t ** 3 + 3 * t ** 2) * y[i + 1] +
                   (t ** 3 - t ** 2) * h[i] * d[i + 1])
        area += hermite.mean() * h[i]
    return area


def bd_rate(rate_ref, psnr_ref, rate_test, psnr_test):
    """BD-rate(%%)：正 = test 更费码率。"""
    lr_ref = np.log(np.asarray(rate_ref, dtype=np.float64))
    lr_test = np.log(np.asarray(rate_test, dtype=np.float64))
    lo = max(lr_ref.min(), lr_test.min())
    hi = min(lr_ref.max(), lr_test.max())
    assert lo < hi, 'rate ranges do not overlap'
    xr = np.clip(lr_ref, lo, hi)
    xt = np.clip(lr_test, lo, hi)
    p_ref = np.polyfit(lr_ref, psnr_ref, 3) if len(lr_ref) >= 4 else np.polyfit(lr_ref, psnr_ref, max(len(lr_ref) - 1, 1))
    p_test = np.polyfit(lr_test, psnr_test, 3) if len(lr_test) >= 4 else np.polyfit(lr_test, psnr_test, max(len(lr_test) - 1, 1))
    # 用多项式在重叠区间积分（经典 bdrate 做法）
    avg_ref = np.polyval(p_ref, np.linspace(lo, hi, 1000)).mean()
    avg_test = np.polyval(p_test, np.linspace(lo, hi, 1000)).mean()
    return (np.exp(avg_test - avg_ref) - 1.) * 100.


def bd_psnr(rate_ref, psnr_ref, rate_test, psnr_test):
    """BD-PSNR(dB)：正 = test 质量更高。"""
    lo = max(psnr_ref.min(), psnr_test.min())
    hi = min(psnr_ref.max(), psnr_test.max())
    assert lo < hi, 'psnr ranges do not overlap'
    # 以 psnr 为自变量、log(rate) 为因变量积分
    order = np.argsort(psnr_ref)
    r_ref = np.polyfit(psnr_ref[order], np.log(rate_ref[order]), 3) if len(rate_ref) >= 4 else np.polyfit(psnr_ref[order], np.log(rate_ref[order]), max(len(rate_ref) - 1, 1))
    order = np.argsort(psnr_test)
    r_test = np.polyfit(psnr_test[order], np.log(rate_test[order]), 3) if len(rate_test) >= 4 else np.polyfit(psnr_test[order], np.log(rate_test[order]), max(len(rate_test) - 1, 1))
    xs = np.linspace(lo, hi, 1000)
    avg_ref = np.polyval(r_ref, xs).mean()
    avg_test = np.polyval(r_test, xs).mean()
    return (avg_test - avg_ref) / np.log(10.)


def load_rd_points(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.replace(',', ' ').split()
            pts.append((float(parts[0]), float(parts[1])))
    pts.sort()
    return np.array(pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True, help='RD points of the anchor: lines of "bpp psnr"')
    parser.add_argument('--test', required=True, help='RD points of the proposed method')
    args = parser.parse_args()

    ref = load_rd_points(args.ref)
    test = load_rd_points(args.test)
    br = bd_rate(ref[:, 0], ref[:, 1], test[:, 0], test[:, 1])
    bp = bd_psnr(ref[:, 0], ref[:, 1], test[:, 0], test[:, 1])
    print(f'BD-rate: {br:+.2f}%    BD-PSNR: {bp:+.4f} dB    (positive BD-rate = test worse)')


if __name__ == '__main__':
    main()
