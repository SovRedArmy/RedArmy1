# ==========================================================================================
# BD-RATE 计算 (Bjontegaard delta rate) —— 纯 numpy 实现
# 采用 JVET 通用测试条件使用的 PCHIP (单调三次 Hermite 插值) 拟合 RD 曲线,
# 在两条曲线的 PSNR 重叠区间上积分 log10(rate) 差:
#     BD-rate = (10^D - 1) * 100%,  D = mean(log-rate distance)
# 每段 Hermite 多项式有解析积分: ∫ = h*[(y0+y1)/2 - h*(d0-d1)/12]
# ==========================================================================================
import numpy as np


def _pchip_slopes(x, y):
    """Fritsch-Carlson 切线: 单调加权调和平均, 保单调防过冲。"""
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(y)
    d = np.zeros(n)
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # 端点单侧三点公式 + 单调夹取
    if n >= 3:
        d[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
        if np.sign(d[0]) != np.sign(delta[0]):
            d[0] = 0.0
        elif np.sign(delta[0]) != np.sign(delta[1]):
            d[0] = 0.5 * delta[0]
        d[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
        if np.sign(d[-1]) != np.sign(delta[-1]):
            d[-1] = 0.0
        elif np.sign(delta[-1]) != np.sign(delta[-2]):
            d[-1] = 0.5 * delta[-1]
    else:
        d[:] = delta
    return d


def _pchip_integral(x, y, xlo, xhi):
    """在 [xlo, xhi] 上积分 PCHIP 曲线 y(x) (x 单调增)。"""
    d = _pchip_slopes(x, y)
    total = 0.0
    for k in range(len(x) - 1):
        x0, x1 = x[k], x[k + 1]
        lo, hi = max(x0, xlo), min(x1, xhi)
        if hi <= lo:
            continue
        y0, y1, d0, d1 = y[k], y[k + 1], d[k], d[k + 1]
        h_full = x1 - x0
        # 完整段积分
        seg_full = h_full * (y0 + y1) / 2.0 - h_full * h_full * (d0 - d1) / 12.0
        if lo <= x0 and hi >= x1:
            total += seg_full
        else:
            # 局部段: 在 [lo,hi] 上重参数化后的 Hermite 积分
            def seg(a, b):
                t0, t1 = (a - x0) / h_full, (b - x0) / h_full
                # y(t) = h00*y0 + h10*d0*h + h01*y1 + h11*d1*h, 关于 t 三次
                # ∫_t0^t1 y dt 用三次多项式系数
                c0 = y0
                c1 = d0 * h_full
                c2 = (3 * (y1 - y0) / h_full - (2 * d0 + d1) * h_full) / h_full  # 系数按 t 幂次
                # 直接构造关于 t 的多项式系数: p(t)=a0+a1 t+a2 t^2+a3 t^3
                a0 = y0
                a1 = d0 * h_full
                a2 = (3 * (y1 - y0) - h_full * (2 * d0 + d1)) / h_full ** 2 * h_full ** 2 / h_full
                a2 = 3 * (y1 - y0) / h_full ** 2 - (2 * d0 + d1) / h_full
                a3 = (-2 * (y1 - y0) / h_full ** 3 + (d0 + d1) / h_full ** 2)
                I = (a0 * t1 + a1 * t1 ** 2 / 2 + a2 * t1 ** 3 / 3 + a3 * t1 ** 4 / 4) - \
                    (a0 * t0 + a1 * t0 ** 2 / 2 + a2 * t0 ** 3 / 3 + a3 * t0 ** 4 / 4)
                return I * h_full

            total += seg(lo, hi)
    return total


def bd_rate(anchor_points, test_points):
    """BD-rate(%). points: [(psnr, bpp), ...] >= 3 个点; 负值 = test 比 anchor 省码率。"""
    a = sorted(anchor_points, key=lambda p: p[0])
    t = sorted(test_points, key=lambda p: p[0])
    xa, ya = np.array([p[0] for p in a]), np.log10([p[1] for p in a])
    xt, yt = np.array([p[0] for p in t]), np.log10([p[1] for p in t])
    qlo = max(xa.min(), xt.min())
    qhi = min(xa.max(), xt.max())
    if qhi - qlo < 1e-6:
        return float('nan')          # 无重叠区间
    D = (_pchip_integral(xt, yt, qlo, qhi) - _pchip_integral(xa, ya, qlo, qhi)) / (qhi - qlo)
    return (10 ** D - 1) * 100.0


def bd_psnr(anchor_points, test_points):
    """BD-PSNR(dB), 在 log-rate 重叠区间上积分 PSNR 差。"""
    a = sorted(anchor_points, key=lambda p: p[1])
    t = sorted(test_points, key=lambda p: p[1])
    # x = log10(rate), y = psnr
    xa, ya = np.log10([p[1] for p in a]), np.array([p[0] for p in a])
    xt, yt = np.log10([p[1] for p in t]), np.array([p[0] for p in t])
    qlo = max(xa.min(), xt.min())
    qhi = min(xa.max(), xt.max())
    if qhi - qlo < 1e-6:
        return float('nan')
    D = (_pchip_integral(xt, yt, qlo, qhi) - _pchip_integral(xa, ya, qlo, qhi)) / (qhi - qlo)
    return D


if __name__ == '__main__':
    # 与经典 cubic BD-rate 实现对照量级的简单自测
    anchor = [(30.0, 0.02), (32.0, 0.05), (34.0, 0.12)]
    test1 = [(30.5, 0.02), (32.5, 0.05), (34.5, 0.12)]     # 同码率 +0.5dB → BD-rate 应明显为负
    test2 = [(30.0, 0.018), (32.0, 0.045), (34.0, 0.108)]  # 同质量 ~10% 省码率 → ≈ -10%
    print('bd_rate same-rate-better:', round(bd_rate(anchor, test1), 2))
    print('bd_rate 10% saving:', round(bd_rate(anchor, test2), 2))
    print('bd_psnr +0.5:', round(bd_psnr(anchor, test1), 3))
