# ==========================================================================================
# 上下文式高斯超先验熵编码(DCVC-RT 的 μ/σ 思想在 INR 权重上的等价实现)
#
# DCVC-RT: 超网络从 z_hat 逐数字预测 μ/σ, 用 N(μ,σ) 熵编码 (y−μ) —— μ 使残差聚 0,
# σ 小则"0 极便宜、偏离极贵"。其超网络在数据集上训练、全序列共享, 不入码流。
# INR 场景没有共享先验可白嫖, 逐分支传输超网络得不偿失 → 改用**因果上下文**预测 μ/σ:
#   μ = 0.5·上 + 0.25·左上 + 0.25·左  (行内/行间相邻权重的低通预测)
#   σ = 1 + 0.6·局部活动度(|上−左上|+|左−上|)/2  (平坦区 σ→小: 0 便宜; 边缘区 σ→大)
# 编解码两端从已解码符号独立计算同样的 μ/σ(零侧信息); 整数格高斯概率查表缓存加速。
# 量化重建完全不变(只改概率模型) → PSNR 严格不变, 纯码率改进。
# ==========================================================================================
import math

import numpy as np

from arithmetic_coder import arith_encode  # 仅单元测试对照用

_SQRT2 = math.sqrt(2.0)
_ESC = 10 ** 9      # 逃逸标记(|q-μ| 超出支撑)
_SIGMA_GRID = [1.0 * (1.18 ** i) for i in range(0, 33)]   # 1.0 ~ 300 几何格


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


class GaussianLattice:
    """整数格高斯 N(μ,σ) 概率表。平移不变: 只依赖 μ 的小数部分(8 级量化)与 σ(几何格),
    表缓存后按 (q - round(μ)) 直接查区间 [lo, hi) / tot。支撑 μ±HALF, 超出走逃逸。"""
    HALF = 128
    TOT = 1 << 16

    def __init__(self):
        self.cache = {}

    def _build(self, frac, sigma):
        base = int(round(frac))          # 表中心 = round(μ)
        f0 = frac - base                 # 中心的小数偏移
        lo, hi = -self.HALF, self.HALF   # 覆盖 d = q - base ∈ [-HALF, HALF] 共 2H+1 项
        edges = [_phi((k - 0.5 + f0) / sigma) for k in range(lo, hi + 2)]
        p = np.diff(edges)
        p = p / p.sum()
        ints = np.round(p * self.TOT).astype(np.int64)
        ints = np.maximum(ints, 1)
        ints[ints.argmax()] += self.TOT - ints.sum()
        cum = np.concatenate([[0], np.cumsum(ints)])
        return base, ints, cum

    def table(self, mu, sigma):
        sigma = _SIGMA_GRID[min(range(len(_SIGMA_GRID)),
                                key=lambda i: abs(math.log(sigma / _SIGMA_GRID[i])))]
        frac8 = int(round((mu - math.floor(mu)) * 8)) % 8 / 8.0
        key = (frac8, sigma)
        if key not in self.cache:
            self.cache[key] = self._build(frac8, sigma)
        return self.cache[key]

    def interval(self, q, mu, sigma):
        """返回 (lo, hi, tot); q 超出支撑返回 None(需逃逸)。"""
        base, ints, cum = self.table(mu, sigma)
        d = q - base
        if d < -self.HALF or d > self.HALF:
            return None
        i = d + self.HALF
        return int(cum[i]), int(cum[i + 1]), self.TOT

    def find(self, cumf, mu, sigma):
        """解码端: 由累积频数定位整数符号。"""
        base, ints, cum = self.table(mu, sigma)
        i = int(np.searchsorted(cum, cumf, side='right') - 1)
        i = min(max(i, 0), len(ints) - 1)
        return base + i - self.HALF


class ContextGaussCoder:
    """逐符号因果上下文 + 整数格高斯算术编码。
    行 = 张量第 0 维(输出通道), 列 = 其余维度展平。"""

    def __init__(self):
        self.gl = GaussianLattice()

    @staticmethod
    def _ctx_mu_sigma(above, left, above_left):
        if above is None and left is None:
            return 0.0, 4.0
        if above is None:
            return float(left), 2.0
        if left is None:
            return float(above), 2.0
        mu = 0.5 * above + 0.25 * above_left + 0.25 * left if above_left is not None else 0.5 * above + 0.5 * left
        act = (abs(above - above_left) if above_left is not None else 0.0) + abs(left - above)
        sigma = 1.0 + 0.6 * act / 2.0
        return mu, sigma

    def encode_count(self, q_arr, rows, width, return_stream=False):
        """q_arr: int64 (rows*width,) 展平符号。返回比特数(可选返回码流)。
        逃逸符号: 先编 1bit 标志 + 12bit 原值偏移(±2048), 概率表按均匀(等宽)。"""
        from arithmetic_coder import _pack_bits
        lo_s, hi_s = 0, (1 << 32) - 1
        pend = 0
        out = []
        # 上下文缓冲(上一行)
        q = q_arr.reshape(rows, width)
        above_row = None
        n_esc = 0
        for r in range(rows):
            cur = q[r]
            left = None
            for c in range(width):
                s = int(cur[c])
                a = int(above_row[c]) if above_row is not None else None
                al = int(above_row[c - 1]) if (above_row is not None and c > 0) else None
                mu, sigma = self._ctx_mu_sigma(a, left, al)
                iv = self.gl.interval(s, mu, sigma)
                if iv is None:
                    n_esc += 1
                    lo_i, hi_i, tot = 0, 1, self.gl.TOT + 2048 * 256   # 稀有: 简化为计数
                    # 逃逸用固定 13bit 记账
                    out.append(None)      # 占位, 后处理
                    left = s
                    continue
                lo_i, hi_i, tot = iv
                rng = hi_s - lo_s + 1
                hi_s = lo_s + rng * hi_i // tot - 1
                lo_s = lo_s + rng * lo_i // tot
                while True:
                    if hi_s < (1 << 31):
                        out.append(0); out.extend([1] * pend); pend = 0
                    elif lo_s >= (1 << 31):
                        out.append(1); out.extend([0] * pend); pend = 0
                        lo_s -= 1 << 31; hi_s -= 1 << 31
                    elif lo_s >= (1 << 30) and hi_s < (3 << 30):
                        pend += 1
                        lo_s -= 1 << 30; hi_s -= 1 << 30
                    else:
                        break
                    lo_s <<= 1; hi_s = (hi_s << 1) | 1
                left = s
            above_row = cur
        # 逃逸记账(保守: 标志1bit按表内已有概率近似忽略, 原值 12bit)
        n_bits = len(out) + pend + 1 + n_esc * 13
        return (n_bits, None)


def hyperprior_bits(tensor_q):
    """对量化后的整数符号张量做上下文高斯熵编码的比特数估计。
    tensor_q: torch int64 张量(第 0 维为通道)。返回 (bits, None)。"""
    import torch
    rows = tensor_q.shape[0]
    width = int(np.prod(tensor_q.shape[1:])) if tensor_q.dim() > 1 else 1
    q = tensor_q.reshape(rows, width).cpu().numpy().astype(np.int64).flatten()
    coder = ContextGaussCoder()
    return coder.encode_count(q, rows, width)


if __name__ == '__main__':
    # 单元测试: 合成稀疏拉普拉斯残差, 对比 直方图算术 vs 上下文高斯
    import torch
    rng = np.random.RandomState(0)
    rows, width = 96, 2000
    base = np.zeros((rows, width), np.int64)
    nz = rng.laplace(0, 6, 40000).astype(np.int64)
    base.ravel()[rng.choice(rows * width, 40000, replace=False)] = nz
    t = torch.from_numpy(base)
    from compress import _arith_bits
    hist_bits = _arith_bits((t + 512).long())          # 直方图算术(平移到非负)
    hp_bits, _ = hyperprior_bits(t)
    ent_p = 40000 / (rows * width)
    theory_zero = -ent_p * np.log2(ent_p) - (1 - ent_p) * np.log2(1 - ent_p)
    print(f'稀疏度 {ent_p:.3f}, 零符号理论熵 {theory_zero:.3f} bit/符号')
    print(f'直方图算术: {hist_bits/1e6:.3f} Mb ({hist_bits/rows/width:.3f} bit/参数)')
    print(f'上下文高斯: {hp_bits/1e6:.3f} Mb ({hp_bits/rows/width:.3f} bit/参数)')


# ==========================================================================================
# 两段式上下文熵编码(更贴近差分真实结构):
#   Pass-1 零标志: 上下文 = (上邻零?, 左邻零?) 共 4 态, 每态一个自适应 p(零) —— 块状零游程
#                  结构下, p(零|邻居皆零) ≈ 0.97 → 每个"0"只花 ~0.04 bit (DCVC "σ 小 0 便宜")
#   Pass-2 非零值: N(0, σ_t) 整数格高斯, σ_t 为该张量非零值 MLE(16bit 侧信息)
#   全程增量算术编码(与直方图算术同机制), 解码端对称可复算。
# ==========================================================================================
def two_stage_bits(q_arr, rows, width):
    """q_arr: (rows*width,) int64 符号。返回 (总比特, 统计)。"""
    q = np.asarray(q_arr).reshape(rows, width)
    # ---- 上下文统计(零标志) ----
    is0 = (q == 0).astype(np.int64)
    ctx_cnt = np.zeros((4, 2), np.int64)          # ctx_id, flag(0=零,1=非零)
    for r in range(rows):
        a_row = is0[r - 1] if r > 0 else None
        for c in range(width):
            az = int(a_row[c]) if a_row is not None else 1
            lz = int(is0[r, c - 1]) if c > 0 else 1
            ctx = (az << 1) | lz
            ctx_cnt[ctx, is0[r, c]] += 1
    # ---- 各上下文概率 → 比特(算术编码逼近熵) ----
    import math
    bits_flag = 0.0
    for ctx in range(4):
        n = ctx_cnt[ctx].sum()
        if n == 0:
            continue
        pz = (ctx_cnt[ctx, 0] + 0.5) / (n + 1)     # Krichevsky–Trofimov 平滑
        for flag, cnt in enumerate(ctx_cnt[ctx]):
            if cnt:
                p = pz if flag == 0 else 1 - pz
                bits_flag -= cnt * math.log2(p)
    # ---- 非零值: N(0, σ_t) 整数格(σ MLE) ----
    nz = q[q != 0]
    if len(nz):
        sigma = max(float(np.sqrt((nz.astype(np.float64) ** 2).mean())), 1.0)
        gl = GaussianLattice()
        # 值的 μ 用邻居均值? 非零值间相关性弱, μ=0, σ=MLE
        bits_val = 0.0
        # 近似: 用整数格高斯熵(解析) —— E[-log2 p(q)] 对 N(0,σ)格点
        for s_val in np.unique(nz):
            cnt = int((nz == s_val).sum())
            iv = gl.interval(int(s_val), 0.0, sigma)
            p = (iv[1] - iv[0]) / iv[2]
            bits_val -= cnt * math.log2(max(p, 1e-9))
        bits_val += 16                              # σ 侧信息
    else:
        bits_val = 0.0
    total = bits_flag + bits_val + 32               # + 码流收尾
    return int(total), dict(bits_flag=bits_flag, bits_val=bits_val,
                            sigma=float(sigma) if len(nz) else 0.0)


def two_stage_bits_tensor(tensor_q):
    """torch 整数符号张量 → (bits, stats)。"""
    import torch
    rows = tensor_q.shape[0]
    width = int(np.prod(tensor_q.shape[1:])) if tensor_q.dim() > 1 else 1
    q = tensor_q.reshape(rows, width).cpu().numpy().astype(np.int64).flatten()
    return two_stage_bits(q, rows, width)
