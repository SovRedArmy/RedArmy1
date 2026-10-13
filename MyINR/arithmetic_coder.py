# ==========================================================================================
# 静态多符号算术编码器 (CACM87 / Witten-Neal-Cleary 风格, 32-bit 区间)
# 借鉴 DCVC 家族的"熵编码用算术编码"思路: 对量化后的权重符号用逐张量频率表做算术编码,
# 逼近符号熵, 优于 NeRV 原版的 Huffman(尤其对零值占多数的 P-model 权重差)。
# 纯 Python 实现, 每符号 ~2us, 百万级符号秒级完成。
# ==========================================================================================
import numpy as np

PRECISION = 32
FULL = 1 << PRECISION
HALF = FULL >> 1
QUARTER = FULL >> 2
CODE_VALUE_BITS = 16            # 频率表归一化的总计数 = 2^16


def _build_cumfreq(symbols, alphabet):
    """由符号直方图构造累积频率表(长度 alphabet+1, 总计数 = 2^16, 每个出现过的符号至少 1)。"""
    counts = np.bincount(symbols, minlength=alphabet).astype(np.int64)
    counts = np.maximum(counts, 1)                      # 未出现符号也给 1, 保证可解码
    counts = (counts.astype(np.float64) / counts.sum() * (1 << CODE_VALUE_BITS)).astype(np.int64)
    counts = np.maximum(counts, 1)
    s = counts.sum()
    # 修正缩放误差, 使总计数恰为 2^16
    if s != (1 << CODE_VALUE_BITS):
        diff = (1 << CODE_VALUE_BITS) - s
        idx = int(np.argmax(counts))
        counts[idx] += diff
        assert counts[idx] > 0
    cumfreq = np.concatenate([[0], np.cumsum(counts)])
    assert cumfreq[-1] == (1 << CODE_VALUE_BITS)
    return cumfreq


def _pack_bits(bits):
    """MSB-first 位打包。bits: 0/1 数组 → bytes"""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
    bits = bits.reshape(-1, 8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    return (bits * weights).sum(1).astype(np.uint8).tobytes()


def _unpack_bits(data, n):
    """逆打包: 取前 n 位。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1)[None, :]) & 1).reshape(-1)
    return bits[:n]


def arith_encode(symbols, alphabet=None):
    """symbols: 非负 int 数组 → bytes。返回 (bitstream, cumfreq_table)。"""
    symbols = np.asarray(symbols, dtype=np.int64)
    if alphabet is None:
        alphabet = int(symbols.max()) + 1 if len(symbols) else 1
    cumfreq = _build_cumfreq(symbols, alphabet)

    low, high, pending = 0, FULL - 1, 0
    out = []
    cf = cumfreq.tolist()
    tot = 1 << CODE_VALUE_BITS
    for s in symbols.tolist():
        rng = high - low + 1
        cumlow, cumhigh = cf[s], cf[s + 1]
        high = low + rng * cumhigh // tot - 1
        low = low + rng * cumlow // tot
        while True:
            if high < HALF:
                out.append(0)
                out.extend([1] * pending)
                pending = 0
            elif low >= HALF:
                out.append(1)
                out.extend([0] * pending)
                pending = 0
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1

    # 终止: pending += 1, 发送 QUARTER 归属位 + pending 个反位 (Doty finish)
    pending += 1
    b = 1 if low >= QUARTER else 0
    out.append(b)
    out.extend([1 - b] * pending)
    out.extend([0] * 4)                       # flush 余量
    return _pack_bits(out), cumfreq


def arith_decode(data, n_symbols, cumfreq):
    """逆过程: 返回长度 n_symbols 的 int 数组。"""
    cumfreq = np.asarray(cumfreq)
    total = int(cumfreq[-1])
    low, high = 0, FULL - 1
    bits = _unpack_bits(data, PRECISION + n_symbols * PRECISION + 64)
    pos = 0

    def next_bit():
        nonlocal pos
        b = int(bits[pos]) if pos < len(bits) else 0
        pos += 1
        return b

    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | next_bit()

    out = np.empty(n_symbols, dtype=np.int64)
    cf = cumfreq.tolist()
    for i in range(n_symbols):
        rng = high - low + 1
        cumf = ((value - low + 1) * total - 1) // rng
        # 在累积表中找到 s: cf[s] <= cumf < cf[s+1]
        s = int(np.searchsorted(cf, cumf, side='right') - 1)
        s = min(max(s, 0), len(cf) - 2)
        out[i] = s
        cumlow, cumhigh = cf[s], cf[s + 1]
        high = low + rng * cumhigh // total - 1
        low = low + rng * cumlow // total
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                value -= QUARTER
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = ((value << 1) | next_bit()) & (FULL - 1)
    return out


def bits_of(symbols, alphabet=None):
    """仅统计算术编码后的比特数(不保留码流)。"""
    data, _ = arith_encode(symbols, alphabet)
    return len(data) * 8


if __name__ == '__main__':
    # 自测: 随机数据 + 零值稀疏数据 的编解码往返与比特数
    rng = np.random.RandomState(0)
    for name, arr in [
        ('uniform-256', rng.randint(0, 256, 20000)),
        ('laplace', np.clip(rng.laplace(0, 8, 50000).astype(np.int64), -255, 255) + 255),
        ('sparse-delta', np.concatenate([np.zeros(90000, np.int64), rng.randint(-120, 120, 10000) + 255])),
    ]:
        assert arr.min() >= 0
        data, cf = arith_encode(arr)
        dec = arith_decode(data, len(arr), cf)
        assert (dec == arr).all(), f'{name} roundtrip FAILED'
        ent = 0.0
        p = np.bincount(arr) / len(arr)
        p = p[p > 0]
        ent = float(-(p * np.log2(p)).sum() * len(arr))
        print(f'{name}: {len(arr)} symbols, {len(data)*8} bits (entropy {ent:.0f}) OK')
