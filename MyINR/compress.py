# ==========================================================================================
# 模型压缩流水线 (NeRV 剪枝/量化 + DCVC 风格算术熵编码 + P-model 权重差编码)
#
#  - I-model   : 训练后期渐进全局 L1 剪枝(trainer 内完成, 权重已含 0) → 逐通道 8bit 量化
#                → 算术编码(零值作为普通符号参与直方图, 掩码成本被信息论地包含在内)
#  - P-model   : 权重差 Δw = w_P - w_{P-1} → 幅值阈值化(训练后期 L1 正则诱导稀疏)
#                → 非零元素逐通道量化 → 算术编码(零值占多数时熵极低)
#  - 帧内容嵌入表: 逐帧整张量 8bit 量化 → 算术编码(作为边信息计入码率)
#  解码端按  w_P = dequant(P_{k-1}) + dequant(Δw)  级联重建, 与训练/评估自洽。
# ==========================================================================================
import collections

import numpy as np
import torch

from arithmetic_coder import arith_encode

AUX_PREFIXES = ('flow_head', 'conf_head')       # 辅助头只在训练期使用, 不计入码率


def is_coded(k):
    return not any(k.startswith(p) for p in AUX_PREFIXES)


def prunable_keys(sd):
    """可剪枝权重集合(stem Linear + conv, 与原版 NeRV 一致): 2D+ 且非 bias。"""
    return {k for k, v in sd.items()
            if is_coded(k) and not k.startswith('frame_embed') and v.dim() >= 2 and 'bias' not in k}


def compute_global_prune_mask(sd, keep_ratio, prev_mask=None):
    """全局 |w| 剪枝掩码: 在可剪枝权重的全体元素中按幅值保留 keep_ratio。
    prev_mask: 之前阶段的掩码(渐进式只减不增)。"""
    vals = []
    for k in prunable_keys(sd):
        v = sd[k].detach().float().abs().flatten()
        if prev_mask is not None and prev_mask.get(k) is not None:
            v = v[prev_mask[k].flatten()]
        vals.append(v)
    allv = torch.cat(vals)
    n_keep = max(1, int(allv.numel() * keep_ratio))
    thr = torch.kthvalue(allv, allv.numel() - n_keep + 1).values
    mask = {}
    for k, v in sd.items():
        if k in prunable_keys(sd):
            m = v.detach().abs() >= thr
            if prev_mask is not None and prev_mask.get(k) is not None:
                m = m & prev_mask[k]
            mask[k] = m
        else:
            mask[k] = None
    return mask


def quantize_tensor(t, bits=8, axis=0, symmetric=True):
    """均匀量化。axis=0: 逐通道; axis=-1: 整张量。
    symmetric=True: 每通道对称量化 scale=|w|max/2^(bits-1), q=round(w/scale) ——
    权重 0 恒映射到同一中间符号, 使剪枝/权重差产生的 0 在熵编码时高度集中(近乎免费编码掩码)。
    返回 (int 符号 q(已平移到非负), 反量化 deq, meta)。"""
    t = t.detach().float()
    if axis == -1 or t.dim() < 2:
        if symmetric:
            scale = t.abs().max() / (2 ** (bits - 1)) + 1e-19
            q = (t / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            deq = q * scale
            return (q + 2 ** (bits - 1)).long(), deq, scale
        tmin = t.min()
        scale = (t.max() - tmin) / (2 ** bits)
        q = ((t - tmin) / (scale + 1e-19)).round().clamp(0, 2 ** bits)
        deq = tmin + scale * q
        return q.long(), deq, (scale, tmin)
    dims = tuple(range(1, t.dim()))
    if symmetric:
        amax = t.abs().amax(dim=dims)
        scale = (amax / (2 ** (bits - 1))).clamp_min(1e-20).view([-1] + [1] * (t.dim() - 1))
        q = (t / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        deq = q * scale
        return (q + 2 ** (bits - 1)).long(), deq, scale.flatten()
    tmax = t.amax(dim=dims)
    tmin = t.amin(dim=dims)
    scale = ((tmax - tmin) / (2 ** bits)).view([-1] + [1] * (t.dim() - 1))
    tmin_v = tmin.view([-1] + [1] * (t.dim() - 1))
    q = ((t - tmin_v) / (scale + 1e-19)).round().clamp(0, 2 ** bits)
    deq = tmin_v + scale * q
    return q.long(), deq, (scale.flatten(), tmin)


def _arith_bits(q_int):
    arr = q_int.flatten().cpu().numpy().astype(np.int64)
    if arr.size == 0:
        return 0
    arr = np.where(np.isnan(arr) | np.isinf(arr), 0, arr)   # NaN/Inf guard
    arr = np.clip(arr, 0, None)                              # 确保非负(算术编码要求)
    alphabet = int(arr.max()) + 1 if arr.max() >= 0 else 1
    if alphabet < 2:
        return arr.size  # 全零: 每符号 1 bit 计数
    data, _ = arith_encode(arr, alphabet)
    return len(data) * 8


def _scale_bits(meta):
    """逐通道 scale 的边信息比特(16bit/通道, 熵编码前的基础开销)。"""
    if meta is None:
        return 0
    m = meta if torch.is_tensor(meta) else torch.tensor([meta])
    return int(m.numel()) * 16


def _entropy_bits_est(q_int):
    """直方图熵估计(快速, 用于位深搜索; 不做实际编码)。"""
    arr = q_int.flatten().cpu().numpy().astype(np.int64)
    if arr.size == 0:
        return 0
    p = np.bincount(arr - arr.min()).astype(np.float64)
    p = p[p > 0] / arr.size
    return float(-(p * np.log2(p)).sum() * arr.size)


def _select_delta_bits(delta, base_bits, tol=0.05, candidates=(4, 5, 6, 7, 8)):
    """P6-lite: 逐张量位深选择 —— 取满足 相对量化误差 ≤ tol 的最小位宽
    (差分量化噪声应显著小于差分本身的幅度), 否则取候选中最大位宽。
    返回 (bits, q, deq, meta)。"""
    best = None
    for b in sorted(candidates):
        q, deq, meta = quantize_tensor(delta, bits=b, axis=0 if delta.dim() >= 2 else -1)
        err = (deq - delta).norm()
        rel = (err / (delta.norm() + 1e-12)).item()
        best = (b, q, deq, meta)
        if rel <= tol:
            break
    return best


def compress_state_dict(sd, bits=8, mode='dense', ref_sd=None, delta_keep=0.5,
                        delta_bits_search=True, delta_tol=0.05, delta_coder='hist', q_steps=None):
    """
    压缩 state_dict, 返回 (总比特, 反量化 state_dict, 分量比特明细)。
    mode='dense': 直接量化+编码(权重中的 0 来自训练期剪枝, 对称量化使 0 对齐同一符号)。
    mode='delta': Δw = w - ref → 幅值阈值保留 delta_keep → 量化+编码;
                  ref 必须是"解码端已有"的反量化权重(保证级联自洽);
                  P6-lite: 逐张量按相对误差容差自适应选择 4-8bit 位深。
    frame_embed 表: 逐帧 min/max 量化 + 逐帧编码。
    """
    detail = {}
    total_bits = 0
    new_sd = {}
    for k, v in sd.items():
        if not is_coded(k):
            new_sd[k] = v
            continue
        if k.startswith('frame_embed'):
            b_sum = 0
            deq_frames = []
            for f in range(v.shape[0]):
                q, deq, meta = quantize_tensor(v[f], bits=bits, axis=-1, symmetric=False)
                b_sum += _arith_bits(q) + _scale_bits(meta)
                deq_frames.append(deq)
            total_bits += b_sum
            detail[k] = b_sum
            new_sd[k] = torch.stack(deq_frames).to(v.dtype) if deq_frames else v
            continue
        if mode == 'delta':
            from hyperprior import hyperprior_bits
            if k not in ref_sd or ref_sd[k].shape != v.shape:
                # 投影不匹配(branch_ratio 不整除): 该张量退化为 dense 编码
                q, deq, meta = quantize_tensor(v, bits=bits, axis=0 if v.dim() >= 2 else -1)
                n_bits = _arith_bits(q) + _scale_bits(meta)
                total_bits += n_bits
                detail[k] = n_bits
                new_sd[k] = deq.to(v.dtype)
                continue
            ref = ref_sd[k].float()
            delta = v.float() - ref
            if k in prunable_keys(sd):
                m = compute_global_prune_mask({k: delta}, delta_keep)[k]
                delta = delta * m
            qsk = k.replace('.', '_')
            if q_steps is not None and qsk in q_steps and v.dim() >= 2:
                # V4.1 UAR 式: 差分用训练出的 q_step 量化(分布已被 R+λD 训练推到步长附近,
                # 大多取整到 0/±1 → 直方图熵编码极省), 不做 amax 定标与位深搜索
                qs = q_steps[qsk].float()
                q_raw = (delta / qs).round().clamp(-2**(bits-1), 2**(bits-1) - 1).long()
                deq_d = q_raw.float() * qs
                n_bits = _arith_bits(q_raw + 2**(bits-1)) + 16
                total_bits += n_bits
                detail[k] = n_bits
                new_sd[k] = (ref + deq_d).to(v.dtype)
                continue
            if delta_bits_search:
                b, q, deq_d, meta = _select_delta_bits(delta, bits, tol=delta_tol)
            else:
                b = bits
                q, deq_d, meta = quantize_tensor(delta, bits=b, axis=0 if v.dim() >= 2 else -1)
            n_bits = _arith_bits(q) + _scale_bits(meta)
            if delta_coder in ('hyper', 'min') and v.dim() >= 2:
                # 上下文熵编码(DCVC-RT μ/σ 思想): 质量不变(反量化同), 仅换概率模型
                # 候选: 两段式(零标志上下文 + 非零高斯) 与 逐符号高斯, 取最小
                try:
                    from hyperprior import two_stage_bits_tensor
                    ts_bits = two_stage_bits_tensor(q)[0] + _scale_bits(meta)
                except Exception:
                    ts_bits = 1 << 60
                try:
                    hp_bits = hyperprior_bits(q)[0] + _scale_bits(meta)
                except Exception:
                    hp_bits = 1 << 60
                if delta_coder == 'hyper':
                    n_bits = min(ts_bits, hp_bits) + 1
                else:   # 'min': hist vs two-stage vs per-symbol-Gaussian, 逐张量择优(+2bit 模式标记)
                    cands = [n_bits, ts_bits, hp_bits]
                    n_bits = min(cands) + 2 * (min(cands) < max(cands))
            total_bits += n_bits
            detail[k] = n_bits
            new_sd[k] = (ref + deq_d).to(v.dtype)
        else:
            qsk = k.replace('.', '_')
            if q_steps is not None and qsk in q_steps and v.dim() >= 2:
                # UAR-NVC 式: 用训练好的 q_step 量化(而非 amax/2^b)
                qs = q_steps[qsk].float()
                q_raw = (v.float() / qs).round().clamp(-2**(bits-1), 2**(bits-1) - 1).long()
                deq = q_raw.float() * qs
                n_bits = _arith_bits(q_raw + 2**(bits-1)) + 16
            else:
                q, deq, meta = quantize_tensor(v, bits=bits, axis=0 if v.dim() >= 2 else -1)
                n_bits = _arith_bits(q) + _scale_bits(meta)
            total_bits += n_bits
            detail[k] = n_bits
            new_sd[k] = deq.to(v.dtype)
    return total_bits, new_sd, detail


def huffman_bits(sd, bits=8, only_nonzero=False):
    """NeRV 原版 Huffman 统计(对照用)。only_nonzero=True 时与原版口径一致(不计 0)。"""
    from dahuffman import HuffmanCodec
    allq = []
    for k, v in sd.items():
        if not is_coded(k) or k.startswith('frame_embed'):
            continue
        q, _, _ = quantize_tensor(v, bits=bits, axis=0 if v.dim() >= 2 else -1)
        qf = q.flatten()
        if only_nonzero:
            vv = v.flatten()
            qf = qf[vv != 0]
        allq.append(qf)
    if not allq:
        return 0
    arr = torch.cat(allq).cpu().numpy().astype(np.int64).tolist()
    codec = HuffmanCodec.from_data(arr)
    tbl = codec.get_code_table()
    cnt = collections.Counter(arr)
    return sum(cnt[s] * tbl[s][0] for s in cnt)
