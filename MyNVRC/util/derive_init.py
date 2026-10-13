# -*- coding: utf-8 -*-
"""
MyHiNeRV 两级架构迁移（模块 C）：宽 base 模型 → clip 小模型的初始化与残差参考。

设计（对应用户五点规范 + NVRC 语境）：
1. θ'_k = ε·derive(base) + (1−ε)·θ*_{k−1}（II-init 融合版；derive 承担 UAR-NVC 里
   随机初始化的"通用先验"角色，θ*_{k−1} 是前一 GoP 的解码权重）；
2. ε = clamp(1 − a·e^{−b·gap + c})，gap = 相邻 GoP 边界帧 MSE×256²（UAR-NVC 公式 9）；
3. gop0（首个 GoP）：θ' = derive(base)（I-model 角色）；
4. θ' 同时用作：a) clip 模型的初始权重；b) 残差编码参考（编码 θ−θ'，见 NVRC.set_residual_ref）。

NVRC 参数体系：扁平键 'prefix__name'（models/layers.get_params_recursive），网格张量的
第 0 维是时间维（T 优先展平），层权重首维是输出通道——derive 按名称匹配 + 形状自适应映射。
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


# ---------------- derive：base 宽模型 → clip 小模型 ----------------

def _interp_time(x: torch.Tensor, t_out: int) -> torch.Tensor:
    """把张量第 0 维（时间维）线性插值到 t_out。"""
    t_in = x.shape[0]
    if t_in == t_out:
        return x
    rest = x.shape[1:]
    y = x.reshape(t_in, -1).t()[None].float()          # [1, M, T]
    y = F.interpolate(y, size=t_out, mode='linear', align_corners=False)
    return y[0].t().reshape((t_out,) + rest)


def _map_tensor(dst: torch.Tensor, src: torch.Tensor, t_first: bool) -> torch.Tensor:
    """src → dst 形状：t_first 时先做时间插值，再逐维取前 d 个通道/深度。"""
    if tuple(src.shape) == tuple(dst.shape):
        return src
    out = src.float()
    if t_first and out.shape[0] != dst.shape[0]:
        out = _interp_time(out, dst.shape[0])
    for d in range(out.ndim):
        if out.shape[d] > dst.shape[d]:
            out = out.narrow(d, 0, dst.shape[d])
    assert tuple(out.shape) == tuple(dst.shape), \
        f'derive shape mismatch: {tuple(src.shape)} -> {tuple(out.shape)} vs dst {tuple(dst.shape)}'
    return out


def is_grid_key(key: str) -> bool:
    """NVRC 扁平参数名中网格类参数（时间维在第 0 维需插值的张量）。"""
    return 'grid' in key or 'encoding' in key


def derive_params(clip_params: dict, base_params: dict, logger=None) -> dict:
    """从 base 宽模型参数推导 clip 小模型参数（名称匹配 + 形状自适应映射）。"""
    derived, n_map, n_skip = {}, 0, 0
    for k, dst in clip_params.items():
        if not torch.is_tensor(dst) or dst is None:
            n_skip += 1
            continue
        src = base_params.get(k, None)
        if src is None or not torch.is_tensor(src):
            n_skip += 1
            continue
        try:
            derived[k] = _map_tensor(dst, src, t_first=is_grid_key(k))
            n_map += 1
        except AssertionError:
            n_skip += 1
    if logger is not None:
        logger.info(f'[derive_init] derive: mapped {n_map} tensors, skipped {n_skip}')
    return derived


# ---------------- II-init：ε 插值 ----------------

def interp_params(a: dict, b: dict, eps: float, logger=None) -> dict:
    """θ' = eps·a + (1−eps)·b（只插值 ndim≥2 的浮点张量；1D 偏置取 a 侧）。"""
    out = dict(a)
    n = 0
    for k, av in a.items():
        if not torch.is_tensor(av) or av is None:
            continue
        bv = b.get(k, None)
        if bv is not None and torch.is_tensor(bv) and av.ndim >= 2 and torch.is_floating_point(av) \
                and tuple(av.shape) == tuple(bv.shape):
            out[k] = (eps * av.float() + (1.0 - eps) * bv.float()).to(av.dtype)
            n += 1
    if logger is not None:
        logger.info(f'[derive_init] interp: eps={eps:.3f}, interp {n}/{len(a)} tensors')
    return out


def ii_init_epsilon(gap_mse256, a=0.6, b=2.2e-5, c=0.5):
    """UAR-NVC 公式(9)：ε = clamp(1 − a·e^{−b·gap + c}, 0, 1)。gap 大 → 信 base。"""
    eps = 1.0 - a * math.exp(-b * float(gap_mse256) + c)
    return float(min(max(eps, 0.0), 1.0))


def boundary_gap_mse(dataset_dir, video_name, t_prev_last, t_cur_first):
    """相邻 GoP 边界帧 MSE×256²（前一 GoP 尾帧 vs 当前 GoP 首帧）。"""
    from PIL import Image
    d = os.path.join(dataset_dir, video_name)
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg')))
    a = np.asarray(Image.open(os.path.join(d, files[t_prev_last])).convert('RGB'), np.float32) / 255.
    b = np.asarray(Image.open(os.path.join(d, files[t_cur_first])).convert('RGB'), np.float32) / 255.
    return float(((a - b) ** 2).mean() * 256 ** 2)


# ---------------- base 权重加载 ----------------

def load_base_params(path, logger=None):
    """从 base 运行（宽模型整段训练）的解码模型保存中加载扁平参数。
    path 指向 save_model 的输出目录（含 pytorch_model.bin）或 .bin 文件本身。"""
    if os.path.isdir(path):
        path = os.path.join(path, 'pytorch_model.bin')
    sd = torch.load(path, map_location='cpu', weights_only=False)
    # save_model 保存 codec 的 state_dict；param_model 的参数键形如
    # 'param_model.params.<flat_name>'——提取为扁平 dict。
    prefix = 'param_model.params.'
    flat = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            flat[k[len(prefix):]] = v.detach().float().cpu()
    if not flat:
        # 兼容：直接就是扁平 dict 的情形
        flat = {k: v.detach().float().cpu() for k, v in sd.items()
                if torch.is_tensor(v) and v.ndim >= 1 and '.' not in k.split('__')[0]}
    if logger is not None:
        logger.info(f'[derive_init] loaded base params: {len(flat)} tensors from {path}')
    return flat
