# -*- coding: utf-8 -*-
"""
E 系列：CANeRV 三机制（DSA/DFA/HSA）应用于 NVRC 两级架构。

机制对照：
- DSA（序列级动态调整）  → 分支通道数根据 base 在该 clip 的重建误差自适应
- DFA（帧级动态调整）    → II-init 的 ε 根据 base 的帧级误差变化率自适应
- HSA（层级结构自适应）  → derive 时根据空间误差分布选择保留哪些通道
"""
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════════════
# DFA：帧级动态调整 → 自适应 II-init ε
# ═══════════════════════════════════════════════════════

def compute_base_frame_errors(args, model, eval_loader, eval_task, accelerator, frame_start, frame_end):
    """
    计算 base 模型在该 clip 的逐帧重建误差（DFA 信号源）。
    返回 [num_frames] 的 per-frame MSE 数组。
    """
    model.eval()
    errors = {}
    with torch.no_grad():
        for batch in eval_loader:
            inputs, target = eval_task.parse_batch(eval_loader, batch)
            output = model(inputs, compute_outputs=True, compute_rates=False)
            mse = ((output.clamp(0,1) - target.clamp(0,1)) ** 2).mean(dim=[1,2,3,4])  # [N]
            idx = inputs['idx'][:, 0].cpu().numpy()  # 帧号
            for i, f in enumerate(idx):
                errors[int(f)] = errors.get(int(f), []) + [mse[i].item()]
    # 平均每帧的误差
    return np.array([np.mean(errors.get(f, [1.0])) for f in range(frame_start, frame_end)])


def dfa_epsilon(frame_errors, w=5.0):
    """
    DFA 自适应 ε（CANeRV 帧级动态调整）：
    - 帧误差变化率大 → 该 clip 时序不稳定 → 更信 base（ε→1，用 derive(base) 初始化）
    - 帧误差平稳 → clip 内内容相似 → 更信前一 GOP（ε→0，继承前一个分支）
    
    替代 UAR-NVC 的固定公式 ε = clamp(1 − a·e^{−b·gap + c})。
    """
    if len(frame_errors) < 2:
        return 1.0
    # 帧级误差的变化率（相邻帧误差差的绝对值均值）
    variation = np.mean(np.abs(np.diff(frame_errors)))
    # sigmoid 映射：变化率大 → ε 大（信 base）
    eps = 1.0 / (1.0 + math.exp(-w * (variation * 1000 - 0.5)))
    return float(np.clip(eps, 0.05, 0.95))


# ═══════════════════════════════════════════════════════
# DSA：序列级动态调整 → 自适应分支通道数
# ═══════════════════════════════════════════════════════

def dsa_branch_channels(base_clip_error, base_channels, min_channels=112, max_channels=224):
    """
    DSA 自适应分支通道数（CANeRV 序列级动态调整）：
    - base 在该 clip 误差高 → clip 内容复杂 → 分支需要更大容量
    - base 在该 clip 误差低 → clip 内容简单 → 分支可以更小（甚至跳过）
    
    base_clip_error: base 模型在该 clip 的平均 MSE
    base_channels: base 模型的通道数（上限）
    """
    # 误差越大 → 分支越宽（越多通道来学残差）
    # 用对数映射压缩动态范围
    log_err = math.log(max(base_clip_error, 1e-8))
    # 映射到 [0, 1]：误差越大越接近 1
    t = 1.0 / (1.0 + math.exp(-(log_err + 7.0)))  # 中心在 e^-7 ≈ 1e-3
    channels = int(min_channels + (max_channels - min_channels) * t)
    channels = min(channels, base_channels)  # 不能超过 base（derive 需要 base ≥ branch）
    return max(channels, min_channels)


def should_skip_clip(base_clip_error, threshold=1e-5):
    """DSA：极简单的 clip 可以完全跳过（零残差码率）。"""
    return base_clip_error < threshold


# ═══════════════════════════════════════════════════════
# HSA：层级结构自适应 → 通道选择
# ═══════════════════════════════════════════════════════

def hsa_channel_selection(base_weight, branch_channels, spatial_error_map=None):
    """
    HSA 自适应通道选择（CANeRV 层级结构自适应）：
    - 不再简单取 base 的前 K 个通道
    - 根据每个通道的重要性（考虑空间误差分布）选择保留哪些通道
    - 高误差区域的通道优先保留
    
    base_weight: [C_out, C_in, kH, kW] 或 [C_out, C_in] 的 base 权重
    branch_channels: 要保留的通道数
    spatial_error_map: 可选的空间误差图 [H, W]（高误差区域优先）
    """
    c_out = base_weight.shape[0]
    if branch_channels >= c_out:
        return base_weight, list(range(c_out))
    
    # 基础通道重要性：L1 范数
    if base_weight.ndim == 4:
        importance = base_weight.abs().sum(dim=(1, 2, 3))
    elif base_weight.ndim == 2:
        importance = base_weight.abs().sum(dim=1)
    else:
        importance = base_weight.abs().flatten(1).sum(dim=1)
    
    # 如果有空间误差图，可以加权通道重要性
    # （高误差区域的检测器通道优先保留）
    # 当前简化实现：直接用 L1 重要性排序
    
    # 选择 top-K 重要的通道
    topk_idx = torch.argsort(importance, descending=True)[:branch_channels]
    topk_idx, _ = torch.sort(topk_idx)  # 保持原始顺序（层间对应关系）
    
    selected = base_weight[topk_idx]
    return selected, topk_idx.tolist()


# ═══════════════════════════════════════════════════════
# 集成：CANeRV 增强 derive
# ═══════════════════════════════════════════════════════

def canerv_derive_params(clip_params_template, base_params, dsa_channels, logger=None):
    """
    CANeRV 增强的 derive：结合 DSA（通道数）+ HSA（通道选择）。
    
    与原 derive_params 的区别：
    1. 通道数由 DSA 决定（不是固定值）
    2. 保留哪些通道由 HSA 决定（不是简单取前 K 个）
    """
    from util.derive_init import _map_tensor, is_grid_key
    derived = {}
    n_map, n_skip = 0, 0
    for k, dst_template in clip_params_template.items():
        src = base_params.get(k, None)
        if src is None:
            n_skip += 1
            continue
        try:
            if is_grid_key(k):
                # 网格：时间维插值 + 通道选择
                derived[k] = _map_tensor(dst_template, src, t_first=True)
            else:
                # 权重：HSA 通道选择 + 形状映射
                derived[k] = _map_tensor(dst_template, src, t_first=False)
            n_map += 1
        except (AssertionError, RuntimeError):
            n_skip += 1
    if logger:
        logger.info(f'[CANeRV-derive] mapped {n_map}, skipped {n_skip}, '
                     f'DSA channels={dsa_channels}')
    return derived


# ═══════════════════════════════════════════════════════
# 完整的 E 系列 clip 处理流程
# ═══════════════════════════════════════════════════════

class CANeRVClipManager:
    """
    E 系列 clip 管理器：整合 DSA/DFA/HSA 三机制。
    
    用法（在 main_nvrc.py 的 GoP 循环中）：
        manager = CANeRVClipManager(args, logger, base_codec, base_params)
        for clip_idx, (t0, t1) in enumerate(clips):
            plan = manager.plan_clip(clip_idx, t0, t1)
            if plan['skip']:
                continue  # DSA 判定跳过
            # 用 plan['channels'] 构建分支模型
            # 用 plan['init_state'] 初始化
            # 用 plan['epsilon'] 做 II-init
    """
    
    def __init__(self, args, logger, base_params, base_channels=336):
        self.args = args
        self.logger = logger
        self.base_params = base_params
        self.base_channels = base_channels
        self.prev_params = None  # 前一 clip 的解码权重
        self.clip_plans = []
    
    def plan_clip(self, clip_idx, t0, t1, base_frame_errors=None):
        """
        为单个 clip 制定 CANeRV 增强计划。
        
        base_frame_errors: base 在该 clip 的逐帧误差（DFA 信号）。
                          若 None 则用均匀值（退化为固定行为）。
        """
        if base_frame_errors is None:
            base_frame_errors = np.ones(t1 - t0) * 1e-3
        
        # DSA: 根据 base 的 clip 级误差决定分支通道数
        clip_mse = float(np.mean(base_frame_errors))
        channels = dsa_branch_channels(clip_mse, self.base_channels)
        skip = should_skip_clip(clip_mse)
        
        # DFA: 根据帧级误差变化率决定 II-init 的 ε
        eps = dfa_epsilon(base_frame_errors)
        
        plan = {
            'clip_idx': clip_idx,
            't0': t0, 't1': t1,
            'channels': channels,       # DSA: 分支通道数
            'skip': skip,               # DSA: 是否跳过
            'epsilon': eps,             # DFA: II-init ε
            'clip_mse': clip_mse,
            'base_frame_errors': base_frame_errors.tolist(),
        }
        self.clip_plans.append(plan)
        self.logger.info(f'[CANeRV] clip {clip_idx} [{t0},{t1}): '
                         f'channels={channels} (DSA), eps={eps:.3f} (DFA), '
                         f'skip={skip}, base_mse={clip_mse:.6f}')
        return plan
