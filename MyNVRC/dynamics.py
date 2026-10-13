# -*- coding: utf-8 -*-
"""
训练动态加权系统（自 MyHiNeRV/dynamics.py 迁移，适配 NVRC 的扁平 patch 索引）。

三个作用（与 MyHiNeRV B 系列一致）：
1. PatchDynamicsTracker：逐 patch 滑动窗口跟踪 L1 损失，得分 = μ + λ·σ（dynamicAL）；
2. compute_patch_train_weights：激进采样权重 [w_min, w_max]（温度软化，无丢弃）；
3. compute_patch_loss_weights：保守损失权重 α·(score 标准化)+1（夹在 [1-δ, 1+δ]）；
4. cosine_window_factor：余弦窗口（早晚期均匀、中期差异化），支持 "p0,p1,p2,p3"。

NVRC 索引约定（datasets.VideoSubset，t 优先展平）：
    flat_idx = t*(idx_max[1]*idx_max[2]) + h*idx_max[2] + w
权重向量按此顺序排列；本文件所有接口都以 (t,h,w) 或 flat_idx 对齐该约定。
"""
import numpy as np


class PatchDynamicsTracker:
    """逐 patch 损失的滑动均值/方差跟踪（指数滑动窗实现，内存 O(P)。）"""

    def __init__(self, num_patches, window=10, var_weight=2.0):
        self.num_patches = int(num_patches)
        self.window = float(window)
        self.var_weight = float(var_weight)
        self.mean = np.zeros(self.num_patches, dtype=np.float64)
        self.var = np.zeros(self.num_patches, dtype=np.float64)
        self.cnt = np.zeros(self.num_patches, dtype=np.float64)

    def update(self, flat_idx, losses):
        """flat_idx: [N] int 数组；losses: [N] float 数组（本 batch 每个 patch 的标量损失）。"""
        flat_idx = np.asarray(flat_idx, dtype=np.int64)
        losses = np.asarray(losses, dtype=np.float64)
        alpha = 2.0 / (self.window + 1.0)
        m_old, v_old = self.mean[flat_idx], self.var[flat_idx]
        m_new = (1 - alpha) * m_old + alpha * losses
        self.mean[flat_idx] = m_new
        self.var[flat_idx] = (1 - alpha) * v_old + alpha * (losses - m_new) ** 2
        self.cnt[flat_idx] += 1

    def get_scores(self):
        """得分 = μ + λ·σ（未被观测过的 patch 得分记 0，首轮训练后自然参与排序）。"""
        return self.mean + self.var_weight * np.sqrt(np.maximum(self.var, 0))

    def get_frame_scores(self, idx_max):
        """聚合为逐帧得分（TeNeRV 式内容/难度参照，供 GOP 切分或诊断）。"""
        t_max, h_max, w_max = idx_max
        s = self.get_scores().reshape(t_max, h_max * w_max)
        return s.mean(axis=1)


def compute_patch_train_weights(scores, temp=1.0, w_min=0.5, w_max=2.0):
    """激进采样权重：softmax(scores/temp) 线性映射到 [w_min, w_max]，全部 > 0（无丢弃）。"""
    s = np.asarray(scores, dtype=np.float64)
    if temp <= 0:
        temp = 1e-8
    z = (s - s.mean()) / (s.std() + 1e-8) / temp
    z = np.clip(z, -4, 4)
    w = (z - z.min()) / (z.max() - z.min() + 1e-8)
    return w_min + (w_max - w_min) * w


def compute_patch_loss_weights(scores, alpha=0.15, clip_delta=0.2):
    """保守损失权重：1 + α·(score 标准化)，夹在 [1-δ, 1+δ]（降低 batch 内损失方差）。"""
    s = np.asarray(scores, dtype=np.float64)
    z = (s - s.mean()) / (s.std() + 1e-8)
    z = np.clip(z, -3, 3)
    return 1.0 + np.clip(alpha * z, -clip_delta, clip_delta)


def cosine_window_factor(progress, window):
    """余弦窗口因子：progress ∈ [0,1]；window = "p0,p1,p2,p3"（升/平台/降）。
    p0 前与 p3 后为 0（均匀采样），p1..p2 平台为 1（全量差异化）。"""
    if not window:
        return 1.0
    p0, p1, p2, p3 = [float(x) for x in window.split(',')]
    if progress <= p0 or progress >= p3:
        return 0.0
    if progress < p1:
        return 0.5 * (1.0 - np.cos(np.pi * (progress - p0) / max(p1 - p0, 1e-6)))
    if progress <= p2:
        return 1.0
    return 0.5 * (1.0 + np.cos(np.pi * (progress - p2) / max(p3 - p2, 1e-6)))


def apply_window(weights, base_w, progress, window):
    """把差异化权重与均匀权重按余弦因子混合：w = f·w_diff + (1-f)·uniform。"""
    f = cosine_window_factor(progress, window)
    return f * weights + (1.0 - f) * base_w


def flat_index(idx_thw, idx_max):
    """(t,h,w) 坐标 → NVRC 扁平 patch 下标（t 优先，与 VideoSubset.indices 一致）。"""
    t, h, w = idx_thw[..., 0], idx_thw[..., 1], idx_thw[..., 2]
    return (t * idx_max[1] * idx_max[2] + h * idx_max[2] + w).astype(np.int64)
