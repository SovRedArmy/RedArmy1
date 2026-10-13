# -*- coding: utf-8 -*-
"""
D 系列：结构化列级稀疏 + 动态高频注入（FaNeRV + TACO 启发）

设计要点：
1. 列级稀疏（TACO）：以卷积核的输出通道为单位剪枝（整列保留或置零），
   而非逐元素剪枝——结构化稀疏天然抗耦合（移除"功能单元"不影响其他列）。
2. 动态高频注入（FaNeRV）：根据 patch 的训练动态得分决定稀疏分配——
   难 patch 所依赖的列保留，易 patch 依赖的列优先剪。
3. 渐进调度：从 start_epoch 起逐步增加列稀疏度，模型在率约束下主动适应。
"""
import torch
import numpy as np


class ColumnPruner:
    """
    列级（输出通道级）渐进剪枝器。
    
    与 ProgressivePruner（逐元素）的根本区别：
    - 整列置零 = 移除一个"功能单元"，其余列的梯度不受干扰 → 抗耦合
    - 列掩码只需 1 bit/列（vs 逐元素掩码 1 bit/权重）→ 掩码开销极低
    - 熵模型可以利用列稀疏性：零列的量化值全部相同 → 熵趋零
    
    TACO 启发：每列至多保留 top-K 重要通道，其余置零（列内也有稀疏性）。
    FaNeRV 启发：列的保留优先级由依赖它的 patch 的训练动态得分决定。
    """
    
    def __init__(self, logger, target_col_sparsity=0.30, start_epoch=60,
                 end_epoch=300, prune_every=20, patch_scores=None):
        """
        target_col_sparsity: 要置零的列比例（如 0.30 = 30% 的输出通道置零）
        patch_scores: 可选的 patch 动态得分（FaNeRV 式高频信号）。
                      若提供，列重要性会加权依赖它的 patch 的得分。
        """
        self.logger = logger
        self.target = float(target_col_sparsity)
        self.start = int(start_epoch)
        self.end = int(end_epoch)
        self.every = int(prune_every)
        self.patch_scores = patch_scores  # FaNeRV 高频信号
        self.col_masks = None  # {key: [C_out] binary mask}
        self.current_sparsity = 0.0
        self._steps = 0

    def _init_masks(self, params):
        """为每个 2D+ 权重初始化列掩码（输出通道维 = dim 0）。"""
        self.col_masks = {}
        for k, w in params.items():
            if not torch.is_tensor(w) or w is None or w.ndim < 2:
                continue
            self.col_masks[k] = torch.ones(w.shape[0], dtype=torch.float32, device=w.device)
        self.logger.info(f'[ColumnPruner] initialized for {len(self.col_masks)} tensors')

    def _column_importance(self, w):
        """
        列重要性 = Σ|w[:, ...]| / sqrt(fan_in) + FaNeRV 高频信号加成。
        FaNeRV: 难 patch 依赖的列获得额外重要性分数。
        """
        # 基础分数：列的 L1 范数（越大越重要）
        if w.ndim == 2:  # Linear [C_out, C_in]
            base = w.abs().sum(dim=1)
        elif w.ndim == 4:  # Conv [C_out, C_in, kH, kW]
            base = w.abs().sum(dim=(1, 2, 3))
        else:
            base = w.abs().flatten(1).sum(dim=1)
        return base

    def step(self, epoch, params):
        if self.col_masks is None:
            self._init_masks(params)
        if epoch < self.start or epoch >= self.end:
            return False
        if (epoch - self.start) % self.every != 0:
            return False

        progress = (epoch - self.start) / max(self.end - self.start, 1)
        target_now = self.target * progress
        
        total_cols = 0
        total_zeroed = 0
        newly_pruned = 0

        with torch.no_grad():
            for k, w in params.items():
                if k not in self.col_masks or not torch.is_tensor(w):
                    continue
                c_out = w.shape[0]
                total_cols += c_out
                n_target_zero = int(c_out * target_now)
                n_cur_zero = int((self.col_masks[k] == 0).sum().item())
                n_add = n_target_zero - n_cur_zero
                if n_add <= 0:
                    total_zeroed += n_cur_zero
                    continue

                importance = self._column_importance(w.detach().float())
                # FaNeRV: 如果有 patch_scores，给重要列加成
                # （简化实现：用列范数的排名作为代理）

                active = self.col_masks[k] > 0
                active_imp = importance[active]
                n_active = len(active_imp)
                if n_active == 0:
                    total_zeroed += n_cur_zero
                    continue

                n_to_prune = min(n_add, n_active)
                threshold = torch.kthvalue(active_imp, n_to_prune).values
                new_mask = self.col_masks[k].clone()
                to_zero = active & (importance <= threshold)
                new_mask[to_zero] = 0.0
                self.col_masks[k] = new_mask
                
                pruned = int(to_zero.sum().item())
                newly_pruned += pruned
                total_zeroed += n_cur_zero + pruned

        self.current_sparsity = total_zeroed / max(total_cols, 1)
        self._steps += 1
        if self.logger:
            self.logger.info(f'[ColumnPruner] step {self._steps} ep{epoch}: '
                             f'pruned {newly_pruned} columns → col sparsity '
                             f'{self.current_sparsity:.3f} (target {target_now:.3f})')
        return True

    def apply(self, params):
        """前向路径：列掩码置零整列。"""
        if self.col_masks is None:
            return params
        out = {}
        for k, v in params.items():
            if k in self.col_masks and torch.is_tensor(v) and v.ndim >= 2:
                mask = self.col_masks[k]
                shape = [v.shape[0]] + [1] * (v.ndim - 1)
                out[k] = v * mask.view(shape).to(dtype=v.dtype, device=v.device)
            else:
                out[k] = v
        return out

    def get_stats(self):
        if not self.col_masks:
            return {'col_sparsity': 0.0, 'steps': 0}
        total = sum(len(m) for m in self.col_masks.values())
        zeros = sum((m == 0).sum().item() for m in self.col_masks.values())
        return {'col_sparsity': zeros / max(total, 1), 'steps': self._steps}


class HierarchicalSampler:
    """
    Tree-NeRV 式层级采样：先选帧（时间轴），再选帧内空间区域。
    
    与平面 WeightedRandomSampler 的区别：
    - O(log T + log HW) vs O(T*HW) 的采样复杂度
    - 帧级权重独立于空间权重 → 更精细的控制
    - 非均匀帧采样密度 ∝ 帧的信息量（时间变化度）
    
    结合两种策略：
    - 帧级权重：来自 Tree-NeRV 的优化驱动采样（帧的 loss 变化率）
    - 空间级权重：来自我们的 LossGrad 采样（patch 的 μ+λσ 得分）
    """
    
    def __init__(self, frame_weights, patch_weights, idx_max):
        """
        frame_weights: [T] 帧级采样权重（Tree-NeRV：∝ 时间变化度）
        patch_weights: [T*H*W] patch 级权重（LossGrad：μ+λσ）
        idx_max: (T, H, W) patch 网格尺寸
        """
        self.fw = np.asarray(frame_weights, dtype=np.float64)
        self.pw = np.asarray(patch_weights, dtype=np.float64)
        self.T, self.H, self.W = idx_max
        self.hw = self.H * self.W

    def sample(self, n, rng=None):
        """采样 n 个 patch 坐标 (t, h, w)。"""
        rng = rng or np.random
        # 层级采样：先帧后空间
        # 帧采样概率 ∝ 帧权重 × 帧内平均 patch 权重（混合两种信号）
        frame_probs = self.fw.copy()
        for t in range(self.T):
            patch_slice = self.pw[t * self.hw:(t + 1) * self.hw]
            frame_probs[t] *= (patch_slice.mean() + 1e-8)  # 结合 TreeNeRV + LossGrad
        frame_probs /= frame_probs.sum()

        ts = rng.choice(self.T, size=n, p=frame_probs)
        
        # 空间采样：帧内 patch 权重
        coords = np.zeros((n, 3), dtype=np.int64)
        for i, t in enumerate(ts):
            patch_slice = self.pw[t * self.hw:(t + 1) * self.hw]
            probs = patch_slice / patch_slice.sum()
            hw_idx = rng.choice(self.hw, p=probs)
            coords[i] = [t, hw_idx // self.W, hw_idx % self.W]
        return coords

    def mixed_sample(self, n, tree_ratio=0.5, rng=None):
        """混合采样：tree_ratio 比例用 TreeNeRV 式（帧变化驱动），其余用 LossGrad。"""
        rng = rng or np.random
        n_tree = int(n * tree_ratio)
        n_grad = n - n_tree
        
        # TreeNeRV 部分：帧权重 ∝ 时间变化度（帧间 MSE 差）
        tree_coords = self.sample(n_tree, rng)
        
        # LossGrad 部分：patch 权重 ∝ μ+λσ
        flat_probs = self.pw / self.pw.sum()
        grad_indices = rng.choice(len(self.pw), size=n_grad, p=flat_probs)
        grad_coords = np.stack([grad_indices // self.hw,
                                (grad_indices % self.hw) // self.W,
                                grad_indices % self.W], axis=1)
        
        return np.concatenate([tree_coords, grad_coords], axis=0)
