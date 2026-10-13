# -*- coding: utf-8 -*-
"""
C5 渐进剪枝器 v2：修复 v1 中"候选集为空 → 阈值崩溃 → 99.4% 全量剪除"的 bug。
"""
import torch


class ProgressivePruner:
    def __init__(self, logger, target_sparsity=0.15, start_epoch=60,
                 end_epoch=300, prune_every=10, exclude_keys=None):
        self.logger = logger
        self.target = float(target_sparsity)
        self.start = int(start_epoch)
        self.end = int(end_epoch)
        self.every = int(prune_every)
        self.exclude = set(exclude_keys or [])
        self.masks = None
        self.current_sparsity = 0.0
        self._prune_count = 0

    def _compute_importance(self, w):
        if w.ndim < 2:
            return w.abs()
        fan_in = max(w.shape[1], 1)
        return w.abs() / (fan_in ** 0.5)

    def _init_masks(self, params):
        self.masks = {}
        for k, v in params.items():
            if k in self.exclude or not torch.is_tensor(v) or v is None:
                continue
            self.masks[k] = torch.ones(v.shape, dtype=torch.float32, device=v.device)

    def step(self, epoch, params):
        if self.masks is None:
            self._init_masks(params)
        if epoch < self.start or epoch >= self.end:
            return False
        if (epoch - self.start) % self.every != 0:
            return False

        progress = (epoch - self.start) / max(self.end - self.start, 1)
        target_now = self.target * progress
        newly_pruned = 0
        total_params = 0
        total_zeroed = 0

        with torch.no_grad():
            for k, w in params.items():
                if k in self.exclude or k not in self.masks or not torch.is_tensor(w):
                    continue
                n = w.numel()
                total_params += n
                n_target = int(n * target_now)
                n_cur_zero = int((self.masks[k].flatten() == 0).sum().item())
                n_add = n_target - n_cur_zero
                if n_add <= 0:
                    total_zeroed += n_cur_zero
                    continue

                score = self._compute_importance(w.detach().float()).flatten()
                active = self.masks[k].flatten() > 0
                active_scores = score[active]
                n_active = len(active_scores)
                if n_active == 0:
                    total_zeroed += n_cur_zero
                    continue

                n_to_prune = min(n_add, n_active)
                threshold = torch.kthvalue(active_scores, n_to_prune).values
                mask_flat = self.masks[k].flatten().clone()
                to_zero = active & (score <= threshold)
                mask_flat[to_zero] = 0.0
                self.masks[k] = mask_flat.reshape(self.masks[k].shape)
                actually = int(to_zero.sum().item())
                newly_pruned += actually
                total_zeroed += n_cur_zero + actually

        self.current_sparsity = total_zeroed / max(total_params, 1)
        self._prune_count += 1
        if self.logger:
            self.logger.info(f'[Pruner] step {self._prune_count} ep{epoch}: '
                             f'+{newly_pruned} → sparsity {self.current_sparsity:.4f} '
                             f'(target {target_now:.4f})')
        return True

    def apply(self, params):
        if self.masks is None:
            return params
        return {k: (v * self.masks[k].to(dtype=v.dtype, device=v.device)
                    if k in self.masks and torch.is_tensor(v) else v)
                for k, v in params.items()}

    def get_stats(self):
        if self.masks is None:
            return {'sparsity': 0.0, 'steps': 0}
        total = sum(m.numel() for m in self.masks.values())
        zeros = sum((m == 0).sum().item() for m in self.masks.values())
        return {'sparsity': zeros / max(total, 1), 'steps': self._prune_count}
