# -*- coding: utf-8 -*-
"""
UAR-NVC QEC 风格的最简熵模型（MyHiNeRV C 系列迁移）：
- 每张量一个可学习标量 q_step（log_step 参数化，继承基类机制）；
- 零均值高斯（无可学习 mean——对齐 UAR-NVC get_data_prob 的 mu=0 假设）；
- scale 可学习（对应 UAR-NVC scale=std(x_q) 的在线估计思想）；
- 量化退火 / rANS 编解码 / soft-round 全部继承 EntropyModel 基类模板。

用途：与 NVRC 原生上下文熵模型（ar / d-axis）做同预算对照——
"简单逐张量 QEC" vs "三层上下文熵模型" 的贡献分离。
"""
import torch

from .entropy_model_base import EntropyModel
from .utils import init_log_step, init_log_scales


class QECEntropyModel(EntropyModel):
    """逐张量 QEC：learnable log_step + learnable log_scales，means 恒为 0。"""

    def _get_params(self):
        """只保留 log_step 与 log_scales（去掉可学习 means——零均值假设）。"""
        params = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            meta_shape = tensor_cfg['meta_shape']
            if self.learned_quant:
                params[f'{tensor_k}_log_step'] = init_log_step(meta_shape, value=-6.9)  # e^-6.9≈0.001（UAR-NVC 初值）
            if self.learned_em:
                params[f'{tensor_k}_log_scales'] = init_log_scales(meta_shape, scales=1.0)
        return params

    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        """零均值：means=0，scales=exp(log_scales)（逐张量标量广播，结构对齐基类）。"""
        if self.learned_em:
            means = []
            scales = []
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_scales = params[f'{tensor_k}_log_scales'].exp()
                tensor_means = torch.zeros_like(tensor_scales)
                means.append(tensor_means.unsqueeze(0).expand(tensor_cfg['reshape']))
                scales.append(tensor_scales.unsqueeze(0).expand(tensor_cfg['reshape']))
            means = torch.cat(means, dim=2)
            scales = torch.cat(scales, dim=2)
        else:
            means = torch.zeros_like(meta_i['scales']).float().expand(-1, -1, self.length)
            scales = meta_i['scales'].float().expand(-1, -1, self.length)
        return means, scales
