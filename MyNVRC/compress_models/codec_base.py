"""
Base class for compression models（压缩模型基类：统一“参数量化 + 码率估计 + 模型前向”的接口）
"""
from .utils import *


class CodecBase(FuncModule):
    """
    Codec base class（编解码器基类）

    forward() 是训练时的统一入口：
        - compute_outputs=True  -> 用量化后参数 params 驱动 INR 主干 self.model(x, params) 重建图像；
        - compute_rates=True    -> 返回估计码率 rate 与率损失 rate_loss；
        - cache 机制：load_cache() 缓存一次 (rates, params)，评估时多个 patch 复用，避免重复熵前向。
    """
    cache_enabled = True
    cached = False
    rates_cache = None
    params_cache = None
    def _forward_rates_params(self, x=None, forward_rates=True, forward_params=True):
        """
        Forward parameters and rates
        （子类实现：返回 (码率, 量化后参数)）
        """
        raise NotImplementedError

    def _get_rate(self, rates):
        """
        Get rate
        （总码率）
        """
        raise NotImplementedError

    def _get_rate_loss(self, rates):
        """
        Get rate loss
        （用于反向传播的率损失）
        """
        raise NotImplementedError

    def forward(self, x, compute_outputs=True, compute_rates=True, compute_params=False, reduce_rates=True):
        """
        Forward pass
        （前向：按需计算重建输出 / 码率 / 量化参数；D-step 只开 outputs，R-step 只开 rates）
        """
        assert compute_outputs or compute_rates or compute_params, \
            'At least one of get_outputs and get_rates should be True'

        all_outputs = []

        # Compute params and rates（未缓存则前向熵模型；已缓存则直接复用）
        if not self.cached:
            rates, params = self._forward_rates_params(x, forward_rates=compute_rates,
                                                       forward_params=compute_outputs or compute_params)
        else:
            rates = self.rates_cache
            params = self.params_cache

        # Outputs
        if compute_outputs:
            outputs = self.model(x, params)
            all_outputs.append(outputs)

        # Rates
        if compute_rates:
            rate = self._get_rate(rates, reduce_rates)
            rate_loss = self._get_rate_loss(rates, reduce_rates)
            all_outputs.append(rate)
            all_outputs.append(rate_loss)

        # Params
        if compute_params:
            all_outputs.append(params)
        
        return tuple(all_outputs) if len(all_outputs) > 1 else all_outputs[0]

    def load_cache(self):
        """
        Load parameters and rate cache
        （缓存量化参数与码率：熵模型对同一权重只需前向一次）
        """
        if self.cache_enabled:
            with torch.no_grad():
                self.rates_cache, _, self.params_cache = self(None, compute_outputs=False, compute_rates=True,
                                                              compute_params=True, reduce_rates=False)
                self.cached = True

    def clear_cache(self):
        """
        Clear parameters and rate cache
        """
        self.rates_cache, self.params_cache = None, None
        self.cached = False

    def compress(self, bs):
        """
        Compress
        """
        raise NotImplementedError

    def decompress(self, bs):
        """
        Decompress
        """
        raise NotImplementedError

    def get_rate_stat(self):
        """
        Get rate statistics
        """
        raise NotImplementedError

    def set_epoch(self, epoch, num_epochs):
        """
        Set epoch
        """
        raise NotImplementedError
    
    def set_iter(self, iter, num_iters):
        """
        Set iter
        """
        raise NotImplementedError