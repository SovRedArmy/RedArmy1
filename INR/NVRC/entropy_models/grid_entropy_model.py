"""
Grid entropy model classes（特征网格熵模型：把 3D 特征网格切成 3D 块并建模）

    GridEntropyModel     —— 分块独立模型：网格按 (T, H, W) 切块，每块每通道学习
                            (log_step, mean, log_scale)（meta_shape = (num_blks, 1, C)）；
    ARGridEntropyModel   —— 自回归上下文模型（论文默认）：每个网格配一个掩码 3D 卷积网络，
                            块内按光栅顺序逐元素编码，用已解码邻居预测当前元素的高斯参数；
                            训练时（compress_mode='none'）一次性并行计算所有位置的参数。
"""
from .utils import *
from .layers import FuncMaskedConv3d, FuncMaskedConv3d
from .weight_entropy_model import WeightEntropyModel


class GridEntropyModel(WeightEntropyModel):
    """（3D 分块网格熵模型：继承 2D 权重模型的流程，仅把切块/重排扩展到 3D 并按网格逐个处理）"""
    def _build_cfgs(self, inputs):
        """（为每个 4D 网格 [T, H, W, C] 计算 3D 块划分、padding 与逐块配置）"""
        assert len(self.blk_size) == 3 and all(s > 0 for s in self.blk_size)

        tensor_cfgs = {}

        with torch.no_grad():
            for tensor_k, tensor in self._sort_tensors(inputs):
                assert tensor.ndim == 4, 'grids should be 4D/5D (T, H, W, C)/(N, T, H, W, C)'
                tensor = tensor.unsqueeze(0)
                reshape = (-1, *tensor.shape[1:])
                padding = tuple((p - s % p) % p for s, p in zip(reshape[1:-1], self.blk_size))
                reshape_pad = (reshape[0],) + tuple(reshape[1:-1][d] + padding[d] for d in range(3)) + (reshape[-1],)
                num_blks = tuple(s // p for s, p in zip(reshape_pad[1:-1], self.blk_size))
                length = math.prod(num_blks)
                tensor_cfgs[tensor_k] = {}
                tensor_cfgs[tensor_k]['shape'] = (-1, *tensor.shape[1:])
                tensor_cfgs[tensor_k]['mask_shape'] = (1, *tensor.shape[1:])
                tensor_cfgs[tensor_k]['reshape'] = reshape
                tensor_cfgs[tensor_k]['reshape_pad'] = reshape_pad
                tensor_cfgs[tensor_k]['reshape_blk'] = (-1, length, math.prod(self.blk_size), tensor.shape[-1])
                tensor_cfgs[tensor_k]['padding'] =  (0, 0, 0, padding[2], 0, padding[1], 0, padding[0])
                tensor_cfgs[tensor_k]['unpadding'] = (0, 0, 0, -padding[2], 0, -padding[1], 0, -padding[0])
                tensor_cfgs[tensor_k]['num_blks'] = num_blks
                tensor_cfgs[tensor_k]['offset'] = 0
                tensor_cfgs[tensor_k]['length'] = length
                tensor_cfgs[tensor_k]['meta_shape'] = (length, 1, tensor.shape[-1])

        return tensor_cfgs

    def _build_forward_order(self, inputs):
        """（多个网格（多尺度）按注册顺序逐个处理，互不共享参数）"""
        return self.tensor_keys

    def _build_mask(self):
        """（构建每个网格的有效位掩码（经块重排），padding 元素不计码率）"""
        masks = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            masks[tensor_k] = self._reshape_tensor(tensor_k, torch.ones(tensor_cfg['mask_shape'], dtype=torch.bool))
        return masks

    def _reshape_tensor(self, tensor_k, tensor):
        """（重排：[T, H, W, C] -> padding -> 切成 3D 块堆叠 [M, T_b*H_b*W_b, C]）"""
        tensor_cfg = self.tensor_cfgs[tensor_k]
        tensor = tensor.unsqueeze(0)
        tensor = F.pad(tensor.view(tensor_cfg['reshape']), tensor_cfg['padding'])
        tensor = tensor.view(tensor.shape[0], tensor_cfg['num_blks'][0], self.blk_size[0], 
                             tensor_cfg['num_blks'][1], self.blk_size[1],
                             tensor_cfg['num_blks'][2], self.blk_size[2],
                             tensor.shape[-1]) \
                       .permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(tensor_cfg['reshape_blk'])
        return tensor

    def _restore_tensor(self, tensor_k, tensor):
        """（_reshape_tensor 的逆变换：块 -> 网格 -> 去 padding）"""
        tensor_cfg = self.tensor_cfgs[tensor_k]
        tensor = tensor.view(tensor.shape[0], *tensor_cfg['num_blks'], *self.blk_size, tensor.shape[-1]) \
                       .permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(tensor_cfg['reshape_pad'])
        tensor = F.pad(tensor, tensor_cfg['unpadding']).view(tensor_cfg['shape'])
        tensor = tensor.squeeze(0)
        return tensor

    def _preprocess(self, inputs):
        """（逐网格重排成块形式（不拼接，各网格独立处理））"""
        outputs = {}
        for tensor_k, tensor in self._sort_tensors(inputs):
            outputs[tensor_k] = self._reshape_tensor(tensor_k, tensor)
        return outputs

    def _postprocess(self, inputs):
        """（逐网格还原形状）"""
        outputs = {}
        for tensor_k, tensor in self._sort_tensors(inputs):
            outputs[tensor_k] = self._restore_tensor(tensor_k, tensor)
        return outputs

    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        """（待编码符号即整个块化网格，附有效位掩码）"""
        return x, getattr(self, f'{k}_mask').expand_as(x)
    
    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        """（熵参数：每块每通道常数 mean/scale（meta_shape=(M,1,C)），广播到块内元素）"""
        if self.learned_em:
            tensor_means = params[k + f'_means']
            tensor_scales = params[k + f'_log_scales'].exp()
            tensor_means = tensor_means.unsqueeze(0)
            tensor_scales = tensor_scales.unsqueeze(0)
            means = getattr(self, f'{k}_mask') * tensor_means
            scales = getattr(self, f'{k}_mask') * tensor_scales
        else:
            means = getattr(self, f'{k}_mask') * meta_i['means'].float()
            scales = getattr(self, f'{k}_mask') * meta_i['scales'].float()
        return means, scales

    def _get_log_step(self, k, i, params):
        """（量化步长：每块每通道一个学习值广播到块内；未学习时固定 delta=2^-4）"""
        if self.learned_quant:
            tensor_log_step = params[k + '_log_step']
            tensor_log_step = tensor_log_step.unsqueeze(0)
            return tensor_log_step.expand(self.tensor_cfgs[k]['reshape_blk'])
        else:
            return -4.


class ARGridEntropyModel(GridEntropyModel):
    """
    （自回归网格熵模型——论文的特征网格上下文模型：
      每个网格一组 depth 层掩码 3D 卷积（首层 mask_cur 屏蔽当前及未来位置），
      输出 2C 通道 = 每通道的 (mean, log_scale)；
      learned_em 被强制为 False，即参数全部来自上下文网络而非逐块学习。）
    """
    def __init__(self, logger, name, inputs, config):
        super().__init__(logger, name, inputs, config)
        self.channels = config.channels
        self.depth = config.depth
        self.kernel_size = config.kernel_size
        self.norm = config.norm
        self.bias = config.bias
        self.act = config.act
        self.depthwise = config.depthwise
        self.learned_em = False

        for k, cfg in self._sort_tensors(self.tensor_cfgs):
            groups_k =  cfg['shape'][-1] if self.depthwise else 1
            channels_k = groups_k * self.channels
            layers = []
            for n in range(self.depth):
                layers.append(
                    FuncMaskedConv3d(cfg['shape'][-1] if n == 0 else channels_k,
                                     cfg['shape'][-1] * 2 if n == self.depth - 1 else channels_k,
                                     self.kernel_size, groups=groups_k, mask_cur=(n == 0), bias=self.bias,
                                     is_head=(n == self.depth - 1))
                )
                if n != self.depth - 1:
                    layers.append(get_func_activation(self.act)())
                    layers.append(get_func_norm(self.norm)(channels_k))
            self.add_module(f'{k}_layers', nn.ModuleList(layers))

    def extra_repr(self):
        s = super().extra_repr() + ', kernel_size={kernel_size}, channels={channels}, depth={depth}, norm={norm}'
        s += ', bias={bias}, act={act}, depthwise={depthwise}'
        return s.format(**self.__dict__)

    def _num_iters(self, k):
        """（扫描步数：训练并行 1 步；实际编解码时 = 块内元素数 T_b*H_b*W_b，逐步串行）"""
        if self.compress_mode in ['none']:
            return 1
        elif self.compress_mode in ['encoding', 'decoding']:
            return math.prod(self.blk_size)
        else:
            raise ValueError(f'Unknown compress mode {self.compress_mode}')

    def _get_ctx_i_loc(self, k, i):
        """（把一维扫描序号 i 映射为块内 3D 坐标 (t, h, w)——光栅扫描顺序）"""
        t, h, w = (i // (self.blk_size[1] * self.blk_size[2]),
                   (i % (self.blk_size[1] * self.blk_size[2])) // self.blk_size[2],
                   (i % (self.blk_size[1] * self.blk_size[2])) % self.blk_size[2])
        return t, h, w

    def _get_ctx_i_range(self, k, i):
        """（第 i 步可用的上下文邻域范围：时间维只看过去，空间维前后都看）"""
        offset = self.depth * (self.kernel_size // 2)
        t, h, w = self._get_ctx_i_loc(k, i)
        t_start, t_end = (max(t - offset, 0), t)
        h_start, h_end = (max(h - offset, 0), min(h + offset, self.blk_size[1] - 1))
        w_start, w_end = (max(w - offset, 0), min(w + offset, self.blk_size[2] - 1))
        return (t_start, t_end + 1), (h_start, h_end + 1), (w_start, w_end + 1)

    def _get_ctx_i_padding(self, k, i):
        """（把邻域补齐到固定感受野大小所需的 F.pad 参数）"""
        offset = self.depth * (self.kernel_size // 2)
        t, h, w = self._get_ctx_i_loc(k, i)
        (t_start, _), (h_start, h_end), (w_start, w_end) = self._get_ctx_i_range(k, i)
        paddings = (0, 0, max(offset - (w - w_start), 0), max(offset - (w_end - 1 - w), 0),
                    max(offset - (h - h_start), 0), max(offset - (h_end - 1 - h), 0),
                    max(offset - (t - t_start), 0), 0)
        return paddings

    def _get_ctx_i_size(self, k, i):
        """（固定上下文窗口尺寸：(offset+1, 2*offset+1, 2*offset+1)）"""
        offset = self.depth * (self.kernel_size // 2)
        size = (offset * 1 + 1, offset * 2 + 1, offset * 2 + 1)
        return size

    def _init_ctx(self, k, x, params):
        """（初始化上下文：训练时直接用完整 x（并行）；编解码时用全零张量逐步填充）"""
        tensor_mask = getattr(self, f'{k}_mask')

        if self.compress_mode in ['none']:
            return x, None
        elif self.compress_mode in ['encoding', 'decoding']:
            return torch.zeros_like(tensor_mask, dtype=torch.float32).float(), tensor_mask.clone()
        else:
            raise ValueError(f'Unknown compress mode {self.compress_mode}')

    def _extract_ctx(self, k, i, ctx, ctx_mask, params):
        """（取出第 i 步的上下文局部窗口并 pad 到固定大小；训练时返回完整 ctx）"""
        if self.compress_mode in ['none']:
            return ctx, ctx_mask
        elif self.compress_mode in ['encoding', 'decoding']:
            N, M = ctx.shape[:2]
            C = ctx.shape[-1]
            (t_start, t_end), (h_start, h_end), (w_start, w_end) = self._get_ctx_i_range(k, i)
            shape = (N, M, *self.blk_size, C)
            padding = self._get_ctx_i_padding(k, i)
            ctx_i = F.pad(ctx.view(*shape)[:, :, t_start:t_end, h_start:h_end, w_start:w_end, :], padding) \
                     .contiguous().view(N, M, -1, C)
            ctx_mask_i = F.pad(ctx_mask.view(*shape)[:, :, t_start:t_end, h_start:h_end, w_start:w_end, :], padding) \
                          .contiguous().view(N, M, -1, C)
            return ctx_i, ctx_mask_i
        else:
            raise ValueError(f'Unknown compress mode {self.compress_mode}')

    def _fill_ctx(self, k, i, ctx, ctx_mask, x_i, params):
        """（把第 i 步解码出的值 x_i 写回上下文张量的对应位置 (t, h, w)）"""
        if self.compress_mode in ['none']:
            return x_i, ctx_mask
        elif self.compress_mode in ['encoding', 'decoding']:
            N, M = ctx.shape[:2]
            C = ctx.shape[-1]
            t, h, w = self._get_ctx_i_loc(k, i)
            ctx.view(N, M, *self.blk_size, C)[:, :, t, h, w, :] = x_i
            return ctx, ctx_mask
        else:
            raise ValueError(f'Unknown compress mode {self.compress_mode}')

    def _init_meta(self, k, x, params):
        """（无边信息：熵参数完全由上下文网络生成）"""
        meta = {}
        return meta

    def _extract_meta(self, k, i, meta):
        meta_i = {}
        return meta_i

    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        """（取出第 i 步待编码的符号（块内 (t,h,w) 位置的所有块与通道）；训练时返回全部）"""
        tensor_mask = getattr(self, f'{k}_mask')

        if self.compress_mode in ['none']:
            return x, tensor_mask.clone().expand_as(x)
        elif self.compress_mode in ['encoding', 'decoding']:
            N, M = ctx.shape[:2]
            C = ctx.shape[-1]
            t, h, w = self._get_ctx_i_loc(k, i)
            x_i = x.view(N, M, *self.blk_size, C)[:, :, t, h, w, :]
            x_i_mask = tensor_mask.view(1, M, *self.blk_size, C)[:, :, t, h, w, :]
            return x_i, x_i_mask.expand_as(x_i)
        else:
            raise ValueError(f'Unknown compress mode {self.compress_mode}')

    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        """
        （核心：用掩码 3D 卷积网络从上下文预测高斯参数——
          训练时一次并行处理整块（Teacher Forcing）；编解码时只取窗口中心位置
          （即当前扫描位置）的预测值。输出经 exp 得到 scales，means/scales 均被
          量化步长缩放到与符号同一尺度，且用有效位掩码屏蔽 padding 元素。）
        """
        if self.compress_mode in ['none']:
            N, M = ctx_i.shape[:2]
            T, H, W = self.blk_size
            C = ctx_i.shape[-1]
        elif self.compress_mode in ['encoding', 'decoding']:
            N, M = ctx_i.shape[:2]
            T, H, W = self._get_ctx_i_size(k, i)
            C = ctx_i.shape[-1]
        else:
            raise ValueError(f'Unknown compress mode {self.compress_mode}')

        # Spatial mask is used here to ensure consistent between forward pass and coding
        f = ctx_i.view(N * M, T, H, W, C)
        if ctx_mask_i is not None:
            f_mask = ctx_mask_i.view(N * M, T, H, W, C).any(dim=-1, keepdim=True)
        else:
            f_mask = None

        # Layers
        f = run_modules(getattr(self, f'{k}_layers'), f, params, f_mask)

        # Extract means and scales
        means, scales = f.view(N, M, T, H, W, C * 2).chunk(2, dim=-1)
        means = means.contiguous()
        scales = scales.exp().contiguous()

        # Extract mask
        tensor_mask = getattr(self, f'{k}_mask').view(1, M, *self.blk_size, C)

        if self.compress_mode in ['none']:
            mask = tensor_mask
            return (mask * means).view(N, M, math.prod(self.blk_size), C), \
                   (mask * scales).view(N, M, math.prod(self.blk_size), C)
        elif self.compress_mode in ['encoding', 'decoding']:
            t, h, w = self._get_ctx_i_loc(k, i)
            mask = tensor_mask[:, :, t, h, w, :]
            return mask * means[:, :, -1, H // 2, W // 2, :], \
                   mask * scales[:, :, -1, H // 2, W // 2, :]
        else:
            raise ValueError(f'Unknown compress mode {self.compress_mode}')

    def _get_log_step(self, k, i, params):
        """（量化步长：训练返回整块；编解码时只取当前扫描位置 i 的步长切片）"""
        if self.learned_quant:
            log_step = super()._get_log_step(k, i, params)
            if self.compress_mode in ['none']:
                return log_step
            elif self.compress_mode in ['encoding', 'decoding']:
                if i is None:
                    return log_step
                else:
                    return log_step[:, :, i, :]
            else:
                raise ValueError(f'Unknown compress mode {self.compress_mode}')
        else:
            return -4.