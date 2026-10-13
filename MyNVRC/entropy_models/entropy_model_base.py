"""
Basic entropy model classes（熵模型基类：统一“量化 + 熵参数预测 + 码率估计 + 算术编解码”的模板流程）

核心设计（模板方法模式）：
    _forward_rate_tensor()  训练时的率估计（一次并行完成所有上下文步）；
    _compress_tensor()      实际算术编码（按 _num_iters() 步串行扫描，逐步填充上下文）；
    _decompress_tensor()    实际算术解码（同上，但符号来自解码器）。
    三者共享一组可重写的钩子：_scale/_unscale（量化步长缩放）、_init_ctx/_extract_ctx/_fill_ctx
    （上下文管理）、_init_meta/_extract_meta（边信息）、_entropy_param（熵参数预测）等，
    子类（网格/权重/自回归模型）只需实现这些钩子即可复用整套流程。
"""
from .utils import *
from .distributions import *
from .quantizer import *


class FullPrecisionModel(FuncModule):
    """全精度模型：不量化不熵编码，直接按原始字节存储（率 = 字节数 * 8），用于 psi 参数与其他 1D 参数"""
    def __init__(self, logger, name, inputs):
        super().__init__()
        self.name = name

    def forward(self, x, params={}, forward_rates=True, forward_params=True):
        rates, x = forward_meta(x)
        return (sum(rates.values()) if forward_rates else None), (x if forward_params else None)

    def compress(self, bs, x, params={}):
        return compress_meta(bs, x)

    def decompress(self, bs, x, params={}):
        return decompress_meta(bs, x)


class HalfPrecisionModel(FullPrecisionModel):
    """半精度模型：先把浮点参数转 fp16 再按字节存储（率减半，量化误差 2^-11）"""
    def _convert_half(self, x):
        """（浮点张量转 fp16）"""
        x_half = {}
        for k, v in x.items():
            x_half[k] = v.half() if v is not None and isinstance(v, torch.Tensor) and v.is_floating_point() else v
        return x_half

    def _restore_half(self, x_half, x_ref):
        """（恢复到参考张量的 dtype/设备）"""
        x = {}
        for k, v in x_half.items():
            x[k] = v.to(x_ref[k].dtype) if v is not None and isinstance(v, torch.Tensor) else v
        return x

    def forward(self, x, params={}, forward_rates=True, forward_params=True):
        x_half = self._convert_half(x)
        return super().forward(x_half, params, forward_rates, forward_params)

    def compress(self, bs, x, params={}):
        x_half = self._convert_half(x)
        bs, rate, x_half = super().compress(bs, x_half)
        return bs, rate, self._restore_half(x_half, x)

    def decompress(self, bs, x, params={}):
        x_half = self._convert_half(x)
        bs, rate, x_half = super().decompress(bs, x_half)
        return bs, rate, self._restore_half(x_half, x)


class _EntropyModelBase(FuncModule):
    """
    The basic entropy model class.
    （可学习熵模型基类：实现“缩放 -> 量化 -> 熵参数 -> 似然/算术编码”的模板流程）

    两条量化通路（论文中 R/D 分离的关键）：
        rate_quantizer  用于率路（估计码率时的软量化，如 soft-round-noise / quant-noise）；
        dist_quantizer  用于失真路（送入网络的量化，训练时同样加噪/软舍入，评估时取 round）。
    """
    def __init__(self, logger, name, inputs, config):
        super().__init__()
        self.name = name
        if config.em_dist == 'gaussian':
            # 条件高斯分布模型（内含 scale_table 量化高斯尺度，便于算术编码建 CDF 表）
            self.dist_model = GaussianConditional(
                scale_table=np.exp(np.linspace(np.log(0.11), np.log(256), 64)).tolist(), scale_bound=0.1
            )
        else:
            raise ValueError

        # Quantisation settings（率路/失真路量化器）
        self.rate_quantizer = get_quantizer(config.rate_quant)
        self.dist_quantizer = get_quantizer(config.dist_quant)

        # Flag to indicate whether the model is in compression mode.
        # compress_mode: 'none'（训练前向）/ 'encoding' / 'decoding'，
        # 自回归模型据此决定是并行一次性处理还是逐步串行扫描。
        self.compress_mode = 'none'

        # Flag to indicate whether the means are quantised. 
        # This is used to ensure the consistency between the estimation and the actual coding.
        # use_quant_mean：对高斯均值也做量化，保证训练时的率估计与实际编码码率一致
        self.use_quant_mean = config.quant_mean

    def extra_repr(self):
        s = 'use_quant_mean={use_quant_mean}'
        return s.format(**self.__dict__)

    def _check_implementation(self, inputs):
        # Test pre- and post-processing（自检：preprocess/postprocess 必须互为逆变换）
        inputs_restored = self._postprocess(self._preprocess(inputs))
        for k in inputs.keys():
            assert (inputs[k] is None and k not in inputs_restored) or torch.allclose(inputs[k], inputs_restored[k]), \
                f'Input {k} is not correctly before/after pre- and post-processing.'

    def _set_params(self, inputs, init_bit_width, params):
        pass

    def set_params(self, inputs, init_bit_width, soft_round_temp, kum_noise, quant_noise, params):
        """每个 epoch 更新两个量化器的退火超参，并可选用 _set_params 初始化量化位宽"""
        with torch.no_grad():
            # Update quantisers
            update_quantizer(self.rate_quantizer,
                             soft_round_temp=soft_round_temp, kum_noise=kum_noise,
                             quant_noise=quant_noise)
            update_quantizer(self.dist_quantizer, 
                             soft_round_temp=soft_round_temp, kum_noise=kum_noise,
                             quant_noise=quant_noise)
            self._set_params(inputs, init_bit_width, params)

    def _preprocess(self, inputs):
        """（输入重排：把原始张量整理成熵模型处理的分块/拼接形式；子类实现）"""
        raise NotImplementedError

    def _postprocess(self, inputs):
        """（_preprocess 的逆变换：把处理结果还原回原始形状）"""
        raise NotImplementedError

    def _num_iters(self, k):
        """（编码扫描步数：普通模型 1 步并行；自回归模型在编解码时为块内元素数）"""
        raise NotImplementedError

    def _init_ctx(self, k, x, params):
        """（初始化上下文张量 ctx 与有效位掩码 ctx_mask）"""
        raise NotImplementedError

    def _extract_ctx(self, k, i, ctx, ctx_mask, params):
        """（取出第 i 步编码所需的上下文片段）"""
        raise NotImplementedError

    def _fill_ctx(self, k, i, ctx, ctx_mask, x_i, params):
        """（把第 i 步已解码的值 x_i 写回上下文，供后续步使用）"""
        raise NotImplementedError

    def _init_meta(self, k, x, params):
        """（初始化边信息 meta：如未学习熵参数时存储的均值/标准差）"""
        raise NotImplementedError

    def _extract_meta(self, k, i, meta):
        """（取出第 i 步的边信息）"""
        raise NotImplementedError

    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        """（取出第 i 步待编码的符号及其有效位掩码）"""
        raise NotImplementedError

    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        """（由上下文预测高斯熵参数 means/scales——上下文模型的核心）"""
        raise NotImplementedError

    def _get_log_step(self, k, i, params):
        """（返回量化步长的对数 log_step；学习式量化步长的核心钩子）"""
        raise NotImplementedError

    def _scale(self, k, i, x_i, params):
        """（除以量化步长 delta = exp(clip(log_step))：z = x / delta）"""
        return scale(x_i, self._get_log_step(k, i, params))

    def _unscale(self, k, i, x_i, params):
        """（乘回量化步长：x_hat = z_hat * delta）"""
        return unscale(x_i, self._get_log_step(k, i, params))

    def _compute_rate(self, k, i, p_i, p_i_mask):
        """（码率 = -log2(likelihood)，仅在有效位上累计）"""
        return -torch.log2(p_i) * p_i_mask if p_i_mask is not None else -torch.log2(p_i)

    def _forward_rate_tensor(self, k, x, params, forward_rate=True, forward_tensor=True, reduce_rates=True):
        """
        训练时的率估计前向（单个张量 k）：
            1. _scale() 用学习到的步长缩放；
            2. rate_quantizer/dist_quantizer 做可微量化（软舍入+噪声 或 QuantNoise）；
            3. _unscale() 得到送入网络的量化张量 x_dq；
            4. 若需率：初始化上下文/边信息 -> 循环 _entropy_param() 预测 (means, scales)
               -> dist_model() 计算似然 -> _compute_rate() 累计码率 -> _fill_ctx() 更新上下文。
        训练时（compress_mode='none'）上下文一次性并行计算，无需串行扫描。
        """
        assert forward_rate or forward_tensor, 'At least one of forward_rates and forward_tensor should be True'
        self.compress_mode = 'none'

        # Init rate
        if reduce_rates:
            rate = 0.
        else:
            rate = {'meta': [], 'data': []}

        # Scale
        # TODO: Extract the log step & prevent re-computation for scale/unscale
        x_s = self._scale(k, None, x, params)

        if forward_rate:
            # Quantisation
            if type(self.rate_quantizer) == type(self.dist_quantizer):
                x_rsq = x_dsq = self.rate_quantizer(x_s, params)
            else:
                x_rsq = self.rate_quantizer(x_s, params)
                x_dsq = self.dist_quantizer(x_s, params)

            # Unscale
            x_dq = self._unscale(k, None, x_dsq, params)
        else:
            # Quantisation
            x_dsq = self.dist_quantizer(x_s, params)

            # Unscale
            x_dq = self._unscale(k, None, x_dsq, params)

            return None, x_dq

        # Init contexts
        ctx, ctx_mask = self._init_ctx(k, x_dq, params)
        assert ctx is None or is_shape_the_same(ctx, x), \
            'ctx should have the same shape as x.'
        assert ctx_mask is None or is_shape_compatible(ctx_mask, ctx), \
            'ctx_mask should have a compatible shape with ctx.'

        # Init meta data
        meta = self._init_meta(k, x_dq, params)
        meta_rates, meta = forward_meta(meta)
        if reduce_rates:
            rate += sum(meta_rates.values())
        else:
            rate['meta'].append(sum(meta_rates.values()))

        for i in range(self._num_iters(k)):
            # Extract context
            ctx_i, ctx_mask_i = self._extract_ctx(k, i, ctx, ctx_mask, params)

            # Extract meta data
            meta_i = self._extract_meta(k, i, meta)

            # Extract codes to be encode
            x_rsq_i, x_i_mask = self._extract_code(k, i, ctx, ctx_mask, x_rsq, params)
            x_dsq_i, _ = self._extract_code(k, i, ctx, ctx_mask, x_dsq, params)
            assert x_i_mask is None or is_shape_compatible(x_i_mask, x_rsq_i), \
                'x_i_mask should have a compatible shape with x_rsq_i.'
            assert is_shape_the_same(x_rsq_i, x_dsq_i), \
                'x_sqn_i should have the same shape as x_sq_i.'

            # Encoding step
            if type(self.dist_model) in [GaussianConditional]:
                # Entropy parameters
                means, scales = self._entropy_param(k, i, ctx_i, ctx_mask_i, meta_i, params)
                assert is_shape_compatible(means, x_rsq_i), \
                    'means should have a compatible shape with x_sq_i.'
                assert is_shape_compatible(scales, x_rsq_i), \
                    'scales should have a compatible shape with x_sq_i.'

                # Scaling means and scales
                means_s = self._scale(k, i, means, params)
                scales_s = self._scale(k, i, scales, params)

                if self.use_quant_mean:
                    means_s = self.dist_quantizer(means_s, params)

                # Likelihood
                p_i = self.dist_model(x_rsq_i, means=means_s, scales=scales_s)
            else:
                raise NotImplementedError

            # Unscaling
            x_qn_i = self._unscale(k, i, x_dsq_i, params)

            # Accumulate rate
            rate_i = self._compute_rate(k, i, p_i, x_i_mask)
            if reduce_rates:
                rate += rate_i.sum()
            else:
                rate['data'].append(rate_i)

            # Update context
            ctx, ctx_mask = self._fill_ctx(k, i, ctx, ctx_mask, x_qn_i, params)
            assert ctx is None or is_shape_the_same(ctx, x), \
                'ctx should have the same shape as x.'
            assert ctx_mask is None or is_shape_compatible(ctx_mask, ctx), \
                'ctx_mask should have a compatible shape with ctx.'

        return rate, (x_dq if forward_tensor else None)

    def _sort_forward(self, inputs):
        """（决定多个输入张量的处理顺序；子类实现）"""
        raise NotImplementedError

    def forward(self, inputs, params, forward_rates=True, forward_params=True):
        """训练前向：对每个（重排后的）张量调用 _forward_rate_tensor，汇总码率并还原形状"""
        assert forward_rates or forward_params, 'At least one of forward_rates and forward_params should be True'
        rates = {}
        outputs = {}
        inputs = self._preprocess(inputs)

        for k, x in self._sort_forward(inputs):
            x_rate, x_q = self._forward_rate_tensor(
                k, x, params, forward_rate=forward_rates, forward_tensor=forward_params
            )
            rates[k] = x_rate
            outputs[k] = x_q

        return (sum(rates.values()) if forward_rates else None), \
               (self._postprocess(outputs) if forward_params else None) 

    def _compress_tensor(self, bs, k, x, params):
        """
        实际算术编码（单个张量）：
            真实量化 x_sq = round(x / delta)；初始化上下文（全零）后按 _num_iters() 步串行扫描，
            每步用已解码邻居预测 (means, scales) 并缓冲符号，最后统一 rANS 编码。
            码流 = 头部（meta/data 长度）+ meta 字节 + 数据字节。
        """
        assert not self.training, 'compression should be done in eval mode.'
        self.compress_mode = 'encoding'
        self.dist_model.update()

        # Init bitstreams
        meta_bs = b''
        data_bs = b''

        # Scale
        x_s = self._scale(k, None, x, params)

        # Quantisation
        x_sq = x_s.round()

        # Unscale
        x_q = self._unscale(k, None, x_sq, params)

        # Init contexts
        ctx, ctx_mask = self._init_ctx(k, x_q, params)
        assert ctx is None or ctx.shape == x.shape, \
            'ctx should have the same shape as x.'
        assert ctx_mask is None or ctx_mask.shape == x.shape, \
            'ctx_mask should have the same shape as x.'

        # Init meta data
        meta = self._init_meta(k, x_q, params)
        meta_bs, _, meta = compress_meta(meta_bs, meta)

        # Init encoder buffer
        self.dist_model.encode_buffer_init()

        for i in range(self._num_iters(k)):
            # Extract condition and context
            ctx_i, ctx_mask_i = self._extract_ctx(k, i, ctx, ctx_mask, params)

            # Extract meta data
            meta_i = self._extract_meta(k, i, meta)

            # Extract codes to be encode
            x_sq_i, x_i_mask = self._extract_code(k, i, ctx, ctx_mask, x_sq, params)
            assert x_i_mask is None or x_sq_i.shape == x_i_mask.shape, \
                'x_i_mask should have the same shape as x_sq_i.'

            # Encoding step
            if type(self.dist_model) in [GaussianConditional]:
                # Entropy parameters
                means, scales = self._entropy_param(k, i, ctx_i, ctx_mask_i, meta_i, params)
                assert means.shape == x_sq_i.shape, \
                    'means should have the same shape as x_sq_i.'
                assert scales.shape == x_sq_i.shape, \
                    'scales should have the same shape as x_sq_i.'

                # Scaling means and scales
                means_s = self._scale(k, i, means, params).round()
                scales_s = self._scale(k, i, scales, params)
                if x_i_mask is not None:
                    means_s = means_s * x_i_mask
                    scales_s = scales_s * x_i_mask

                # Accumulate outputs to the buffer
                self.dist_model.encode_buffer_append(x_sq_i, means_s, scales_s)
            else:
                raise NotImplementedError

            # Unscaling
            x_q_i = self._unscale(k, i, x_sq_i, params)

            # Refill context
            ctx, ctx_mask = self._fill_ctx(k, i, ctx, ctx_mask, x_q_i, params)
            assert ctx is None or x.shape == ctx.shape, \
                'ctx should have the same shape as x.'
            assert ctx_mask is None or x.shape == ctx_mask.shape, \
                'ctx_mask should have the same shape as x.'

        # Encoding
        data_bs += self.dist_model.encode_with_buffer()

        # Combine all bitstreams
        header = {'meta_length': torch.tensor(len(meta_bs), dtype=torch.int64), 
                  'data_length': torch.tensor(len(data_bs), dtype=torch.int64)}
        bs, header_rates, header = compress_meta(bs, header)
        rate = sum(header_rates.values())

        bs += meta_bs
        rate += len(meta_bs) * 8

        bs += data_bs
        rate += len(data_bs) * 8

        return bs, rate, ctx

    def _decompress_tensor(self, bs, k, x, params):
        """
        实际算术解码（单个张量）：与 _compress_tensor 完全对称——
        先读头部与 meta，再逐步用上下文预测 (means, scales) 并从 rANS 流中解出符号。
        注意 x 只提供形状/设备模板，数值本身被解码结果覆盖。
        """
        assert not self.training, 'compression should be done in eval mode.'
        self.compress_mode = 'decoding'
        self.dist_model.update()

        # Init bitstreams
        meta_bs = b''
        data_bs = b''

        # Split bitstream
        header = {'meta_length': torch.tensor(0, dtype=torch.int64), 
                  'data_length': torch.tensor(0, dtype=torch.int64)}
        bs, header_rates, header = decompress_meta(bs, header)
        rate = sum(header_rates.values())

        meta_bs, bs = bs[:header['meta_length']], bs[header['meta_length']:] 
        rate += len(meta_bs) * 8

        data_bs, bs = bs[:header['data_length']], bs[header['data_length']:] 
        rate += len(data_bs) * 8

        # Scale
        x_s = self._scale(k, None, x, params)

        # Quantisation
        x_sq = x_s.round()

        # Unscale
        x_q = self._unscale(k, None, x_sq, params)

        # Init contexts
        ctx, ctx_mask = self._init_ctx(k, x_q, params)
        assert ctx is None or x.shape == ctx.shape, \
            'ctx should have the same shape as x.'
        assert ctx_mask is None or x.shape == ctx_mask.shape, \
            'ctx_mask should have the same shape as x.'

        # Init meta data
        meta = self._init_meta(k, x_q, params)
        meta_bs, _, meta = decompress_meta(meta_bs, meta)

        # Init decoder buffer
        self.dist_model.decode_buffer_init(data_bs)

        for i in range(self._num_iters(k)):
            # Extract condition and context
            ctx_i, ctx_mask_i = self._extract_ctx(k, i, ctx, ctx_mask, params)

            # Extract meta data
            meta_i = self._extract_meta(k, i, meta)

            # Extract template codes
            x_sq_i, x_i_mask = self._extract_code(k, i, ctx, ctx_mask, x_sq, params)
            assert x_i_mask is None or x_sq_i.shape == x_i_mask.shape, \
                'x_i_mask should have the same shape as x_sq_i.'
 
            # Decoding step
            if type(self.dist_model) in [GaussianConditional]:
                # Entropy parameters
                means, scales = self._entropy_param(k, i, ctx_i, ctx_mask_i, meta_i, params)
                assert means.shape == x_sq_i.shape, \
                    'means should have the same shape as x_sq_i.'
                assert scales.shape == x_sq_i.shape, \
                    'scales should have the same shape as x_sq_i.'

                # Scaling means and scales
                means_s = self._scale(k, i, means, params).round()
                scales_s = self._scale(k, i, scales, params)
                if x_i_mask is not None:
                    means_s = means_s * x_i_mask
                    scales_s = scales_s * x_i_mask

                # Decoding
                x_sq_i = self.dist_model.decode_with_buffer(x_sq_i, means_s, scales_s)
            else:
                raise NotImplementedError

            # Unscaling
            x_q_i = self._unscale(k, i, x_sq_i, params)

            # Refill context
            ctx, ctx_mask = self._fill_ctx(k, i, ctx, ctx_mask, x_q_i, params)
            assert ctx is None or x.shape == ctx.shape, \
                'ctx should have the same shape as x.'
            assert ctx_mask is None or x.shape == ctx_mask.shape, \
                'ctx_mask should have the same shape as x.'

        return bs, rate, ctx

    def compress(self, bs, inputs, params):
        """对一组输入张量逐个执行 _compress_tensor 并拼接码流"""
        with torch.no_grad():
            rate = 0
            outputs = {}
            inputs = self._preprocess(inputs)

            for k, x in self._sort_forward(inputs):
                bs, x_rate, x = self._compress_tensor(bs, k, x, params)
                rate += x_rate
                outputs[k] = x

            outputs = self._postprocess(outputs)

            return bs, rate, outputs

    def decompress(self, bs, inputs, params):
        """对一组输入张量逐个执行 _decompress_tensor（inputs 仅作形状模板）"""
        with torch.no_grad():
            rate = 0
            outputs = {}
            inputs = self._preprocess(inputs)

            for k, x in self._sort_forward(inputs):
                bs, x_rate, x = self._decompress_tensor(bs, k, x, params)
                rate += x_rate
                outputs[k] = x

            outputs = self._postprocess(outputs)

            return bs, rate, outputs


class EntropyModel(_EntropyModelBase):
    """
    通用逐张量条件高斯熵模型（论文中 Level-2 的 phi 压缩模型）：
        - 把所有输入张量展平拼接成一行向量 'all'；
        - 每个张量学习一个标量 log_step / mean / log_scale（或用边信息统计量代替）；
        - 无上下文（means/scales 为常数），一次并行完成编码。
    """
    def __init__(self, logger, name, inputs, config):
        super().__init__(logger, name, inputs, config)
        self.learned_quant = config.learned_quant
        self.learned_em = config.learned_em
        self.tensor_keys = self._build_keys(inputs)
        self.tensor_cfgs = self._build_cfgs(inputs)
        self.forward_order = self._build_forward_order(inputs)
        self.length = sum([tensor_cfg['length'] for tensor_cfg in self.tensor_cfgs.values()])
        self._check_implementation(inputs)
        self._build_masks()

    def extra_repr(self):
        s = super().extra_repr() + ', learned_quant={learned_quant}, learned_em={learned_em}'
        return s.format(**self.__dict__)

    def _sort_tensors(self, inputs):
        """（按注册顺序列出非空张量）"""
        return [(tensor_k, inputs[tensor_k]) for tensor_k in self.tensor_keys]

    def _build_keys(self, inputs):
        """（收集非空输入张量的键并排序）"""
        return sorted(tensor_k for tensor_k in inputs.keys() if inputs[tensor_k] is not None)

    def _build_cfgs(self, inputs):
        """（为每个张量记录形状/展平方式/拼接偏移等配置）"""
        tensor_cfgs = {}
        offset = 0

        with torch.no_grad():
            for tensor_k, tensor in self._sort_tensors(inputs): 
                tensor = tensor.unsqueeze(0)
                length = math.prod(tensor.shape[1:])
                tensor_cfgs[tensor_k] = {}
                tensor_cfgs[tensor_k]['shape'] = (-1, *tensor.shape[1:])
                tensor_cfgs[tensor_k]['reshape'] = (-1, 1, math.prod(tensor.shape[1:]))
                tensor_cfgs[tensor_k]['offset'] = offset
                tensor_cfgs[tensor_k]['length'] = length
                tensor_cfgs[tensor_k]['meta_shape'] = (1, 1)
                offset += length

        return tensor_cfgs

    def _build_forward_order(self, inputs):
        """（处理顺序：所有张量拼接为 'all' 一次处理）"""
        return ['all']

    def _build_masks(self):
        """（构建每个张量的有效位掩码，padding 产生的无效元素不计码率）"""
        masks = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            if tensor_k in masks and masks[tensor_k] is not None:
                masks[tensor_k] = masks[tensor_k].bool()
                continue
            masks[tensor_k] = torch.ones(tensor_cfg['shape'][1:], dtype=torch.bool)

        for k, m in self._preprocess(masks).items():
            self.register_buffer(f'{k}_mask', m, persistent=False)

    def _get_params(self):
        """（熵模型自身的可学习参数：每个张量一个标量 log_step / means / log_scales）"""
        params = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            meta_shape = tensor_cfg['meta_shape']
            # Log step
            if self.learned_quant:
                params[f'{tensor_k}_log_step'] = init_log_step(meta_shape)
            # EM parameters
            if self.learned_em:
                params[f'{tensor_k}_means'] = init_means(meta_shape, means=0.)
                params[f'{tensor_k}_log_scales'] = init_log_scales(meta_shape, scales=1.0)
        return params

    def _set_params(self, inputs, init_bit_width, params):
        """（可选：按目标位宽初始化量化步长——用张量动态范围除以量化级数）"""
        with torch.no_grad():
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                # Log step
                if self.learned_quant and init_bit_width is not None:
                    # Get log steps
                    log_step = params[f'{tensor_k}_log_step']
                    log_step = log_step.unsqueeze(0)

                    # Set width
                    set_width(self._reshape_tensor(tensor_k, inputs[tensor_k]), log_step, init_bit_width)

    def _reshape_tensor(self, tensor_k, tensor):
        """（把张量重排为统一的一行向量）"""
        tensor = tensor.unsqueeze(0)
        tensor = tensor.view(self.tensor_cfgs[tensor_k]['reshape'])
        return tensor

    def _restore_tensor(self, tensor_k, tensor):
        """（_reshape_tensor 的逆变换，还原原始形状）"""
        tensor = tensor.view(self.tensor_cfgs[tensor_k]['shape'])
        tensor = tensor.squeeze(0)
        return tensor.contiguous()

    def _preprocess(self, inputs):
        """（把所有张量展平并沿最后一维拼接成单个张量 'all'）"""
        outputs = []
        for tensor_k, tensor in self._sort_tensors(inputs):
            outputs.append(self._reshape_tensor(tensor_k, tensor))
        return {'all': torch.cat(outputs, dim=2)}

    def _postprocess(self, inputs):
        """（按 offset 把 'all' 切回各张量并还原形状）"""
        outputs = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            outputs[tensor_k] = self._restore_tensor(
                tensor_k, inputs['all'][:, :, tensor_cfg['offset']:tensor_cfg['offset'] + tensor_cfg['length']]
            )
        return outputs

    def _num_iters(self, k):
        """（一次并行处理，无需串行扫描）"""
        return 1

    def _init_ctx(self, k, x, params):
        """（无上下文模型）"""
        return None, None

    def _extract_ctx(self, k, i, ctx, ctx_mask, params):
        """（无上下文模型）"""
        return None, None

    def _fill_ctx(self, k, i, ctx, ctx_mask, x_i, params):
        """（无上下文模型，直接返回当前值）"""
        return x_i, ctx_mask

    def _init_meta(self, k, x, params):
        """（边信息：学习式熵模型无边信息；否则存储张量的 fp16 均值/标准差作为码流一部分）"""
        if self.learned_em:
            meta = {}
        else:
            meta = {
                'means': x.mean(dim=2, keepdims=True).half(),
                'scales': x.std(dim=2, keepdims=True).half() if x.shape[2] > 1 else torch.zeros_like(x).half()
            }
        return meta

    def _extract_meta(self, k, i, meta):
        """（返回整份边信息）"""
        return meta

    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        """（待编码符号即整个张量，附有效位掩码）"""
        return x, getattr(self, f'{k}_mask')

    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        """（熵参数：每张量常数 mean/scale 广播到所有元素；或使用边信息统计量）"""
        if self.learned_em:
            means = []
            scales = []

            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_means = params[f'{tensor_k}_means']
                tensor_scales = params[f'{tensor_k}_log_scales'].exp()
                tensor_means = tensor_means.unsqueeze(0)
                tensor_scales = tensor_scales.unsqueeze(0)
                means.append(tensor_means.expand(tensor_cfg['reshape']))
                scales.append(tensor_scales.expand(tensor_cfg['reshape']))

            means = torch.cat(means, dim=2)
            scales = torch.cat(scales, dim=2)
        else:
            means = meta_i['means'].float().expand(-1, -1, self.length)
            scales = meta_i['scales'].float().expand(-1, -1, self.length)
        return means, scales

    def _sort_forward(self, inputs):
        """（按 forward_order 迭代）"""
        return [(k, inputs[k]) for k in self.forward_order]

    def _get_log_step(self, k, i, params):
        """（量化步长：每张量一个学习到的标量，广播到所有元素；未学习时固定 delta=2^-4）"""
        if self.learned_quant:
            log_steps = []
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_log_step = params[f'{tensor_k}_log_step']
                tensor_log_step = tensor_log_step.unsqueeze(0)
                log_steps.append(tensor_log_step.expand(tensor_cfg['reshape']))
            return torch.concat(log_steps, dim=2)
        else:
            return -4.