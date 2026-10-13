"""
NVRC codec（NVRC 编解码器：论文的核心贡献——对 INR 模型参数本身进行分层压缩）

三层分层压缩结构（对应论文 Sec.3.3 的 bitstream hierarchy）：
    Level 1  模型参数 theta（特征网格 + 层权重 + 其他）：
             - 特征网格 -> grid_em（ARGridEntropyModel，块内自回归上下文模型）；
             - 层权重   -> weight_em（DoubleAxisEntropyModel，双轴分解的高斯模型）；
             - 其他 1D 参数 -> meta_em（FullPrecisionModel，直接存原精度）。
    Level 2  上述两个熵模型自身的参数 phi（量化步长 + 熵参数）：
             - grid_em 的参数 -> grid_meta_em（逐张量条件高斯 EntropyModel）；
             - weight_em 的参数 -> weight_meta_em（逐张量条件高斯 EntropyModel）。
    Level 3  Level-2 熵模型的参数 psi -> meta_em 直接以全精度存储（不再熵编码）。

所有层的比特都计入总码率 R = R_theta + R_phi + R_psi，实现真正的端到端率失真优化。
"""
from .utils import *
from .codec_base import *
from entropy_models.entropy_model_base import (
    FullPrecisionModel, HalfPrecisionModel, _EntropyModelBase, EntropyModel
)
from entropy_models.weight_entropy_model import (
    WeightEntropyModel, SingleAxisEntropyModel, DoubleAxisEntropyModel,
)
from entropy_models.grid_entropy_model import (
    GridEntropyModel, ARGridEntropyModel
)
from entropy_models.qec_entropy_model import QECEntropyModel  # MyHiNeRV 迁移：UAR-NVC 式最简 QEC


@dataclass
class EMConfigBase:
    """熵模型基础配置：分布类型（高斯）、率/失真量化器类型、是否量化均值、是否学习量化步长与熵参数"""
    em: str = 'em'
    em_dist: str = 'gaussian'
    rate_quant: str = 'soft-round-noise'   # 率路量化：软舍入 + 噪声（阶段 1）/ QuantNoise（阶段 2）
    dist_quant: str = 'soft-round-noise'   # 失真路量化：同上
    quant_mean: bool = False               # 是否对高斯均值也做量化（保证估计与实际编码一致）
    learned_quant: bool = True             # 是否学习量化步长（对数域）
    learned_em: bool = True                # 是否学习熵参数（均值/尺度）


@dataclass
class GridEMConfig(EMConfigBase):
    """特征网格熵模型配置：3D 块大小、上下文网络通道数/深度/卷积核等"""
    blk_size: List[int] = field(default_factory=lambda: [20, 5, 5])  # 3D 块尺寸 (T, H, W)，论文用 [16, 8, 8]
    per_blk: bool = True                  # 逐块学习量化步长
    channels: int = 4                     # 上下文网络通道宽度
    depth: int = 3                        # 上下文网络深度
    kernel_size: int = 3                  # 掩码 3D 卷积核
    norm: str = 'layernorm'
    bias: bool = True
    act: str = 'gelu'
    depthwise: bool = False               # 是否深度可分离卷积


@dataclass
class WeightEMConfig(EMConfigBase):
    """层权重熵模型配置：2D 块大小（输出通道 x 输入通道）与是否逐轴分解"""
    blk_size: List[int] = field(default_factory=lambda: [16, 32])
    per_blk: bool = True
    per_axis: bool = True                 # 量化步长按输出/输入两轴分解 delta[i,j] = delta_out[i] * delta_in[j]


@dataclass
class MetaEMConfig(EMConfigBase):
    """二级熵模型（压缩 phi 参数）配置"""
    pass


@dataclass
class OptimizeConfig:
    """优化配置：是否优化码率、量化超参的退火区间（从 start 到 end 线性变化）"""
    optimize_rate: bool = True
    init_bit_width: int = -1              # 初始量化位宽（-1 表示不初始化）
    soft_round_temp: List[float] = field(default_factory=lambda: [0.3, 0.1])  # 软舍入温度退火区间
    kum_noise: List[float] = field(default_factory=lambda: [2.0, 1.0])        # Kumaraswamy 噪声参数退火区间
    quant_noise: List[float] = field(default_factory=lambda: [0.0, 0.9])      # QuantNoise 比例退火区间


@dataclass
class NVRCConfig:
    """NVRC 总配置：grid_em / weight_em / 两个 meta 熵模型 + 优化设置"""
    grid_em: GridEMConfig = field(default_factory=GridEMConfig)
    weight_em: WeightEMConfig = field(default_factory=WeightEMConfig)
    grid_meta_em: MetaEMConfig = field(default_factory=MetaEMConfig)
    weight_meta_em: MetaEMConfig = field(default_factory=MetaEMConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    def __post_init__(self):
        self.grid_em = GridEMConfig(**self.grid_em) \
                       if isinstance(self.grid_em, dict) else self.grid_em
        self.weight_em = WeightEMConfig(**self.weight_em) \
                         if isinstance(self.weight_em, dict) else self.weight_em
        self.grid_meta_em = MetaEMConfig(**self.grid_meta_em) \
                            if isinstance(self.grid_meta_em, dict) else self.grid_meta_em
        self.weight_meta_em = MetaEMConfig(**self.weight_meta_em) \
                              if isinstance(self.weight_meta_em, dict) else self.weight_meta_em
        self.optimize = OptimizeConfig(**self.optimize) \
                        if isinstance(self.optimize, dict) else self.optimize


class NVRC(CodecBase):
    """
    NVRC codec（NVRC 编解码器主类）

    组织方式：把所有待压缩张量都包装成 ParamModel（nn.ParameterDict），
    训练时通过 _forward_rates_params() 前向得到 (各部分码率, 量化后参数)，
    编码/解码时按“由上至下”的顺序依次压缩/恢复：psi -> phi -> theta。
    """
    def __init__(self, logger, model, config):
        super().__init__()
        self.logger = logger
        self.config = copy.deepcopy(config)

        logger.info(f'NVRC:')
        logger.info(f'     Optimize rate - {self.config.optimize.optimize_rate}  Init bit width - {self.config.optimize.init_bit_width}')
        logger.info(f'     Soft-rounding temp - {self.config.optimize.soft_round_temp}  Kumaraswamy - {self.config.optimize.kum_noise}')
        logger.info(f'     QuantNoise - {self.config.optimize.quant_noise}')
        logger.info(f'     Cache enabled - {self.cache_enabled}')

        # INR（被压缩的 INR 主干模型，如 HiNeRV）
        self.model = model

        # INR param model（提取主干模型的全部参数，包装成 ParameterDict 便于整体量化/编解码）
        init_params = self.model.get_params_recursive()
        self.param_model = ParamModel(logger, init_params)

        # MyHiNeRV 迁移（模块 C）：残差编码参考 θ'。非 None 时，熵模型估计/编解码的
        # 对象是 θ − θ'（训练时率估计一致），解码端加回。θ' 由 base+前一 GoP 插值
        # 确定性重建（见 util/derive_init.py），不进码流。
        self._residual_ref = None

        # C5 渐进剪枝器（progressive_prune.py）：非 None 时前向先掩码再进熵模型
        self.pruner = None

        # 按参数类型划分：特征网格 grid / 层权重 weight / 其他（1D 参数等）
        params = self.param_model()
        self._set_split_params(init_params)
        grids, weights, _ = self._split_params(params)

        # Grid em（构建特征网格的上下文熵模型，并同样用 ParamModel 包装其参数）
        self._build_grid_em(logger, grids=grids, config=self.config.grid_em)
        self.grid_em.init_func_modules()
        self.grid_em_param_model = ParamModel(
            logger, self.grid_em.get_params_recursive()
        )

        # Weight em（构建层权重的双轴熵模型）
        self._build_weight_em(logger, weights=weights, config=self.config.weight_em)
        self.weight_em.init_func_modules()
        self.weight_em_param_model = ParamModel(
            logger, self.weight_em.get_params_recursive()
        )

        # Grid meta（构建压缩 grid_em 参数的二级熵模型）
        self._build_grid_meta_em(logger, grid_em_weights=self.grid_em_param_model(),
                                 config=self.config.grid_meta_em)
        self.grid_meta_em.init_func_modules()
        self.grid_meta_em_param_model = ParamModel(
            logger, self.grid_meta_em.get_params_recursive()
        )

        # Weight meta（构建压缩 weight_em 参数的二级熵模型）
        self._build_weight_meta_em(logger, weight_em_weights=self.weight_em_param_model(),
                                   config=self.config.weight_meta_em)
        self.weight_meta_em.init_func_modules()
        self.weight_meta_em_param_model = ParamModel(
            logger, self.weight_meta_em.get_params_recursive()
        )

        # Meta（第三层：直接全精度存储二级熵模型的参数 psi）
        self._build_meta_em(logger, inputs={})
        self.meta_em.init_func_modules()

    def _build_grid_em(self, logger, grids, config: GridEMConfig):
        """根据配置选择特征网格熵模型类型（ar=自回归上下文模型，blk=分块独立模型，fp32/fp16=直接存储）"""
        logger.info(f'     Grid EM:')
        logger.info(f'          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}')
        logger.info(f'          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}')
        logger.info(f'          Blk size - {config.blk_size}  Per blk - {config.per_blk}')
        logger.info(f'          Channels - {config.channels}  Depth - {config.depth}')
        logger.info(f'          Kernel size - {config.kernel_size}  Norm - {config.norm}  Bias - {config.bias}  Act - {config.act}')
        logger.info(f'          Depthwise - {config.depthwise}')

        try:
            em_cls = {
                'em': EntropyModel,
                'qec': QECEntropyModel,      # MyHiNeRV 迁移：逐张量 q_step+零均值高斯
                'blk': GridEntropyModel,
                'ar': ARGridEntropyModel,
                'fp32': FullPrecisionModel,
                'fp16': HalfPrecisionModel,
            }[config.em]
        except KeyError:
            raise ValueError(f'Unknown grid entropy model {config.em}')
        if issubclass(em_cls, _EntropyModelBase):
            self.grid_em = em_cls(logger, name='grid_em', inputs=grids, config=config)
        else:
            self.grid_em = em_cls(logger, name='grid_em', inputs=grids)

    def _build_weight_em(self, logger, weights, config: WeightEMConfig):
        """根据配置选择层权重熵模型类型（d-axis=双轴分解，s-axis=单轴，blk=分块，fp32/fp16=直接存储）"""
        logger.info(f'     Weight EM:')
        logger.info(f'          EM - {config.em}  Dist - {config.em_dist}  Rate quant - {config.rate_quant}')
        logger.info(f'          Blk size - {config.blk_size}  Per blk - {config.per_blk}  Per axis - {config.per_axis}')

        em_cls = {
            'em': EntropyModel,
            'qec': QECEntropyModel,          # MyHiNeRV 迁移：逐张量 q_step+零均值高斯
            'blk': WeightEntropyModel,
            's-axis': SingleAxisEntropyModel,
            'd-axis': DoubleAxisEntropyModel,
            'fp32': FullPrecisionModel,
            'fp16': HalfPrecisionModel,
        }[config.em]

        if issubclass(em_cls, _EntropyModelBase):
            self.weight_em = em_cls(logger, name='weight_em', inputs=weights, config=config)
        else:
            self.weight_em = em_cls(logger, name='weight_em', inputs=weights)

    def _build_grid_meta_em(self, logger, grid_em_weights, config: MetaEMConfig):
        """构建压缩 grid_em 自身参数的二级熵模型（默认逐张量条件高斯 EntropyModel）"""
        logger.info(f'     Grid meta - {config.em}')
        logger.info(f'          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}')
        logger.info(f'          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}')

        if grid_em_weights:
            em_cls = {
                'em': EntropyModel,
                'fp32': FullPrecisionModel,
                'fp16': HalfPrecisionModel,
            }[config.em]

            if issubclass(em_cls, _EntropyModelBase):
                self.grid_meta_em = em_cls(logger, name='grid_meta_em', inputs=grid_em_weights, config=config)
            else:
                self.grid_meta_em = em_cls(logger, name='grid_meta_em', inputs=grid_em_weights)
        else:
            self.grid_meta_em = FullPrecisionModel(logger, name='grid_meta_em', inputs={})

    def _build_weight_meta_em(self, logger, weight_em_weights, config: WeightEMConfig):
        """构建压缩 weight_em 自身参数的二级熵模型"""
        logger.info(f'     Weight meta - {config.em}')
        logger.info(f'          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}')
        logger.info(f'          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}')

        if weight_em_weights:
            em_cls = {
                'em': EntropyModel,
                'fp32': FullPrecisionModel,
                'fp16': HalfPrecisionModel,
            }[config.em]

            if issubclass(em_cls, _EntropyModelBase):
                self.weight_meta_em = em_cls(logger, name='weight_meta_em', inputs=weight_em_weights,
                                             config=config)
            else:
                self.weight_meta_em = em_cls(logger, name='weight_meta_em', inputs=weight_em_weights)
        else:
            self.weight_meta_em = FullPrecisionModel(logger, name='weight_meta_em', inputs={})

    def _build_meta_em(self, logger, inputs):
        """构建第三层熵模型（全精度直存，用于 psi 参数与其他 1D 参数）"""
        self.meta_em = FullPrecisionModel(logger, name='meta_em', inputs=inputs)

    def _set_split_params(self, params):
        """为每个参数标记类别：grid（特征网格）/ weight（>=2D 层权重）/ others（其余）"""
        self.param_split_types = {}

        for k, v in params.items():
            if v is not None:
                assert isinstance(v, nn.Parameter), 'Inputs should be nn.Parameter.'
                if is_grid_param(v):
                    self.param_split_types[k] = 'grid'
                elif v.ndim >= 2:
                    self.param_split_types[k] = 'weight'
                else:
                    self.param_split_types[k] = 'others'
            else:
                self.param_split_types[k] = 'others'
        
    def _split_params(self, params):
        # Split parameters into grids and weights
        grids = {}
        weights = {}
        others = {}

        for k, v in params.items():
            param_split_type = self.param_split_types[k]
            if param_split_type == 'grid':
                grids[k] = v
            elif param_split_type == 'weight':
                weights[k] = v
            else:
                others[k] = v

        return grids, weights, others

    def _forward_rates_params(self, x=None, forward_rates=True, forward_params=True):
        """
        前向计算（训练核心）：按“先高层后低层”的依赖顺序逐级量化并估计码率——
            meta_em(psi)            -> grid/weight meta em 的参数（全精度，率 = 存储比特）
            grid_meta_em(phi_g)     -> grid_em 参数的量化值与码率
            weight_meta_em(phi_w)   -> weight_em 参数的量化值与码率
            grid_em / weight_em     -> 模型参数 theta 的量化值与码率
        返回 (各部分码率字典 rates, 量化后模型参数 params_q)。
        MyHiNeRV 迁移（模块 C）：设置了 _residual_ref 时，θ 的熵估计在残差域
        （θ − θ'）进行，返回的 params_q 仍是实际权重域（θ' + quantized residual）。
        """
        # Grid meta params（第三层：grid_meta_em 的参数 psi_g，全精度直存）
        grid_meta_em_params = self.grid_meta_em_param_model()
        grid_meta_em_params_rate, grid_meta_em_params_q = self.meta_em(
            grid_meta_em_params, forward_rates=forward_rates, forward_params=True
        )

        # Grid em params
        grid_em_params = self.grid_em_param_model()
        grid_em_params_rate, grid_em_params_q = self.grid_meta_em(
            grid_em_params, grid_meta_em_params_q, forward_rates=forward_rates, forward_params=True
        )

        # Weight meta params
        weight_meta_em_params = self.weight_meta_em_param_model()
        weight_meta_em_params_rate, weight_meta_em_params_q = self.meta_em(
            weight_meta_em_params, forward_rates=forward_rates, forward_params=True
        )

        # Weight em params
        weight_em_params = self.weight_em_param_model()
        weight_em_params_rate, weight_em_params_q = self.weight_meta_em(
            weight_em_params, weight_meta_em_params_q, forward_rates=forward_rates, forward_params=True
        )

        # Model params
        params = self.param_model()
        # C5 渐进剪枝：掩码置零后再进熵模型（零权重码率趋零）
        if self.pruner is not None:
            params = self.pruner.apply(params)
        ref = self._residual_ref
        if ref is not None:
            # 模块 C：熵估计/量化在残差域进行（覆盖到的键），INR 前向仍用实际权重
            params_in = {k: (v - ref[k].to(device=v.device, dtype=v.dtype) if k in ref else v)
                         for k, v in params.items()}
        else:
            params_in = params
        grids, weights, others = self._split_params(params_in)
        grids_rate, grids_q = self.grid_em(
            grids, grid_em_params_q, forward_rates=forward_rates, forward_params=forward_params
        )
        weights_rate, weights_q = self.weight_em(
            weights, weight_em_params_q, forward_rates=forward_rates, forward_params=forward_params
        )
        others_rate, others_q = self.meta_em(
            others, forward_rates=forward_rates, forward_params=forward_params
        )

        # Total rate
        if forward_rates:
            rates = {
                k: torch.tensor(v, device=next(iter(params.values())).device) if not isinstance(v, torch.Tensor) else v
                for k, v in {
                    'grid': grids_rate, 'weight': weights_rate,
                    'grid_em_params': grid_em_params_rate, 'weight_em_params': weight_em_params_rate,
                    'grid_meta_em_params': grid_meta_em_params_rate,
                    'weight_meta_em_params': weight_meta_em_params_rate,
                    'others': others_rate
                }.items()
            }
        else:
            rates = None

        #  All params
        if forward_params:
            params_q = {**grids_q, **weights_q, **others_q} \
                       if self.config.optimize.optimize_rate else {**grids, **weights, **others}
            if ref is not None:
                # 模块 C：解码域加回参考 → 实际权重域
                params_q = {k: (q + ref[k].to(device=q.device, dtype=q.dtype) if k in ref else q)
                            for k, q in params_q.items()}
        else:
            params_q = None

        return rates, params_q

    def _get_rate(self, rates, reduce_rates):
        """（总码率：求和或保持字典形式）"""
        if reduce_rates:
            return sum(rates.values())
        else:
            return rates

    def _get_rate_loss(self, rates, reduce_rates):
        """（率损失：训练时优化用；把 7 部分码率全部相加，与实际编码码流一致）"""
        if reduce_rates:
            return rates['grid'] + rates['weight'] \
                   + rates['grid_em_params'] + rates['weight_em_params'] \
                   + rates['grid_meta_em_params'] + rates['weight_meta_em_params'] \
                   + rates['others']
        else:
            return {
                'grid': rates['grid'],
                'weight': rates['weight'],
                'grid_em_params': rates['grid_em_params'],
                'weight_em_params': rates['weight_em_params'],
                'grid_meta_em_params': rates['grid_meta_em_params'],
                'weight_meta_em_params': rates['weight_meta_em_params'],
                'others': rates['others']
            }

    def get_num_params(self):
        """统计 7 类参数各自的数量（grid/weight/两级熵模型参数/其他），用于分析码率构成"""
        grid_meta_em_params = self.grid_meta_em_param_model()
        grid_em_params = self.grid_em_param_model()
        weight_meta_em_params = self.weight_meta_em_param_model()
        weight_em_params = self.weight_em_param_model()
        params = self.param_model()
        grids, weights, others = self._split_params(params)
        num_params = {
            k: sum([p.numel()  if p is not None else 0 for p in v.values()])
            for k, v in {
                'grid': grids, 'weight': weights,
                'grid_em_params': grid_em_params, 'weight_em_params': weight_em_params,
                'grid_meta_em_params': grid_meta_em_params, 'weight_meta_em_params': weight_meta_em_params,
                'others': others,
            }.items()
        }
        return num_params

    def set_epoch(self, epoch, num_epochs):
        """
        每个训练 epoch 开始时调用：
            - 按进度 epoch_ratio 线性退火量化超参（软舍入温度、Kumaraswamy 噪声、QuantNoise 比例）；
            - 把退火后的超参与（可选的）初始位宽设置到各层熵模型上。
        """
        epoch_ratio = epoch / (num_epochs - 1) if num_epochs > 1 else 0.
        init_bit_width = self.config.optimize.init_bit_width \
                         if (not self.config.optimize.optimize_rate or epoch == 0) and \
                            self.config.optimize.init_bit_width != -1 else None
        soft_round_temp = self.config.optimize.soft_round_temp[0] + \
                          (self.config.optimize.soft_round_temp[1] - self.config.optimize.soft_round_temp[0]) * \
                          epoch_ratio
        kum_noise = self.config.optimize.kum_noise[0] + \
                    (self.config.optimize.kum_noise[1] - self.config.optimize.kum_noise[0]) * epoch_ratio
        quant_noise = self.config.optimize.quant_noise[0] + \
                      (self.config.optimize.quant_noise[1] - self.config.optimize.quant_noise[0]) * epoch_ratio

        # Model params
        params = self.param_model()
        grid, weight, _ = self._split_params(params)
        if isinstance(self.grid_em, _EntropyModelBase):
            self.grid_em.set_params(grid, init_bit_width,
                                    soft_round_temp, kum_noise, quant_noise,
                                    self.grid_em_param_model())
        if isinstance(self.weight_em, _EntropyModelBase):
            self.weight_em.set_params(weight, init_bit_width,
                                      soft_round_temp, kum_noise, quant_noise,
                                      self.weight_em_param_model())

        # Grid em params
        if isinstance(self.grid_meta_em, _EntropyModelBase):
            self.grid_meta_em.set_params(self.grid_em_param_model(), init_bit_width, 
                                         soft_round_temp, kum_noise, quant_noise,
                                         self.grid_meta_em_param_model())

        # Weight em params
        if isinstance(self.weight_meta_em, _EntropyModelBase):
            self.weight_meta_em.set_params(self.weight_em_param_model(), init_bit_width,
                                           soft_round_temp, kum_noise, quant_noise,
                                           self.weight_meta_em_param_model())

    def set_iter(self, iter, num_iters):
        """（按迭代设置，转发到 set_epoch）"""
        return self.set_epoch(iter, num_iters)

    def set_zero(self):
        """把所有子模型（参数模型 + 各级熵模型 + INR 主干）的参数全部清零"""
        for model in [self.param_model, self.grid_em_param_model, self.weight_em_param_model,
                      self.grid_meta_em_param_model, self.weight_meta_em_param_model,
                      self.model, self.grid_em, self.weight_em, self.grid_meta_em, self.weight_meta_em]:
            for param in model.parameters():
                param.data.zero_()

    def compress(self, bs):
        """
        编码（训练完成后调用）：按依赖顺序把全部参数写入码流——
            meta_em(psi) -> grid/weight meta em(phi) -> grid_em/weight_em(theta)。
        在确定性上下文 + no_grad 下执行，保证比特可复现。
        """
        self.eval()

        with DeterministicContext():
            with torch.no_grad():
                # Grid meta params（第三层：先压 psi，因为解码 phi 需要它）
                grid_meta_em_params = self.grid_meta_em_param_model()
                bs, _, grid_meta_em_params = self.meta_em.compress(bs, grid_meta_em_params)

                # Weight meta params
                weight_meta_em_params = self.weight_meta_em_param_model()
                bs, _, weight_meta_em_params = self.meta_em.compress(bs, weight_meta_em_params)

                # Grid em params
                grid_em_params = self.grid_em_param_model()
                bs, _, grid_em_params = self.grid_meta_em.compress(bs, grid_em_params, grid_meta_em_params)

                # Weight em params
                weight_em_params = self.weight_em_param_model()
                bs, _, weight_em_params = self.weight_meta_em.compress(bs, weight_em_params, weight_meta_em_params)

                # Model params（模块 C：残差域编码——先减 θ' 再喂熵模型）
                params = self.param_model()
                # C5 渐进剪枝：压缩时同样先掩码
                if self.pruner is not None:
                    params = self.pruner.apply(params)
                ref = self._residual_ref
                if ref is not None:
                    params = {k: (v - ref[k].to(device=v.device, dtype=v.dtype) if k in ref else v)
                              for k, v in params.items()}
                grid, weight, others = self._split_params(params)
                bs, _, _ = self.grid_em.compress(bs, grid, grid_em_params)
                bs, _, _ = self.weight_em.compress(bs, weight, weight_em_params)
                bs, _, _ = self.meta_em.compress(bs, others)

        return bs

    def decompress(self, bs):
        """
        解码：按与 compress 相同的顺序从码流恢复参数，并用 ParamModel.load() 把
        解码后的权重写回各模型（INR 主干因此获得量化后的重建权重）。
        """
        self.eval()

        with DeterministicContext():
            with torch.no_grad():
                # Grid meta params（逐层恢复 psi -> phi -> theta）
                grid_meta_em_params = self.grid_meta_em_param_model()
                bs, _, grid_meta_em_params = self.meta_em.decompress(bs, grid_meta_em_params)

                # Weight meta params
                weight_meta_em_params = self.weight_meta_em_param_model()
                bs, _, weight_meta_em_params = self.meta_em.decompress(bs, weight_meta_em_params)

                # Grid em params
                grid_em_params = self.grid_em_param_model()
                bs, _, grid_em_params = self.grid_meta_em.decompress(bs, grid_em_params, grid_meta_em_params)

                # Weight em params
                weight_em_params = self.weight_em_param_model()
                bs, _, weight_em_params = self.weight_meta_em.decompress(bs, weight_em_params, weight_meta_em_params)

                # Model params（模块 C：解码出的残差加回 θ' → 实际权重域）
                params = self.param_model()
                ref = self._residual_ref
                grid, weight, others = self._split_params(params)
                bs, _, grid = self.grid_em.decompress(bs, grid, grid_em_params)
                bs, _, weight = self.weight_em.decompress(bs, weight, weight_em_params)
                bs, _, others = self.meta_em.decompress(bs, others)
                if ref is not None:
                    def _add_ref(d):
                        return {k: (v + ref[k].to(device=v.device, dtype=v.dtype) if k in ref else v)
                                for k, v in d.items()}
                    grid, weight, others = _add_ref(grid), _add_ref(weight), _add_ref(others)

                self.grid_meta_em_param_model.load(grid_meta_em_params)
                self.weight_meta_em_param_model.load(weight_meta_em_params)
                self.grid_em_param_model.load(grid_em_params)
                self.weight_em_param_model.load(weight_em_params)
                self.param_model.load({**grid, **weight, **others})

        return bs

    # ---------- MyHiNeRV 迁移（模块 C）：残差参考注入 ----------
    def set_residual_ref(self, ref_params):
        """注入残差编码参考 θ'（扁平参数 dict；None = 关闭，恢复 intra 模式）。
        调用时机：GoP 初始化后、训练开始前（θ' 同时是 II-init 的初始化权重）。"""
        if ref_params is None:
            self._residual_ref = None
            return
        self._residual_ref = {k: v.detach().clone() for k, v in ref_params.items()}

    def snapshot_params(self):
        """导出当前（量化/解码后的）实际权重快照（扁平参数 dict，CPU 张量）。
        用途：a) 下一 GoP 的 II-init 来源 θ*_prev；b) 调试对拍。"""
        params = self.param_model()
        return {k: v.detach().float().cpu().clone() for k, v in params.items()
                if v is not None and torch.is_tensor(v)}

    def get_rate_stat(self):
        """返回各部分（7 类）参数的估计码率、数量与平均每参数比特数"""
        with torch.no_grad():
            rates, _ = self._forward_rates_params(x=None, forward_rates=True, forward_params=False)
            rates = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in rates.items()}
            num_params = self.get_num_params()
            num_params = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in num_params.items()}
            bits_per_params = {k: rates[k] / num_params[k] if num_params[k] > 0 else math.nan for k in rates.keys()}
        return rates, num_params, bits_per_params


def build_compress_model(logger, model, cfg):
    # Build NVRC model（工厂函数：由配置构建 NVRC 编解码器）
    return NVRC(logger, model, config=NVRCConfig(**cfg))