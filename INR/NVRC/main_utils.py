from utils import *
from io_utils import *
import datasets
import tasks
import compress_models
import entropy_models
import entropy_models.grid_entropy_model
import models

"""
Common settings
"""
# For reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

dynamo.config.verbose = True
dynamo.config.cache_size_limit = 24
dynamo.config.compiled_autograd = True

deepspeed.utils.logger.setLevel('WARNING')


@dataclass
class ExperimentConfig:
    # Training（训练设置：epochs/轮询评估/交替优化步数 rate_steps 等）
    epochs: int = 360
    warmup_epochs: int = 30
    eval_epochs: int = 30
    eval_only: bool = False
    rate_steps: int = None  # D/R 交替优化周期 K：每 K 个失真步执行 1 个率步（论文取 8）

    # Batch size
    grad_accum: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 1

    # Optimizer & scheduler
    opt: str = 'adam'
    sched: str = 'cosine'
    lr: float = 1e-4
    warmup_lr: float = 1e-7
    min_lr: float = 1e-7
    auto_lr_scaling: bool = False
    max_norm: float = 1.0
    norm_type: float = 2.0
    weight_decay: float = 1e-6
    weight_decay_scaling: bool = True
    opt_eps: float = 1e-8
    opt_betas: list = field(default_factory=lambda: [0.9, 0.999])
    momentum: float = 0.9
    decay_milestones: list = field(default_factory=lambda: [0.5, 0.75])
    decay_rate: float = 0.5
    lr_scale: list = field(default_factory=lambda: [])

    # Coding config
    start_frame: int = 0
    num_frames: int = -1
    intra_period: int = -1

    # Output
    output: str = 'output'
    exp_name: str = None

    # Logging
    log_epochs: int = -2
    log_iters: int = -2
    train_enable_log: bool = False
    eval_enable_log: bool = False

    # Resuming
    resume: str = None
    resume_model_only: bool = False
    resume_model_pat: list = None

    # Misc
    seed: int = 0
    workers: int = 4
    prefetch_factor: int = 4
    pin_mem: bool = True

    # Nested configs
    train_data = None
    eval_data = None
    train_task = None
    eval_task = None
    compress_model = None
    model = None

    def __post_init__(self):
        for f in fields(self):
            if f.type in (int, float) and getattr(self, f.name) is not None:
                setattr(self, f.name, f.type(float(getattr(self, f.name))))

    def _to_dict(self):
        output = {k: getattr(self, k) for k in self.__dict__}
        output['train_data'] = self.train_data.__dict__ if self.train_data is not None else None
        output['eval_data'] = self.eval_data.__dict__ if self.eval_data is not None else None
        output['train_task'] = self.train_task.__dict__ if self.train_task is not None else None
        output['eval_task'] = self.eval_task.__dict__ if self.eval_task is not None else None
        return output


@dataclass
class DataConfig:
    # 数据配置：数据集目录/名称、视频尺寸（-1 表示自动推断）、patch 尺寸与格式
    dataset_dir: str = '~/Datasets/PNG/UVG/1920x1080'
    dataset: str = 'Beauty'
    video_size: list = field(default_factory=lambda: [-1, -1, -1])
    patch_size: list = field(default_factory=lambda: [1, -1, -1])
    fmt: str = 'png'


@dataclass
class TrainingTaskConfig:
    loss: list = field(default_factory=lambda: ['mse'])
    metric: list = field(default_factory=lambda: ['psnr'])
    color_space: str = 'RGB'
    lamb: list = field(default_factory=lambda: [1.0])
    enable_log: bool = False


@dataclass
class OverfitTaskConfig:
    loss: str = 'mse'
    metric: str = 'psnr'
    color_space: str = 'RGB'
    lamb: float = 1.0
    enable_log: bool = False


def parse_args():
    """解析命令行参数并加载各 YAML 配置（实验/数据/任务/压缩模型/INR 模型），命令行参数可覆盖 YAML 中的同名项"""
    parser = argparse.ArgumentParser(description='NVRC training script', allow_abbrev=False)
    group = parser.add_argument_group('Script parameters')
    group.add_argument('--exp-config', type=str, default='scripts/configs/nvrc/overfit/s1-360e.yaml',
                       help='path to experiment config')
    group.add_argument('--train-data-config', type=str, default='scripts/configs/data/video/png/uvg.yaml',
                       help='path to training data config')
    group.add_argument('--eval-data-config', type=str, default=None,
                       help='path to evaluation data config')
    group.add_argument('--train-task-config', type=str, default='scripts/configs/tasks/overfit/l1_ms-ssim.yaml',
                       help='path to training task config')  
    group.add_argument('--eval-task-config', type=str, default=None,
                       help='path to evaluation task config')
    group.add_argument('--compress-model-config', type=str, default='scripts/configs/nvrc/compress_models/nvrc.yaml',
                       help='compression model name')
    group.add_argument('--model-config', type=str, default='scripts/configs/nvrc/models/uvg_hinerv-v2-s_1920x1080.yaml',
                       help='model name')

    # Arguments for overriding (default: None)
    group.add_argument('--train-dataset-dir', type=str, default=None, help='Training dataset directory')
    group.add_argument('--train-dataset', type=str, default=None, help='Training dataset name')
    group.add_argument('--train-video-size', type=int, nargs='+', default=None, help='Train video size (with padding)')
    group.add_argument('--train-patch-size', type=int, nargs='+', default=None, help='Train patch size (with padding)')
    group.add_argument('--train-fmt', type=str, default=None, help='Train video format (e.g. png, yuv420p)')

    group.add_argument('--eval-dataset-dir', type=str, default=None, help='Evaluation dataset directory')
    group.add_argument('--eval-dataset', type=str, default=None, help='Evaluation dataset name')
    group.add_argument('--eval-video-size', type=int, nargs='+', default=None, help='Eval video size (with padding)')
    group.add_argument('--eval-patch-size', type=int, nargs='+', default=None, help='Eval patch size (with padding)')
    group.add_argument('--eval-fmt', type=str, default=None, help='Eval video format (e.g. png, yuv420p)')

    group.add_argument('--train-loss', type=str, nargs='+', default=None, help='Loss function for training')
    group.add_argument('--train-metric', type=str, nargs='+', default=None, help='Metric for training')
    group.add_argument('--eval-loss', type=str, nargs='+', default=None, help='Loss function for evaluation')
    group.add_argument('--eval-metric', type=str, nargs='+', default=None, help='Metric for evaluation')
    group.add_argument('--train-enable-log', type=str_to_bool, default=None, help='Enable logging for training')
    group.add_argument('--eval-enable-log', type=str_to_bool, default=None, help='Enable logging for evaluation')
    group.add_argument('--color-space', type=str, default=None, help='Color space for training')
    group.add_argument('--lamb', type=float, nargs='+', default=None, help='Lambda for training')

    group.add_argument('--use-epochs', type=str_to_bool, default=None, help='Epoch or iteration for training')
    group.add_argument('--epochs', type=int, default=None, help='Number of epochs for training')
    group.add_argument('--warmup-epochs', type=int, default=None, help='Number of warmup epochs')
    group.add_argument('--eval-epochs', type=int, default=None, help='Number of epochs for evaluation')
    group.add_argument('--eval-only', type=str_to_bool, default=None, help='Evaluate only')
    group.add_argument('--rate-steps', type=int, default=None, help='Rate steps for evaluation')

    group.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    group.add_argument('--train-batch-size', type=int, default=1, help='Batch size for training')
    group.add_argument('--eval-batch-size', type=int, default=1, help='Batch size for evaluation')

    group.add_argument('--opt', type=str, default=None, help='Optimizer')
    group.add_argument('--sched', type=str, default=None, help='Scheduler')
    group.add_argument('--lr', type=float, default=None, help='Learning rate')
    group.add_argument('--warmup-lr', type=float, default=None, help='Warmup learning rate')
    group.add_argument('--min-lr', type=float, default=None, help='Minimum learning rate')
    group.add_argument('--auto-lr-scaling', type=str_to_bool, default=None, help='Auto scaling learning rate')
    group.add_argument('--max-norm', type=float, default=None, help='Maximum norm for gradient clipping')
    group.add_argument('--weight-decay', type=float, default=None, help='Weight decay')
    group.add_argument('--weight-decay-scaling', type=str_to_bool, default=None, help='Weight decay scaling')
    group.add_argument('--decay-milestones', type=float, nargs='+', default=None, help='Decay milestones')
    group.add_argument('--decay-rate', type=float, default=None, help='Decay rate')

    group.add_argument('--start-frame', type=int, default=None, help='Start frame for coding')
    group.add_argument('--num-frames', type=int, default=None, help='Number of frames for coding')
    group.add_argument('--intra-period', type=int, default=None, help='Intra period for coding')

    group.add_argument('--output', type=str, default=None, help='Output directory')
    group.add_argument('--exp-name', type=str, default=None, help='Experiment name')

    group.add_argument('--log-epochs', type=int, default=None, help='Log every N epochs')

    group.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    group.add_argument('--resume-model-only', type=str_to_bool, default=False, help='Resume model only')
    group.add_argument('--bitstream', type=str, default=None, help='Bitstream file for evaluation')

    group.add_argument('--seed', type=int, default=None, help='Random seed')
    group.add_argument('--workers', type=int, default=None, help='Number of workers for data loading')
    group.add_argument('--prefetch-factor', type=int, default=None, help='Prefetch factor for data loading')
    group.add_argument('--pin-mem', type=str_to_bool, default=None, help='Pin memory for data loading')
    args_raw = parser.parse_args()

    def _load_yaml_config(config):
        if config is not None:
            with open(config, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {}

    # Experiment config
    args = ExperimentConfig(**_load_yaml_config(args_raw.exp_config))

    # Data config
    args.train_data = DataConfig(**_load_yaml_config(args_raw.train_data_config))
    args.eval_data = DataConfig(**_load_yaml_config(args_raw.eval_data_config)) \
                     if args_raw.eval_data_config else copy.deepcopy(args.train_data)

    # Task config
    args.train_task = OverfitTaskConfig(**_load_yaml_config(args_raw.train_task_config))
    args.eval_task = OverfitTaskConfig(**_load_yaml_config(args_raw.eval_task_config)) \
                                       if args_raw.eval_task_config else copy.deepcopy(args.train_task)

    # Compression model config
    args.compress_model = _load_yaml_config(args_raw.compress_model_config)

    # Model config
    args.model = _load_yaml_config(args_raw.model_config)

    # Override configs
    assert ((args_raw.train_dataset_dir is None) == (args_raw.train_dataset is None)) and \
           ((args_raw.eval_dataset_dir is None) == (args_raw.eval_dataset is None)), \
        'The dataset_dir and dataset should be set together'
    if args_raw.train_dataset_dir is not None and args_raw.eval_dataset_dir is None:
        args_raw.eval_dataset_dir = args_raw.train_dataset_dir
        args_raw.eval_dataset = args_raw.train_dataset
        args_raw.eval_fmt = args_raw.train_fmt

    # Train/eval dataset
    for k in ['dataset_dir', 'dataset', 'video_size', 'patch_size', 'fmt']:
        if getattr(args_raw, f'train_{k}') is not None:
            assert hasattr(args.train_data, k), f'The {k} should be in the data config'
            setattr(args.train_data, k, getattr(args_raw, f'train_{k}'))
        if getattr(args_raw, f'eval_{k}') is not None:
            assert hasattr(args.eval_data, k), f'The {k} should be in the data config'
            setattr(args.eval_data, k, getattr(args_raw, f'eval_{k}'))

    # Train/eval task
    for k in ['loss', 'metric', 'enable_log']:
        if getattr(args_raw, f'train_{k}') is not None:
            assert hasattr(args.train_task, k), f'The {k} should be in the task config'
            setattr(args.train_task, k, getattr(args_raw, f'train_{k}'))
        if getattr(args_raw, f'eval_{k}') is not None:
            assert hasattr(args.eval_task, k), f'The {k} should be in the task config'
            setattr(args.eval_task, k, getattr(args_raw, f'eval_{k}'))
    for k in ['color_space', 'lamb']:
        if getattr(args_raw, k) is not None:
            assert hasattr(args.train_task, k), f'The {k} should be in the task config'
            setattr(args.train_task, k, getattr(args_raw, k))
            setattr(args.eval_task, k, getattr(args_raw, k))

    # Train config
    for k in ['use_epochs', 'epochs', 'warmup_epochs',
              'eval_epochs', 'eval_only', 'rate_steps',
              'grad_accum', 'train_batch_size', 'eval_batch_size',
              'lr', 'warmup_lr', 'min_lr', 'auto_lr_scaling', 'max_norm', 'weight_decay', 'weight_decay_scaling',
              'opt', 'sched', 'decay_milestones', 'decay_rate']:
        if getattr(args_raw, k) is not None:
            assert hasattr(args, k), f'The {k} should be in the config'
            setattr(args, k, getattr(args_raw, k))

    # Coding config
    for k in ['start_frame', 'num_frames', 'intra_period']:
        if getattr(args_raw, k) is not None:
            assert hasattr(args, k), f'The {k} should be in the config'
            setattr(args, k, getattr(args_raw, k))

    # Others
    for k in ['output', 'exp_name',
              'log_epochs',
              'resume', 'resume_model_only', 'bitstream',
              'seed', 'workers', 'prefetch_factor', 'pin_mem']:
        if getattr(args_raw, k) is not None:
            assert hasattr(args, k), f'The {k} should be in the config'
            setattr(args, k, getattr(args_raw, k))

    return args


def parse_yuv_name(yuv_name):
    # e.g. XXXXX_1920x1080_yuv420p（从 YUV 文件名解析尺寸与格式）
    size = [-1] + list(reversed([int(s) for s in yuv_name.split('_')[1].split('x')]))
    fmt = yuv_name.split('_')[2]
    return size, fmt


def profile_model(logger, model, loader, task, get_macs=True):
    """统计模型参数量（INR / 含编解码器）与推理 MACs（用 deepspeed flops_profiler，分别统计 D 步与 R 步）"""
    class _ProfilingModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, compute_outputs=True, compute_rates=True):
            return self.model(x, compute_outputs=compute_outputs, compute_rates=compute_rates)

    with torch.no_grad():
        inputs, _ = task.parse_batch(next(iter(loader)))
    
        # Compute number of parameters
        num_params = sum(v.numel() for v in model.model.parameters()) + sum(v.numel() for v in model.param_model.parameters())
        num_params_codec = sum(v.numel() for v in model.parameters())
        logger.info(f'Parameters (model): {num_params / 10**6:.2f}M')
        logger.info(f'Parameters (w/ codec): {num_params_codec / 10**6:.2f}M')

        if hasattr(model, 'get_num_params'):
            num_params_detailed = model.get_num_params()
            logger.info(f'Parameters (detailed):')
            for k in num_params_detailed:
                logger.info(f'    {k}: {num_params_detailed[k] / 10**6:.2f}M')

        if get_macs:
            # Compute MACs
            _profile_model = _ProfilingModel(model)
            _profile_model.eval()
            _, num_macs, _ = deepspeed.profiling.flops_profiler.get_model_profile(model=_profile_model,
                                                                                  args=[inputs, True, False],
                                                                                  print_profile=False,
                                                                                  detailed=False,
                                                                                  warm_up=1, as_string=False)
            _, num_macs_rate, _ = deepspeed.profiling.flops_profiler.get_model_profile(model=_profile_model,
                                                                                       args=[inputs, False, True],
                                                                                       print_profile=False,
                                                                                       detailed=False,
                                                                                       warm_up=1, as_string=False)
            num_macs /= inputs['rel_batch_size']
            logger.info(f'MACs: {num_macs / 10 ** 9 :.2f}G')
            logger.info(f'MACs (Rate): {num_macs_rate / 10 ** 9 :.2f}G')
        else:
            num_macs, num_macs_rate = np.nan, np.nan
        return num_params, num_params_codec, num_macs, num_macs_rate


def profile_model_rate_stat(logger, model, loader, task):
    """打印各部分参数（grid/weight/熵模型参数等）的数量、码率与平均每参数比特数"""
    with torch.no_grad():
        rates, num_params, bits_per_params = model.get_rate_stat()
        logger.info(f'Detailed rate statistics:')
        for k in rates.keys():
            logger.info(f'{k} - Number of params: {num_params[k]:.4f}    Rate: {rates[k]:.4f}    Bits per param: {bits_per_params[k]:.4f}')


def train_epoch(output_dir, name, epoch, args, logger, accelerator, model, loader, task, optimizer, scheduler, log_output):
    """
    单个训练 epoch（NVRC 的核心训练流程）：
        1. set_epoch(): 按训练进度更新量化器超参（软舍入温度 / Kumaraswamy 噪声 / QuantNoise 比例）；
        2. 对每个 batch 执行“失真步”d_step（重建 + D = lambda * L）；
        3. 若启用交替优化（rate_steps = K > 0）：每 K 个失真步才执行一次“率步”r_step，
           且率步的梯度乘以 K（保证与每步联合优化等效），每 K 步才更新一次参数；
           否则每个 batch 都同时计算 D + R 并逐步更新（普通联合优化）；
        4. 梯度累积（grad_accum）、梯度裁剪（max_norm = 1.0）、scheduler 步进与日志。
    """
    start_time = time.time()
    model.train()

    accum_loss = None
    accum_metrices = None
    accum_rate_loss = None
    accum_rate = None
    reduced_loss = None
    reduced_metrics = None
    reduced_rate_loss = None
    reduced_rate = None
    samples = 0

    unwrap_model(model).set_epoch(epoch, args.epochs)  # 更新量化/熵模型的退火超参
    update_optimizer_scheduler(epoch, args.epochs, args, optimizer, scheduler)  # 权重衰减随进度线性衰减
    scheduler.step(epoch=epoch)
    optimizer.zero_grad()
    num_updates = epoch * len(loader)
    alternate_rate_steps = args.rate_steps is not None and args.rate_steps > 0  # 是否启用 D/R 交替优化
    rate_steps = args.rate_steps if args.rate_steps is not None and args.rate_steps > 0 else 1
    assert args.grad_accum * rate_steps <= len(loader), \
        'grad_accum * rate_steps should be less than the number of batches in the loader'
    print_every = max(args.grad_accum * rate_steps, len(loader) // 4)


    def _d_forward_step():
        inputs, target, output, loss, metrics = task.d_step(model, batch)
        return inputs, target, output, loss, metrics

    def _r_forward_step():
        rate, rate_loss = task.r_step(model, batch)
        return rate, rate_loss

    def _grad_scale_step():
        if accelerator.sync_gradients:
            accelerator.unscale_gradients(optimizer)
            if args.max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm, args.norm_type)

    def _opt_step():
        optimizer.step()
        optimizer.zero_grad()

    if accelerator.state.dynamo_plugin.backend != accelerate.utils.DynamoBackend.NO:
        _d_forward_step = torch.compile(_d_forward_step, **accelerator.state.dynamo_plugin.to_kwargs())
        _r_forward_step = torch.compile(_r_forward_step, **accelerator.state.dynamo_plugin.to_kwargs())
        _grad_scale_step = torch.compile(_grad_scale_step, **accelerator.state.dynamo_plugin.to_kwargs())

    for i, batch in enumerate(loader):
        # Distorion step（失真步：只反传 D 损失；若非交替模式则同时计算 D + R）
        inputs, _, outputs, loss, metrics = _d_forward_step()
        if alternate_rate_steps:
            accelerator.backward(loss / args.grad_accum)
        else:
            rate, rate_loss = _r_forward_step()
            accelerator.backward((loss + rate_loss) / args.grad_accum)

        if (i + 1) % args.grad_accum == 0:
            _grad_scale_step()
            _opt_step()

        # Rate step（率步：每 grad_accum * K 个 batch 执行一次，梯度乘 K 以保持等效更新量）
        if alternate_rate_steps and (i + 1) % (args.grad_accum * rate_steps) == 0:
            rate, rate_loss = _r_forward_step()
            if isinstance(rate_loss, torch.Tensor) and rate_loss.requires_grad:
                accelerator.backward(rate_steps * rate_loss)
                _grad_scale_step()
                _opt_step()

        # Update scheduler
        if (i + 1) % args.grad_accum == 0:
            num_updates += args.grad_accum
            scheduler.step_update(num_updates=num_updates)

        # Accumulate loss and metrics
        accum_loss = loss.detach() if accum_loss is None else accum_loss + loss.detach()
        accum_metrices = {k: metrics[k].mean() for k in metrics} \
                          if accum_metrices is None else {k: accum_metrices[k] + metrics[k].mean() for k in metrics}
        if (i + 1) % (args.grad_accum * rate_steps) == 0:
            accum_rate_loss = rate_loss.detach() if accum_rate_loss is None else accum_rate_loss + rate_loss.detach()
            accum_rate = rate.detach() if accum_rate is None else accum_rate + rate.detach()

        samples += inputs['rel_batch_size'] * accelerator.num_processes

        if ((i + 1) % print_every == 0) and accum_rate_loss is not None:
            # Compute averging loss and metrics globally
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean').item() / (i + 1)
            reduced_metrics = {k: v.item() / (i + 1) \
                               for k, v in accelerator.reduce(accum_metrices, reduction='mean').items()}
            reduced_rate_loss = accelerator.reduce(accum_rate_loss, reduction='mean').item() / \
                                ((i + 1) // (args.grad_accum * rate_steps))
            reduced_rate = accelerator.reduce(accum_rate, reduction='mean').item() / \
                           ((i + 1) // (args.grad_accum * rate_steps))
            # Logging loss & metrics
            log_msg = f'Train' +  f' - Epoch {epoch} [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    lr: {optimizer.param_groups[0]['lr']:.2e}'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss + reduced_rate_loss:.4f}'
            log_msg = log_msg + f'    r_loss: {reduced_rate_loss:.4f}'
            log_msg = log_msg + f'    d_loss: {reduced_loss:.4f}'
            log_msg = log_msg + f'    bpp: {reduced_rate:.4f}'
            log_msg = log_msg + ''.join(f'    {k}: {v:.4f}' for k, v in reduced_metrics.items())
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_outputs(inputs=inputs, outputs=outputs, metrics=metrics)

    # Flush outputs
    if log_output:
        task.flush(dir=os.path.join(output_dir, 'outputs', name, f'{epoch}'))

    return reduced_loss + reduced_rate_loss, {'bpp': reduced_rate, **reduced_metrics}


def eval_epoch(output_dir, name, epoch_or_str, args, logger, accelerator, model, loader, task, log_output):
    """
    评估 epoch：先 load_cache() 一次性缓存量化后的参数与码率（避免重复熵编码前向），
    逐 patch 计算失真指标，每 rate_steps 个 batch 计算一次码率，最后 clear_cache()。
    """
    start_time = time.time()
    model.eval()

    accum_loss = None
    accum_metrices = None
    accum_rate_loss = None
    accum_rate = None
    reduced_loss = None
    reduced_metrics = None
    reduced_rate_loss = None
    reduced_rate = None
    samples = 0
    rate_steps = args.rate_steps if args.rate_steps is not None and args.rate_steps > 0 else 1
    assert args.grad_accum * rate_steps <= len(loader), \
        'grad_accum * rate_steps should be less than the number of batches in the loader'
    print_every = max(args.grad_accum * rate_steps, len(loader) // 4)

    model.load_cache()  # 缓存量化参数与估计码率，整个评估过程复用

    def _d_forward_step():
        inputs, target, output, loss, metrics = task.d_step(model, batch)
        return inputs, target, output, loss, metrics

    def _r_forward_step():
        rate, rate_loss = task.r_step(model, batch)
        return rate, rate_loss

    if accelerator.state.dynamo_plugin.backend != accelerate.utils.DynamoBackend.NO:
        _d_forward_step = torch.compile(_d_forward_step, **accelerator.state.dynamo_plugin.to_kwargs())
        _r_forward_step = torch.compile(_r_forward_step, **accelerator.state.dynamo_plugin.to_kwargs())

    for i, batch in enumerate(loader):
        # Eval step
        with torch.no_grad():
            # Distorion step
            inputs, _, outputs, loss, metrics = _d_forward_step()
            mean_loss = loss.mean()
            mean_metrics = {k: v.mean() for k, v in metrics.items()}

            # Rate step
            if (i + 1) % rate_steps == 0:
                rate, rate_loss = _r_forward_step()

        # Accumulate loss and metrics
        accum_loss = mean_loss.detach() if accum_loss is None else accum_loss + mean_loss.detach()
        accum_metrices = {k: mean_metrics[k].mean() \
                             if accum_metrices is None else accum_metrices[k] + mean_metrics[k].mean() 
                          for k in mean_metrics}
        if (i + 1) % rate_steps == 0:
            accum_rate_loss = rate_loss.detach() if accum_rate_loss is None else accum_rate_loss + rate_loss.detach()
            accum_rate = rate.detach() if accum_rate is None else accum_rate + rate.detach()

        samples += inputs['rel_batch_size'] * accelerator.num_processes

        if ((i + 1) % print_every == 0):
            # Compute averging loss and metrics globally
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean').item() / (i + 1)
            reduced_metrics = {k: v.item() / (i + 1) \
                               for k, v in accelerator.reduce(accum_metrices, reduction='mean').items()}
            reduced_rate_loss = accelerator.reduce(accum_rate_loss, reduction='mean').item() / ((i + 1) // rate_steps)
            reduced_rate = accelerator.reduce(accum_rate, reduction='mean').item() / ((i + 1) // rate_steps)
            # Logging loss & metrics
            log_msg = f'Eval' +  f' - [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss + reduced_rate_loss:.4f}'
            log_msg = log_msg + f'    r_loss: {reduced_rate_loss:.4f}'
            log_msg = log_msg + f'    d_loss: {reduced_loss:.4f}'
            log_msg = log_msg + f'    bpp: {reduced_rate:.4f}'
            log_msg = log_msg + ''.join(f'    {k}: {v:.4f}' for k, v in reduced_metrics.items())
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_outputs(inputs=inputs, outputs=outputs, metrics=metrics)

    model.clear_cache()

    # Flush outputs
    if log_output:
        task.flush(dir=os.path.join(output_dir, 'outputs', name, f'{epoch_or_str}'))

    return reduced_loss + reduced_rate_loss, {'bpp': reduced_rate, **reduced_metrics}


def do_eval(args, epoch, num_epochs):
    """（是否在本 epoch 结束后评估：每 eval_epochs 轮或最后一轮）"""
    return (epoch + 1) % args.eval_epochs == 0 or (epoch + 1) == num_epochs


def do_log(args, epoch, num_epochs):
    """（是否在本 epoch 输出重建结果）"""
    return (args.log_epochs > 0 and (epoch + 1) % args.log_epochs == 0) or \
           (args.log_epochs == -1 and (epoch + 1) == num_epochs)


def create_encoding_decoding_videos(output_dir, args, logger, training=True, input_only=False):
    # Create videos（按配置创建输入视频对象（PNG/YUV/PNG 图像）与输出重建视频对象，并设置 padding 与 patch 尺寸）
    if training:
        data_cfg = args.train_data
    else:
        data_cfg = args.eval_data

    logger.info(f'Input video: {os.path.join(data_cfg.dataset_dir, data_cfg.dataset)}')
    if data_cfg.fmt == 'png':
        # Input videos
        video_in = PNGVideo(os.path.join(data_cfg.dataset_dir, data_cfg.dataset),
                             (-1, -1, -1), -1, -1, False)
        # Output videos
        if not input_only:
            video_out = PNGVideo(os.path.join(output_dir, 'train' if training else 'eval', data_cfg.dataset),
                                 video_in.get_video_size(), video_in.get_num_channels(),
                                 video_in.get_bit_depth(), True)
    elif data_cfg.fmt == 'yuv420p':
        # Input videos
        yuv_size, yuv_fmt = parse_yuv_name(data_cfg.dataset)
        video_in = YUVVideo(os.path.join(data_cfg.dataset_dir, data_cfg.dataset + '.yuv'),
                             yuv_size, yuv_fmt, False)
        # Output videos
        if not input_only:
            video_out = YUVVideo(os.path.join(output_dir, 'train' if training else 'eval', data_cfg.dataset + '.yuv'),
                                 video_in.get_video_size(), video_in.get_fmt(), True)
    elif data_cfg.fmt == 'png_image':
       # Input image
        video_in = PngImage(os.path.join(data_cfg.dataset_dir, data_cfg.dataset),
                             (-1, -1, -1), -1, -1, False)
        # Output image
        if not input_only:
            video_out = PngImage(os.path.join(output_dir, 'train' if training else 'eval', data_cfg.dataset),
                                 video_in.get_video_size(), video_in.get_num_channels(),
                                 video_in.get_bit_depth(), True)
    else:
        raise ValueError

    # Set video settings
    assert all(video_in.get_video_size()[d] == video_out.get_video_size()[d] for d in range(3)), \
        'The videos should have the same size'
    assert all(video_in.get_patch_size()[d] == video_out.get_patch_size()[d] for d in range(3)), \
        'The videos should have the same patch size'
    assert all(video_in.get_num_channels() == video_out.get_num_channels() for d in range(3)), \
        'The videos should have the same number of channels'
    assert (video_in.patch_size[0] in [-1, 1]) or not args.mode == 'low-delay', \
        'The temporal patch size should be 1 for low-delay mode'
    padding = []
    for d in range(3):
        if data_cfg.video_size[d] != -1:
            padding.append([data_cfg.video_size[d] - video_in.get_video_size()[d] - \
                            (data_cfg.video_size[d] - video_in.get_video_size()[d]) // 2,
                            (data_cfg.video_size[d] - video_in.get_video_size()[d]) // 2])
        else:
            padding.append([0, 0])

    video_in.set_padding(padding)
    patch_size = [
        data_cfg.patch_size[0] if data_cfg.patch_size[0] != -1 else 1,
        data_cfg.patch_size[1] if data_cfg.patch_size[1] != -1 else video_in.get_patch_size()[1] + sum(padding[1]),
        data_cfg.patch_size[2] if data_cfg.patch_size[2] != -1 else video_in.get_patch_size()[2] + sum(padding[2])
    ]
    video_in.set_patch_size(patch_size)

    if not input_only:
        video_out.set_padding(padding)
        video_out.set_patch_size(patch_size)

    logger.info(f'{'Train' if training else 'Eval'} video settings:')
    logger.info(f'     Video size: {video_in.get_video_size()}    Patch size: {video_in.get_patch_size()}')
    logger.info(f'     Padding: {padding}')

    if not input_only:
        return video_in, video_out
    else:
        return video_in


def create_overfit_dataset(*args,**kwargs):
    # 转发到 datasets.create_overfit_dataset
    return datasets.create_overfit_dataset(*args,**kwargs)


def create_overfit_loader(args, logger,  accelerator, dataset, start_frame=-1, num_frames=-1, training=True):
    """
    Create the overfit dataset loader.
    （为当前 GoP 创建 DataLoader：用 VideoSubset 截取帧范围，训练时打乱、drop_last）
    """
    # Create subset for supporting different latency settings
    if start_frame != -1 or num_frames != -1:
        start_frame = 0 if start_frame == -1 else start_frame
        num_frames = dataset.video.get_video_size()[0] - start_frame if num_frames == -1 else num_frames
        subset = datasets.VideoSubset(dataset, start_frame, num_frames)
    else:
        subset = dataset

    # Assert that the dataset is in frame mode & the batch size is 

    # Create loader
    loader_kwargs = {
        'dataset': subset,
        'batch_size': args.train_batch_size if training else args.eval_batch_size,
        'shuffle': training,
        'num_workers': args.workers,
        'pin_memory': args.pin_mem,
        'drop_last': True,
    }
    if args.workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    return torch.utils.data.DataLoader(**loader_kwargs)


def create_overfit_task(*args,**kwargs):
    # 转发到 tasks.create_overfit_task
    return tasks.create_overfit_task(*args,**kwargs)


def create_codec(output_dir, name, args, logger, accelerator, loader, task):
    # Create codec（先构建 INR 主干 HiNeRV，再把它包装成 NVRC 压缩模型）
    # INR Model
    logger.info(f'Create model: {args.model['type']}')
    model = models.create_model(args.model, logger, task.parse_batch(next(iter(loader)))[0])

    # Codec
    logger.info(f'Create compression model: {args.compress_model['type']}')
    compress_model = compress_models.create_compress_model(logger, model, cfgs=args.compress_model)
    return compress_model


def profile_codec(output_dir, name, args, logger, accelerator, model, loader, task, get_macs=True):
    # Compute number of parameters & MACs (with the original model)（统计并保存模型参数量/MACs 信息）
    num_params, num_params_codec, num_macs, num_macs_rate = profile_model(logger, unwrap_model(model), loader, task,
                                                                          get_macs)

    # Log model info
    model_info = [f'Model info:', str(model), f'Full model info:', str(model),
                    f'Parameters: {num_params / 10**6:.2f}M',
                    f'Parameters (w/ codec): {num_params_codec / 10**6:.2f}M',
                    f'MACs: {num_macs / 10 ** 9 :.2f}G', f'MACs (Rate): {num_macs_rate / 10 ** 9 :.2f}G']

    os.makedirs(os.path.join(output_dir, f'rank_{accelerator.process_index}', 'models'), exist_ok=True)
    with open(os.path.join(output_dir, f'rank_{accelerator.process_index}', 'models', f'{name}.txt'), 'w') as f:
        f.write('\n'.join(model_info))

    return num_params, num_params_codec, num_macs, num_macs_rate


def create_optimizer_scheduler(output_dir, name, args, logger, accelerator, model, loader):
    # 创建优化器与调度器；权重衰减只作用于 linear/conv 权重类参数（param_type 过滤）
    def _filter_param_for_weight_decay(k, v):
        if fnmatch.fnmatch(v.param_type, '*linear_weight') or fnmatch.fnmatch(v.param_type, '*conv_weight'):
            return True

    logger.info(f'Create optimizer: {args.opt}')
    logger.info(f'Create scheduler: {args.sched}')
    optimizer, scheduler = get_optimizer_scheduler(args, logger, accelerator, model, loader,
                                                   wd_filter_fn=_filter_param_for_weight_decay)

    os.makedirs(os.path.join(output_dir, f'rank_{accelerator.process_index}', 'opt_sch'), exist_ok=True)
    with open(os.path.join(output_dir, f'rank_{accelerator.process_index}', 'opt_sch', f'{name}.txt'), 'w') as f:
        f.write('\n'.join([str(optimizer), str(scheduler)]))

    return optimizer, scheduler


def update_optimizer_scheduler(epoch, num_epochs, args, optimizer, scheduler):
    # weight_decay_scaling：权重衰减随训练进度线性衰减到 0（论文第一阶段使用）
    epoch_ratio = epoch / (num_epochs - 1) if num_epochs > 1 else 0.
    if args.weight_decay_scaling:
        for group in optimizer.param_groups:
            group['weight_decay'] = group['weight_decay_orig'] * (1 - epoch_ratio)


def create_ckpt_and_restore(output_dir, name, args, logger, accelerator, model, metric=None):
    # Create checkpoint manager（创建 checkpoint 管理器；支持从上次实验恢复训练状态/仅恢复模型权重，用于两阶段训练衔接）
    ckpt_manager = CheckpointManager(os.path.join(output_dir, 'checkpoints'), name, logger,
                                     accelerator, metric)

    # Restore training state (For resume training)
    epoch = 0

    if args.resume:
        resume = os.path.expanduser(args.resume)
        if not args.resume_model_only:
            path = os.path.join(resume, 'checkpoints', name)
            if os.path.exists(path):
                logger.info(f'Restore from training state: {path}')
                epoch = ckpt_manager.load(path) + 1
            else:
                logger.info(f'Checkpoint does not exist: {path}')
        else:
            path = os.path.join(resume, 'checkpoints', name, 'pytorch_model.bin')
            state_dict = torch.load(path, map_location=accelerator.device)
            if os.path.exists(path):
                logger.info(f'Restore from model: {path}')
                # Include only the matched keys
                unwrap_model(model).load_state_dict(state_dict)
            else:
                logger.info(f'Checkpoint does not exist: {path}')
    else:
        logger.info(f'Start training from scratch.')


    # Restore from bitstream (For fine-tuning or evaluation)
    """
    if args.bitstream:
        logger.info(f'Restore model weights from bitstream: {args.bitstream}')
        model.decompress()
        num_bytes = entropy_model.decompress(args, logger, accelerator, model, '', args.bitstream)
        bits_per_pixel = num_bytes * 8 / math.prod(eval_dataset.video_size)
        logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
        logger.info(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')
        raise NotImplementedError
    """

    return epoch, ckpt_manager


def save_model(output_dir, name, epoch_or_str, model):
    # 保存模型 state_dict
    path = os.path.join(output_dir, 'outputs', name, f'{epoch_or_str}')
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'pytorch_model.bin'))


def write_bitstream(output_dir, name, bitstream):
    # 把码流写入 bitstreams/{name}.bits
    os.makedirs(os.path.join(output_dir, 'bitstreams'), exist_ok=True)
    with open(os.path.join(output_dir, 'bitstreams', f'{name}.bits'), 'wb') as f:
        f.write(bitstream)


def read_bitstream(output_dir, name):
    # 从 bitstreams/{name}.bits 读回码流
    with open(os.path.join(output_dir, 'bitstreams', f'{name}.bits'), 'rb') as f:
        return f.read()


def get_bitstream_size(output_dir, name):
    # 码流大小（bits）
    return os.path.getsize(os.path.join(output_dir, 'bitstreams', f'{name}.bits')) * 8


def set_zero(model):
    # 将模型全部参数清零（若模型提供 set_zero 则调用之），用于验证解码结果确来自码流
    if hasattr(model, 'set_zero'):
        model.set_zero()
    else:
        for param in model.parameters():
            param.data.zero_()


def write_results(output_dir, name, size, metrics):
    # 把码流大小与指标写入 results/{name}.txt（CSV 格式）
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    with open(os.path.join(output_dir, 'results', f'{name}.txt'), 'w') as f:
        f.write(','.join(['size'] + list(metrics.keys())) + '\n')
        f.write(','.join([str(size)] + [f'{metrics[k]:.4f}' for k in metrics.keys()]) + '\n')


def write_results_eval(output_dir, name, epoch_or_iter, args, logger, accelerator, metrics, others={},
                       write_results_to_logger=True, write_results_to_file=True):
    assert isinstance(metrics, (list, tuple))
    assert all(list(metrics_i.keys())[0].lower() in ('bpp', 'size') for metrics_i in metrics), \
        'The first metric should be bpp or size'

    if accelerator.is_main_process:
        keys = [k.title() if k == 'bpp' else k.upper() for k in metrics[0].keys()]
        values = [[float(metrics_i[k]) for k in metrics[0].keys()] for metrics_i in metrics]

        # Log the results to the file
        if write_results_to_file:
            os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
            with open(os.path.join(output_dir, 'results', f'{name}.txt'), 'w') as f:
                f.write(','.join(keys) + '\n')
                for i in range(len(values)):
                    f.write(','.join([str(metrics[i][k]) for k in metrics[0].keys()]) + '\n')