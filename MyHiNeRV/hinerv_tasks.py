"""
视频回归任务（MyHiNeRV 版）：封装"模型前向 → 损失 → 指标 → 日志"流程。

相对 HiNeRV 的改动：
- parse_input 兼容附带 RAFT 光流伪标签的 batch（5 元组）；
- 训练期调用 model.forward_with_aux 获取辅助输出，组装新增损失：
    * FFT 频域损失（Scene Matters，零码率）；
    * 多尺度金字塔损失（FaNeRV/MSNeRV 频率尺度分解：每级小视频 vs max-pool GT）；
    * 动态高频注入（FaNeRV：训练后期 V_target = V + β·HP(V − sg(V̂)) 替换全分辨率参考）；
    * RAFT 伪标签光流约束 + 置信度熵正则（Scene Matters SFCM，零码率）；
    * 时序对比损失（Scene Matters TCL，作用在 CRSPE 每帧嵌入上，零码率）。
- 指标额外输出逐帧 MSE（'frame_mse'），供训练动态跟踪器（dynamics.py）记录；
- 所有损失对帧等权（不做帧级损失加权，帧权重仅通过采样器生效）。
"""
from utils import *
from losses import (compute_loss, compute_metric, compute_regularization,
                    fft_loss, maxpool_pyramid, highfreq_enhanced_target,
                    temporal_contrastive_loss, flow_constraint_loss, mse as mse_loss)


class VideoRegressionTask:
    """
    视频拟合任务。loss_cfg 仍为 [权重, '损失名', ...]（基础损失，默认 0.7·L1 + 0.3·MS-SSIM_5x5）；
    新增辅助损失权重由 args 的 w_* 参数控制（默认值见 set_task_args）。
    """
    def __init__(self, args, logger, accelerator, root, training=True, enable_log_eval=False):
        self.root = root
        self.accelerator = accelerator
        loss = list(args.loss) if isinstance(args.loss, (list, tuple)) else [args.loss]
        self.loss_cfg = [1.0, loss[0]] if len(loss) == 1 else loss
        self.metric_cfg = args.train_metric if training else args.eval_metric
        self.reg_cfg = args.reg
        self.training = training
        self.enable_log_eval = enable_log_eval

        # Aux loss configs（MyHiNeRV）
        self.w_fft = getattr(args, 'w_fft', 0.)                  # FFT 频域损失权重
        self.fft_hf_gamma = getattr(args, 'fft_hf_gamma', 0.)    # FFT 高频加权（0=纯 Scene Matters 版）
        self.w_scale = getattr(args, 'w_scale', 0.)              # 多尺度金字塔损失权重（每级）
        self.w_flow = getattr(args, 'w_flow', 0.)                # 光流约束权重
        self.w_flow_ent = getattr(args, 'w_flow_ent', 0.)        # 置信度熵正则权重
        self.w_tcl = getattr(args, 'w_tcl', 0.)                  # 时序对比损失权重
        self.tcl_sigma = getattr(args, 'tcl_sigma', 10.)         # TCL 时间高斯 σ
        self.tcl_tau = getattr(args, 'tcl_tau', 0.1)             # TCL 温度 τ
        self.hf_start = getattr(args, 'hf_start', 1.1)           # 高频注入起始进度（>1 = 关闭）
        self.hf_beta = getattr(args, 'hf_beta', 1.)              # 高频注入强度 β
        self.w_rate = getattr(args, 'w_rate', 0.)                # QEC 率失真损失权重（UAR-NVC QEC）
        self.rate_ramp = 1.0                                     # 率损失 ramp 因子（joint-rate 前 N ep 线性升到 1）
        self.aux_ramp_start = getattr(args, 'aux_ramp_start', 0.3)  # 辅助损失 ramp-in 起始进度
        self.aux_ramp_len = getattr(args, 'aux_ramp_len', 0.3)      # ramp-in 时长（进度比例）
        # 中段窗口调度（A4R）：p0..p1 余弦升、p1..p2 全量、p2..p3 余弦降、之后归零，
        # 兼防 epoch 0 干扰早期结构与晚期收敛。格式 "p0,p1,p2,p3"；空 = 用 ramp-in（旧行为）。
        self.aux_window = None
        aw = getattr(args, 'aux_window', '')
        if isinstance(aw, str) and aw:
            self.aux_window = tuple(float(x) for x in aw.split(','))
            assert len(self.aux_window) == 4 and self.aux_window[0] < self.aux_window[1] <= self.aux_window[2] < self.aux_window[3]
        self.hf_end = getattr(args, 'hf_end', 1.0)                # 高频注入结束进度（<1 = 中段窗口化）
        self.progress = 0.                                       # 当前训练进度 [0,1]（主循环更新）
        # patch 级损失权重（保守版 dynamicAL：降低 MSE 方差以缩小 Jensen 偏移）
        self.patch_loss_weights = None   # numpy [P] 或 None
        self.patch_num_patches = None    # (nT, nH, nW) 用于线性索引

        logger.info(f'VideoRegressionTask:')
        logger.info(f'     Root: {self.root}')
        logger.info(f'     Losses: {self.loss_cfg}    Metrics: {self.metric_cfg}    Regularization: {self.reg_cfg}')
        logger.info(f'     Aux: fft={self.w_fft}(gamma={self.fft_hf_gamma})  scale={self.w_scale}  '
                    f'flow={self.w_flow}(ent={self.w_flow_ent})  tcl={self.w_tcl}(sigma={self.tcl_sigma},tau={self.tcl_tau})')
        logger.info(f'     HF-inject: start={self.hf_start}  beta={self.hf_beta}')
        logger.info(f'     Training: {self.training}')
        logger.info(f'     Log evaulation: {self.enable_log_eval}')

    def set_progress(self, progress: float):
        """更新训练进度（0~1，epoch/num_epochs），决定是否启用动态高频注入。"""
        self.progress = progress

    def set_rate_ramp(self, factor: float):
        """更新率损失 ramp 因子（joint-rate 模式：前 rate_warmup_epochs 个 epoch 从 0 升到 1）。"""
        self.rate_ramp = float(factor)

    def set_dr_gate(self, batch_idx: int, k: int):
        """C4 D/R 交替门控（NVRC rate_steps=K 同式数学等效）：
        每 K 个 batch 才计率损失、乘 K 补偿——逐步联合 ≡ 逐步 D + 每 K 步 K·R。
        k=0 关闭（每步联合，C1-C3 行为）。"""
        self._dr_on = (k <= 0) or ((batch_idx + 1) % k == 0)
        self._dr_mult = k if k > 0 else 1

    def set_patch_loss_weights(self, weights, num_patches):
        """存储 patch 级损失权重（保守版，[0.85,1.15] 区间）与索引映射。"""
        self.patch_loss_weights = weights
        self.patch_num_patches = num_patches

    def _get_batch_loss_weights(self, inputs, device):
        """根据 batch 的 patch 坐标查损失权重，返回 [N] 张量或 None。"""
        if self.patch_loss_weights is None:
            return None
        nT, nH, nW = self.patch_num_patches
        idx = inputs['idx']  # [N, 3] = (t, h, w) patch 坐标
        lin = idx[:, 0] * (nH * nW) + idx[:, 1] * nW + idx[:, 2]
        w = torch.as_tensor(self.patch_loss_weights[lin.cpu().numpy()], dtype=torch.float32, device=device)
        return w

    def parse_input(self, loader, batch):
        """
        把 dataloader 的 batch 整理成 input/target。
        batch 可为 (idx, x) 或（附带光流伪标签时）(idx, x, flow_f, flow_b, valid)。
        """
        if len(batch) == 5:
            idx, x, flow_f, flow_b, flow_valid = batch
            input_flow = {'flow_f': flow_f, 'flow_b': flow_b, 'flow_valid': flow_valid}
        else:
            idx, x = batch[0], batch[1]
            input_flow = None

        assert idx.ndim == 2, 'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

        input = {
            'x': x if self.training else None,
            'idx': idx,
            'idx_max': loader.dataset.num_patches,
            'batch_size': x.shape[0] * loader.dataset.patch_size[0] / (loader.dataset.num_patches[1] * loader.dataset.num_patches[2]),  # in the number of images
            'video_size': loader.dataset.video_size,
            'patch_size': loader.dataset.patch_size,
            'flow': input_flow,
        }

        target = x

        return input, target

    def parse_output(self, loader, batch):
        """整理模型输出（保证连续内存布局）。"""
        output = batch
        output = output.contiguous(memory_format=torch.contiguous_format)
        return output

    def compute_loss(self, x, y, model=None):
        """按 loss_cfg 加权求和基础损失（对帧等权）。"""
        total_loss = None
        for i in range(len(self.loss_cfg) // 2):
            weight = float(self.loss_cfg[i * 2])
            loss_type = self.loss_cfg[i * 2 + 1]
            loss = weight * compute_loss(loss_type, x, y)
            total_loss = total_loss + loss if total_loss is not None else loss
        for i in range(len(self.reg_cfg) // 2):
            weight = float(self.reg_cfg[i * 2])
            reg_type = self.reg_cfg[i * 2 + 1]
            loss = weight * compute_regularization(reg_type, model)
            total_loss = total_loss + loss if total_loss is not None else loss
        return total_loss

    def compute_aux_losses(self, model, outputs, targets, aux, input_flow, video_size=None,
                           batch_loss_weights=None):
        """
        组装新增辅助损失（除 rate 外全部零码率；返回 dict，值为标量或 None）。
        - outputs/targets: [N, C, T, H, W]
        - aux: forward_with_aux 的辅助输出（None 表示模型未提供）
        - video_size: (T,H,W)，QEC 率估计的归一化像素数
        """
        losses = {}

        # 辅助损失调度：默认 ramp-in（只升不降）；--aux-window 给出时用中段窗口
        # （余弦升 → 全量保持 → 余弦降 → 0），四个机制的影响力集中在训练中段。
        if self.aux_window is not None:
            import math as _m
            p0, p1, p2, p3 = self.aux_window
            p = self.progress
            if p < p0 or p >= p3:
                ramp = 0.
            elif p < p1:
                ramp = 0.5 * (1. - _m.cos(_m.pi * (p - p0) / max(p1 - p0, 1e-6)))
            elif p <= p2:
                ramp = 1.
            else:
                ramp = 0.5 * (1. + _m.cos(_m.pi * (p - p2) / max(p3 - p2, 1e-6)))
        else:
            ramp = min(max((self.progress - self.aux_ramp_start) / max(self.aux_ramp_len, 1e-6), 0.), 1.)

        # 动态高频注入：全分辨率参考替换为增强目标（FaNeRV；--hf-end <1 时同样中段窗口化）
        if self.hf_start <= self.progress < self.hf_end:
            targets_eff = highfreq_enhanced_target(targets, outputs, beta=self.hf_beta)
        else:
            targets_eff = targets
        # 主损失（增强目标）：patch 级损失权重（保守版）在 batch 维微调梯度分配
        if batch_loss_weights is not None:
            per_frame = self.compute_loss(outputs, targets_eff)  # [N, T]
            losses['main'] = (per_frame * batch_loss_weights[:, None].expand_as(per_frame)).sum() /                             (batch_loss_weights.sum() * per_frame.shape[-1])
        else:
            losses['main'] = self.compute_loss(outputs, targets_eff).mean()  # 基础损失（增强目标）

        # FFT 频域损失（对原始 GT，非增强目标）
        if self.w_fft > 0. and ramp > 0.:
            losses['fft'] = ramp * self.w_fft * fft_loss(outputs, targets, high_freq_gamma=self.fft_hf_gamma).mean()

        # 多尺度金字塔损失（FaNeRV 尺度自适应：低分辨率 0.5·MSE + 0.5·L1；全分辨率由 main 负责）
        if aux is not None and self.w_scale > 0. and ramp > 0. and len(aux.get('scales', [])) > 0:
            scale_losses = []
            pyramid = maxpool_pyramid(targets, aux['scale_ratios'])
            for pred, gt_small in zip(aux['scales'], pyramid):
                l = 0.5 * compute_loss('mse', pred, gt_small) + 0.5 * compute_loss('l1', pred, gt_small)
                scale_losses.append(l.mean())
            losses['scale'] = ramp * self.w_scale * torch.stack(scale_losses).mean()

        # RAFT 伪标签光流约束 + 置信度熵正则（Scene Matters SFCM）
        if aux is not None and input_flow is not None and self.w_flow > 0. and ramp > 0. and aux.get('flow') is not None:
            pred_f, pred_b, conf = aux['flow']
            gt_f = input_flow['flow_f'].to(device=pred_f.device, dtype=pred_f.dtype)
            gt_b = input_flow['flow_b'].to(device=pred_b.device, dtype=pred_b.dtype)
            valid = input_flow['flow_valid'].to(pred_f.device).view(pred_f.shape[0], pred_f.shape[2])  # [N, T]
            l_flow, l_ent = flow_constraint_loss(pred_f.float(), pred_b.float(), conf.float(),
                                                 gt_f.float(), gt_b.float(), valid=valid)
            losses['flow'] = ramp * self.w_flow * l_flow
            losses['flow_ent'] = ramp * self.w_flow_ent * l_ent

        # 时序对比损失（Scene Matters TCL，作用于 CRSPE 每帧嵌入）
        if aux is not None and self.w_tcl > 0. and ramp > 0. and aux.get('frame_idx') is not None:
            m = unwrap_model(model)
            if getattr(m, 'crspe', None) is not None:
                ids = aux['frame_idx'][aux['frame_mask'].bool()]
                l_tcl = temporal_contrastive_loss(m.crspe.weight, ids, sigma=self.tcl_sigma, tau=self.tcl_tau)
                if l_tcl is not None:
                    losses['tcl'] = ramp * self.w_tcl * l_tcl

        # QEC 率失真损失（UAR-NVC QEC：量化整数的高斯熵估计，bpp 单位；QAT 启动后生效）。
        # joint-rate 模式下从主训练 epoch 0 生效，前 rate_warmup_epochs 个 epoch 乘 ramp 因子。
        if self.w_rate > 0. and video_size is not None:
            from compression.quant_utils import get_rate_bpp
            bpp_est = get_rate_bpp(model, float(np.prod(video_size)))
            if bpp_est is not None:
                gate = getattr(self, '_dr_on', True)
                mult = getattr(self, '_dr_mult', 1)
                if gate:
                    losses['rate'] = self.rate_ramp * self.w_rate * mult * bpp_est

        return losses

    def compute_metrics(self, x, y, model=None):
        """逐项计算指标；训练时额外输出逐帧 MSE（训练动态跟踪用）与
        QAT 期的可学习量化步长均值（观察率/失真对 q_step 的博弈）。"""
        metrics = {}
        for metric_type in self.metric_cfg:
            metrics[metric_type] = compute_metric(metric_type, x, y)
        if self.training:
            metrics['frame_mse'] = mse_loss(x, y)  # [N, T]
            if model is not None:
                from compression.quant_utils import get_mean_qstep
                qs = get_mean_qstep(model)
                if qs is not None:
                    metrics['qstep'] = torch.tensor(qs, device=x.device)
        return metrics

    def step(self, model, loader, batch):
        """
        单步执行：解析输入 → 前向（训练期取辅助输出）→ 损失 → 指标。
        返回 (inputs, targets, outputs, loss, metrics, loss_dict)。
        """
        inputs, targets = self.parse_input(loader, batch)

        if self.training and hasattr(unwrap_model(model), 'forward_with_aux'):
            output, aux = unwrap_model(model).forward_with_aux(inputs)
            outputs = self.parse_output(loader, output)
            batch_w = self._get_batch_loss_weights(inputs, outputs.device) if self.training else None
            losses = self.compute_aux_losses(model, outputs, targets, aux, inputs['flow'],
                                             video_size=inputs['video_size'],
                                             batch_loss_weights=batch_w)
            loss = sum(v for v in losses.values() if v is not None)
        else:
            outputs = self.parse_output(loader, model(inputs))
            loss = self.compute_loss(outputs, targets)
            losses = {'main': loss}

        metrics = self.compute_metrics(outputs, targets, model=model)
        return inputs, targets, outputs, loss, metrics, losses

    def log_eval(self, dir_name, inputs, outputs, metrics):
        """
        把重建帧保存为 PNG（文件名含帧号与指标值），便于目视检查。
        """
        if not self.enable_log_eval:
            return

        metric = metrics[self.metric_cfg[0]]

        N, C, T, H, W = outputs.shape
        _, H_img, W_img = inputs['video_size']
        T_patch, H_patch, W_patch = inputs['patch_size']

        assert H_img == H_patch and W_img == W_patch, 'Only full image output is supported'

        idx = inputs['idx'].cpu()
        outputs = outputs.detach().cpu()
        metric = metric.detach().cpu()

        output_dir = os.path.join(self.root, dir_name)

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()

        for n in range(N):

            t, h, w = idx[n].numpy()
            patch_idx = 1 + t * T_patch

            for dt in range(T):
                img_id = f'{patch_idx + dt:04d}_{metric[n, dt].numpy():.2f}'
                img = outputs[n, :, dt, :, :].float().cpu()
                torchvision.utils.save_image(img, os.path.join(output_dir, img_id + '.png'), 'png')


def set_task_args(parser):
    """注册任务相关参数：基础损失/正则/指标 + 新增辅助损失权重。"""
    group = parser.add_argument_group('Task parameters')
    group.add_argument('--loss', default=['mse'], type=str, nargs='+', help='Loss (default: "mse")')
    group.add_argument('--reg', default=[], type=str, nargs='+', help='Regularization')
    group.add_argument('--train-metric', default=['psnr'], type=str, nargs='+', help='Metric (default: "psnr")')
    group.add_argument('--eval-metric', default=['psnr', 'ms_ssim'], type=str, nargs='+', help='Metric (default: "psnr")')
    group.add_argument('--log-eval', type=str_to_bool, default=True, help='Log the output during evaluation.')

    group = parser.add_argument_group('MyHiNeRV aux loss parameters')
    group.add_argument('--w-fft', type=float, default=0.1, help='weight of the FFT frequency loss (Scene Matters). 0: disabled')
    group.add_argument('--fft-hf-gamma', type=float, default=0., help='high-frequency emphasis gamma of the FFT loss (0: plain L1 on magnitude).')
    group.add_argument('--w-scale', type=float, default=0.2, help='weight of the multi-scale pyramid loss (FaNeRV/MSNeRV). 0: disabled')
    group.add_argument('--w-flow', type=float, default=0.01, help='weight of the RAFT pseudo-label flow constraint (Scene Matters SFCM). 0: disabled')
    group.add_argument('--w-flow-ent', type=float, default=0.001, help='weight of the confidence entropy regularization of SFCM.')
    group.add_argument('--w-tcl', type=float, default=0.01, help='weight of the temporal contrastive loss on CRSPE embeddings. 0: disabled')
    group.add_argument('--tcl-sigma', type=float, default=10., help='sigma of the temporal Gaussian soft labels of TCL.')
    group.add_argument('--tcl-tau', type=float, default=0.1, help='temperature of TCL.')
    group.add_argument('--hf-start', type=float, default=0.5, help='training progress to start dynamic high-frequency injection (FaNeRV). >1: disabled')
    group.add_argument('--aux-ramp-start', type=float, default=0.3, help='progress at which aux losses start ramping in (early epochs: main loss only).')
    group.add_argument('--aux-ramp-len', type=float, default=0.3, help='ramp-in duration (in progress fraction) for aux losses.')
    group.add_argument('--aux-window', type=str, default='',
                       help='mid-training window schedule for aux losses "p0,p1,p2,p3" (cosine up / hold / cosine down / zero). Overrides ramp-in. Empty: use ramp-in.')
    group.add_argument('--hf-end', type=float, default=1.0, help='progress to stop dynamic high-frequency injection (mid-windowed when <1).')
    group.add_argument('--hf-beta', type=float, default=1.0, help='strength of dynamic high-frequency injection.')
    group.add_argument('--w-rate', type=float, default=1.0, help='weight of the QEC rate (bpp) loss (UAR-NVC QEC; active when --quant-mode qec). 0: disabled')
    group.add_argument('--joint-rate', type=str_to_bool, default=False,
                       help='joint rate-distortion training (UAR-NVC style): activate quantized forward from main epoch 0 (requires --quant-mode qec/hyper)')
    group.add_argument('--rate-warmup-epochs', type=int, default=15,
                       help='linear ramp-up epochs of the rate loss at the start of joint-rate main training')
    group.add_argument('--quant-soft-anneal', default='', type=str,
                       help='C4 NVRC-style soft-round anneal "t0,t1" (e.g. "0.5,0.3"); empty = hard STE')
    group.add_argument('--dr-alternate', default=0, type=int,
                       help='C4 D/R alternation period K (NVRC rate_steps): rate loss on every K-th batch x K; 0 = joint every batch')
    group.add_argument('--col-prune-target', default=0., type=float,
                       help='D1 column-wise structured sparsity target (0 = off; e.g. 0.30 = 30%% of output channels zeroed)')
    group.add_argument('--col-prune-start', default=60, type=int, help='D1 start epoch for progressive column pruning')
    group.add_argument('--col-prune-end', default=300, type=int, help='D1 end epoch (reach target sparsity)')
    group.add_argument('--col-prune-every', default=20, type=int, help='D1 prune every N epochs')
    group.add_argument('--sampling-mode', default='flat', type=str, choices=['flat', 'hierarchical'],
                       help='B5 sampling strategy: flat (WeightedRandomSampler) or hierarchical (TreeNeRV frame + LossGrad patch)')
    group.add_argument('--tree-ratio', default=0.5, type=float,
                       help='B5 mix ratio: fraction of TreeNeRV-driven samples (0=pure LossGrad, 1=pure TreeNeRV)')
