from utils import *
from io_utils import *
from losses import *


class OverfitTask:
    """
    过拟合编码任务（OverfitTask）：定义单个视频过拟合训练/评估时的损失、指标与优化目标。

    主要职责：
        - parse_batch(): 把 DataLoader 给出的 (patch 坐标, patch 像素) 解析为模型输入字典 inputs；
        - d_step(): 失真步——只前向重建、计算 D = lambda * L(x_hat, x)；
        - r_step(): 率步——只前向码率、计算 R（所有参数熵编码比特数，按视频像素数归一化）；
        - compute_d_loss()/compute_r_loss()/compute_metrics(): 分别组合损失、直通率、指标；
        - log_outputs()/flush(): 把重建 patch 写回输出视频并保存逐 patch 指标。
    """
    def __init__(self, logger, video, loss_cfg, metric_cfg, lamb,
                 channel_scale=None, channel_shift=None,
                 enable_log=True, training=True, device=None):
        self.logger = logger
        self.video = video
        self.channels = self.video.get_num_channels()

        # 损失配置：[权重1, 损失名1, 权重2, 损失名2, ...]，例如 [0.7, 'l1', 0.3, 'ms-ssim']
        self.loss_cfg = [1.0, loss_cfg[0]] if len(loss_cfg) == 1 else loss_cfg
        self.metric_cfg = metric_cfg

        assert channel_scale is None or channel_shift is None or \
            (len(channel_scale) == self.channels and len(channel_shift) == self.channels)
        self.channel_scale, self.channel_shift = \
            compute_scale_shift(self.channels, self.video.get_bit_depth(), channel_scale, channel_shift)

        self.enable_log = enable_log
        self.training = training
        self.device = device
        self.metrics_buffer = {}

        # lambda：率失真权衡系数 D + lambda * R（此处仅单个码率点）
        assert isinstance(lamb, (list, tuple)) and len(lamb) == 1, 'lamb should be a list/tuple with a single value'
        self.lamb = torch.tensor(sorted(lamb), dtype=torch.float32, device=self.device)

        logger.info(f'OverfitTask:')
        logger.info(f'     Root: {self.video.get_path()}')
        logger.info(f'     Training: {self.training}')
        logger.info(f'     Losses: {self.loss_cfg}    Metrics: {self.metric_cfg}')
        logger.info(f'     Lamb: {self.lamb.tolist()}')
        logger.info(f'     Enable log: {self.enable_log}')

    def get_metrics(self):
        return self.metric_cfg

    def get_video_size(self):
        return tuple(v + sum(p) for v, p in zip(self.video.get_video_size(), self.video.get_padding()))

    def get_patch_size(self):
        return self.video.get_patch_size()

    def get_start_frame(self):
        return self.video.get_start_frame()

    def get_num_frames(self):
        return self.video.get_num_frames()

    def set_frames(self, start_frame, num_frames):
        self.video.set_frames(start_frame, num_frames)

    def create_cache(self):
        self.video.create_cache(enable=self.enable_log)

    def parse_batch(self, batch):
        """
        Parse the input and output batch during training/evaluation step
        （把一个 batch 解析为模型输入字典 inputs 与目标像素 x：
          idx 为 3D patch 坐标，rel_batch_size 为本 batch 像素占整段视频的比例，用于码率归一化）
        """
        idx, x = batch
        idx_max = self.video.get_idx_max()
        assert idx.ndim == 2, \
            'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
        assert x.ndim == 5,  \
            'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

        inputs = {
            'vidx': torch.zeros(idx.shape[0], dtype=torch.int32, device=idx.device),
            'vidx_max': 1,
            'idx': idx,                    # patch 坐标 (t, h, w)
            'idx_max': idx_max,            # 各维度 patch 数量上限
            'x': x if self.training else None,  # 训练时传入目标像素（供 HiNeRV 采样 batch 大小）
            'lamb': self.lamb,             # 率失真权衡系数
            'rp': None,
            'rel_batch_size': x.shape[0] * x.shape[2] * \
                math.prod(self.get_patch_size()[1:]) / math.prod(self.get_video_size()[1:]),
            'video_size': (self.get_num_frames(),) + self.get_video_size()[1:],  # 视频尺寸 (T, H, W)
            'patch_size': self.get_patch_size(),                                 # patch 尺寸
            'channels': self.video.get_num_channels()                            # 通道数 (RGB=3)
        }

        return inputs, x

    def parse_output(self, output):
        """
        Parse the output from the model during training/evaluation step
        （整理模型输出的形状/内存布局）
        """
        return output.contiguous()

    def compute_d_loss(self, x, y, lamb, w=None):
        """失真损失 D = sum_i weight_i * lambda * Loss_i(x_hat, x)，按帧均值。
        w: [N] 逐样本保守损失权重（动态加权系统，None = 均值）。"""
        loss = 0.
        for i in range(len(self.loss_cfg) // 2):
            weight = float(self.loss_cfg[i * 2])
            loss_type = self.loss_cfg[i * 2 + 1]
            loss_nt = compute_loss(loss_type, x, y)          # [N, T]
            if w is not None:
                # 加权平均（等价于 MyHiNeRV 的保守损失权重：降低 batch 内损失方差）
                loss += weight * lamb * (w.unsqueeze(1) * loss_nt).sum() / w.sum()
                # 记录逐样本均值（供 PatchDynamicsTracker 更新，μ+λ·σ 得分）
                if not hasattr(self, '_sample_losses') or self._sample_losses is None:
                    self._sample_losses = loss_nt.detach().mean(dim=1)
                else:
                    self._sample_losses = loss_nt.detach().mean(dim=1)
            else:
                loss += weight * lamb * loss_nt.mean()
                self._sample_losses = loss_nt.detach().mean(dim=1)
        return loss

    # ---------- 动态加权系统（MyHiNeRV B 系列迁移） ----------
    def set_patch_weights(self, loss_weights, num_patches):
        """存储逐 patch 保守损失权重与 patch 总数（None = 关闭）。"""
        self.patch_loss_weights = loss_weights
        self.patch_num_patches = num_patches

    def _lookup_batch_weights(self, idx):
        """按 batch 的 (t,h,w) 坐标查保守损失权重（无权重或长度不符时返回 None）。"""
        w_map = getattr(self, 'patch_loss_weights', None)
        if w_map is None:
            return None
        import numpy as _np
        idx_max = self.video.get_idx_max()
        flat = idx[:, 0].long() * (idx_max[1] * idx_max[2]) + idx[:, 1].long() * idx_max[2] + idx[:, 2].long()
        w = w_map.to(idx.device)[flat]
        return w

    def compute_r_loss(self, r):
        """率损失直通返回（r_step 中已按视频像素数归一化）"""
        return r

    def compute_metrics(self, x, y):
        """计算各项评估指标（PSNR/MS-SSIM 等）"""
        metrics = {}
        with torch.no_grad():
            for metric_type in self.metric_cfg:
                metrics[metric_type] = compute_metric(metric_type, x, y)
        return metrics

    def d_step(self, model, batch):
        """失真步：模型只重建输出（compute_outputs=True），不计算码率，返回 D-loss"""
        inputs, target = self.parse_batch(batch)
        output = model(inputs, compute_outputs=True, compute_rates=False)
        output = self.parse_output(output)
        w = self._lookup_batch_weights(inputs['idx'])
        loss = self.compute_d_loss(output, target, inputs['lamb'], w=w)
        metrics = self.compute_metrics(output, target)
        return inputs, target, output, loss, metrics

    def r_step(self, model, batch, sub_step=0, num_sub_steps=1):
        """率步：模型只计算码率（compute_rates=True），不做重建；返回 (bpp, R-loss)"""
        inputs, _ = self.parse_batch(batch)
        inputs['r_sub_step'] = sub_step
        inputs['r_num_sub_steps'] = num_sub_steps
        rate, rate_loss = model(inputs, compute_outputs=False, compute_rates=True)
        rate = rate / math.prod(inputs['video_size'])       # bits -> bpp（按视频总像素归一化）
        rate_loss = self.compute_r_loss(rate_loss / math.prod(inputs['video_size']))
        return rate, rate_loss

    def log_outputs(self, inputs, outputs, metrics):
        """
        Log the outputs
        （把重建 patch 反归一化后写回输出视频缓存，并记录每个 patch 的平均指标）
        """
        if self.enable_log:
            N, _, _, _, _ = outputs.shape
            idx = inputs['idx']
            outputs = outputs.detach()
            metrics = {k: v.detach() for k, v in metrics.items()}

            scale = self.channel_scale.to(outputs.device)
            shift = self.channel_shift.to(outputs.device)

            # Loop over all samples
            for n in range(N):
                # Save the patch
                if isinstance(self.video, PNGVideo):
                    self.video.write_patch(idx[n].cpu().numpy(),
                                           ((outputs[n] - shift.view(self.channels, 1, 1, 1)) \
                                            / scale.view(self.channels, 1, 1, 1)) \
                                           .permute(1, 2, 3, 0).round().cpu().numpy())
                else:
                    yuv420_patch = yuv444_to_yuv420(outputs[n].permute(1, 0, 2, 3), mode='avg_pool')
                    self.video.write_patch(idx[n].cpu().numpy(),
                                           [((patch_i - shift[i]) / scale[i]).round().cpu().numpy() \
                                            for i, patch_i in enumerate(yuv420_patch)])

                # Save the averge metrics for each patch
                idx_str = '(' + ' '.join([str(i) for i in idx[n].tolist()]) + ')'
                self.metrics_buffer[idx_str] = \
                    ','.join([idx_str] + \
                             [f'{k}: {v[n].mean().item():.4f}' for k, v in metrics.items() if k in self.metric_cfg])

    def flush(self, dir):
        """
        Flush the outputs to the disk
        （把缓存中的重建视频与逐 patch 指标写入磁盘）
        """
        if self.enable_log:
            # Save video outputs
            self.video.flush(dir)

            # Save metrics
            with open(os.path.join(dir, 'metrics.txt'), 'w') as f:
                f.write('\n'.join(self.metrics_buffer.values()))
            self.metrics_buffer = {}


def create_overfit_task(args, logger, video, channel_scale=None, channel_shift=None, training=True, device=None):
    # Create task（根据训练/评估配置创建 OverfitTask）
    if training:
        config = args.train_task
    else:
        config = args.eval_task

    task = OverfitTask(logger, video, loss_cfg=config.loss, metric_cfg=config.metric,
                       lamb=config.lamb, channel_scale=channel_scale, channel_shift=channel_shift,
                       enable_log=config.enable_log, training=training, device=device)
    return task