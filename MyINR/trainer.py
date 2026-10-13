# ==========================================================================================
# ClipTrainer (v2): 训练单个(clip 级)INR 模型。
#
# v2 相对 v1 的改进(P0-P6, 见 REPORT.md):
#   P2  II 插值初始化: P-model 参数 θ' = ε·θ_本模型随机初始化 + (1−ε)·θ_前段反量化权重,
#       ε 由相邻 GOP 帧间 MSE 自适应(UAR-NVC II 模块), 避免 D-init 局部极小
#   P1  RQEC-lite 率感知微调: P-model 训练损失中加入逐张量熵率估计项
#       rate = Σ_t numel_t·log2(2e·b_t)/Σnumel, b_t = mean|θ−θ'| (Laplace 熵),
#       训练动态权衡"偏离初始化的码率代价"与重建质量
#   P3  多尺度损失加权: 中间尺度权重 ms_lw(<1) 且在训练后 40% 线性退火到 0,
#       最终尺度恒为 1(替代 v1 的均匀 1/N 平均)
#   P4  辅助损失退火: FFT/TCL 在最后 30% epoch 线性退火到 0; HFB 在 80% 后关闭;
#       TCL 支持静态场景门控(tcl_enabled=False 直接关闭)
#   P5  权重差"先阈值后恢复": 训练 85% 处按幅值阈值化差分(保留 delta_keep),
#       之后在掩码约束下继续训练 15% 恢复精度 —— 消除 v1 "训完硬删 75% 差分"的损伤
#   v2 移除显式光流模块(伪标签/光流头/置信度头), 四个辅助损失剩 FFT/TCL(+CRSPE/HFB)
# mode='baseline' 时完全退化为原版 NeRV 协议(单帧 batch=1 / single-res / Fusion6 / avg-pool)。
# ==========================================================================================
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import (EmbeddingBank, fft_loss, high_freq_boosted_target,
                    maxpool_down, scale_adaptive_loss, spatial_loss, tcl_loss)
from modules import NeRVGenerator
from modules import STEQuant, register_qat, set_qat_start, remove_qat
from video_data import VideoData, evaluate_psnr


class Cfg:
    """一次运行的全部配置(带默认值, 由命令行覆盖)。"""

    def __init__(self, **kw):
        # 协议
        self.mode = 'ours'                # 'baseline' | 'ours'
        self.epochs = 150
        self.warmup = 30
        self.lr = 5e-4
        self.seed = 1
        # 结构
        self.stem_dim_num = '512_1'
        self.fc_hw_dim = '9_16_26'
        self.lower_width = 96
        self.expansion = 1
        self.reduction = 2
        self.strides = [5, 3, 2, 2, 2]
        self.act = 'swish'
        self.embed = '1.25_40'
        # 模块开关 (v2: 无光流)
        self.use_crspe = True
        self.crspe_channels = 4
        self.use_ms = True                # 多尺度 max-pool 金字塔监督
        self.ms_lw = 0.3                  # P3: 中间尺度权重(v1 为 1/N 均匀)
        self.ms_anneal_frac = 0.6         # P3: 60% epoch 后中间尺度退火到 0
        self.use_hfb = True               # 高频增强目标
        self.hfb_beta = 1.0
        self.hfb_off_frac = 0.8           # P4: 80% epoch 后关闭 HFB
        self.use_fft = True
        self.fft_weight = 0.1
        self.use_tcl = True
        self.tcl_weight = 0.005
        self.tcl_start_frac = 0.2
        self.tcl_enabled = None           # P4: None=由调用方门控; False=强制关闭(静态场景)
        self.use_fw = True                # dynamicAL 帧权重
        self.fw_mode = 'resource'         # 'loss'=v1 损失调制(消融 A6 语义);
        #                                   'resource'=资源分配(修正版): 采样概率∝当前残差,
        #                                   损失保持纯度量不变, 末段退火回均匀
        self.fw_alpha = 1.0               # 采样概率 ∝ priority^alpha
        self.fw_floor = 0.3               # 采样概率下限(×均匀), 防止任何帧被饿死
        self.fw_ema = 0.9                 # 逐帧残差 EMA 系数
        self.fw_anneal_frac = 0.25        # 末段退火回均匀采样的 epoch 比例
        # P-model 相关
        self.delta_l1 = 0.3               # v1 兼容: 朴素差分 L1(rate_weight=0 时生效)
        self.rate_weight = 0.005          # V3: R+λD 的 λ(UAR-NVC 式率失真联合)
        self.use_rate_dense = False        # V3: 底座自身权重也加熵率项(R+λD), 允许底座更粗糙换码率
        self.delta_keep = 0.25            # P5: 差分阈值保留比例
        self.delta_thr_frac = 0.85        # P5: 阈值化时机(85% epoch), 其后恢复训练
        self.ii_eps = None                # P2: II 初始化混合系数(None=不启用, 由调用方计算)
        # 压缩
        self.prune = True
        self.prune_keep = 0.4
        self.prune_fracs = [0.35, 0.45, 0.55, 0.65]
        self.quant_bit = 8
        # 训练细节
        self.pair_batch = True            # ours: 相邻帧对 (t,t+1); baseline: 单帧
        self.residual_mode = False         # True=分支输出原始残差(无 sigmoid/tanh, 可为负)
        self.eval_freq = 25
        self.log_freq = 200
        self.aux_off_frac = 0.7           # P4: 最后 30% epoch 辅助损失(FFT/TCL)退火到 0
        self.rate_warm_frac = 0.5          # P3 补充: 率项 warmup(前 50% epoch 纯失真, 之后 ramp)
        for k, v in kw.items():
            setattr(self, k, v)


def build_pe(cfg):
    lbase, levels = [float(x) for x in cfg.embed.split('_')]
    return lbase, int(levels)


def pe_encode(norm_idx, lbase, levels):
    """帧索引位置编码 → (B, 2L)"""
    out = []
    for i in range(levels):
        v = norm_idx * lbase ** i * math.pi
        out += [torch.sin(v), torch.cos(v)]
    return torch.stack(out, 1)


class ClipTrainer:
    """训练视频某一段(或整段)的 INR 模型。"""

    def __init__(self, video: VideoData, cfg: Cfg, frame_range, device='cuda'):
        self.video = video
        self.cfg = cfg
        self.device = device
        self.lo, self.hi = frame_range
        self.n = self.hi - self.lo
        self.lbase, self.levels = build_pe(cfg)
        self.is_baseline = (cfg.mode == 'baseline')
        torch.manual_seed(cfg.seed + self.lo)
        np.random.seed(cfg.seed + self.lo)

        self.model = NeRVGenerator(
            embed_length=2 * self.levels, stem_dim_num=cfg.stem_dim_num, fc_hw_dim=cfg.fc_hw_dim,
            expansion=cfg.expansion, reduction=cfg.reduction, lower_width=cfg.lower_width,
            stride_list=cfg.strides, act=cfg.act, sigmoid=False,
            n_frames=self.n,
            crspe_channels=(0 if self.is_baseline or not cfg.use_crspe else cfg.crspe_channels),
            multi_scale=(cfg.use_ms and not self.is_baseline),
            residual_mode=cfg.residual_mode).to(device)
        # QAT: 仅在率约束启用时注册(STE 参数化会改变 state_dict 键名, 影响初始化加载)
        # 分支(rate_weight=0, 纯失真)不应注册 → 保持标准键名, 投影初始化才能正确加载
        if cfg.use_rate_dense or (hasattr(cfg, 'rate_weight') and cfg.rate_weight > 0):
            self._n_qat = register_qat(self.model)
        else:
            self._n_qat = 0
        self._qat_active = False
        self._qs_params = [p for name, p in self.model.named_parameters() if 'q_step_raw' in name]
        # 优化器: 模型参数(不含 q_step) + q_step 高学习率组(需快速收敛)
        _main_params = [p for name, p in self.model.named_parameters() if 'q_step_raw' not in name]
        self.opt = torch.optim.Adam([
            {'params': _main_params, 'lr': cfg.lr},
            {'params': self._qs_params, 'lr': cfg.lr},   # q_step 用同 LR(softplus 参数化后不需要高 LR)
        ], betas=(0.5, 0.999))
        self.prune_mask = None
        self.delta_mask = None
        self.embedder = None
        self.bank = None

    # ---------------------------------------------------------- 工具
    def _embedder(self):
        if self.embedder is None:
            from embedder import AlexnetEmbedder
            self.embedder = AlexnetEmbedder(self.device)
        return self.embedder

    def _norm_idx(self, local_ids):
        return torch.tensor([i / self.n for i in local_ids], dtype=torch.float32, device=self.device)

    def _targets(self, gt, out_list):
        """多尺度目标: ours 用 max-pool 金字塔; baseline 用 adaptive_avg_pool(单尺度)。"""
        if self.is_baseline:
            return [F.adaptive_avg_pool2d(gt, o.shape[-2:]) for o in out_list]
        return [maxpool_down(gt, o.shape[-2:]) for o in out_list]

    def _spatial(self, out_list, target_list, ep_frac):
        """逐尺度损失。P3: 中间尺度权重 ms_lw×退火, 最终尺度恒 1; 最终尺度 HFB(P4: 后段关闭)。"""
        cfg = self.cfg
        n = len(out_list)
        if self.is_baseline:
            return spatial_loss(out_list[-1], target_list[-1], 'Fusion6')
        anneal = 1.0
        if ep_frac > cfg.ms_anneal_frac:
            anneal = max(0.0, 1.0 - (ep_frac - cfg.ms_anneal_frac) / max(1e-6, 1.0 - cfg.ms_anneal_frac))
        loss = 0.0
        for i, (o, t) in enumerate(zip(out_list, target_list)):
            if i < n - 1:
                loss = loss + cfg.ms_lw * anneal * scale_adaptive_loss(o, t)
            else:
                tt = t
                if cfg.use_hfb and ep_frac < cfg.hfb_off_frac:
                    tt = high_freq_boosted_target(t, o, beta=cfg.hfb_beta)
                loss = loss + scale_adaptive_loss(o, tt)
        return loss

    def _rate_term(self, ref_sd):
        """UAR-NVC 式 R+λD(与 QAT STE 配合):
        率项推 q_step 向上(更粗=更省), QAT 前向的失真推 q_step 向下(更细=更准) → 真实平衡。"""
        total, cnt = 0.0, 0
        for name, module in self.model.named_modules():
            if not hasattr(module, 'parametrizations') or 'weight' not in getattr(module, 'parametrizations', {}):
                continue
            ste = None
            for p in module.parametrizations.weight:
                if isinstance(p, STEQuant):
                    ste = p; break
            if ste is None:
                continue
            w = module.parametrizations.weight.original
            kk = name + '.weight'
            if 'frame_embed' in kk or w.dim() < 2:
                continue
            if ref_sd is not None:
                if kk not in ref_sd or ref_sd[kk].shape != w.shape:
                    continue
                x = w - ref_sd[kk]
            else:
                x = w
            sigma_x = x.pow(2).mean().sqrt().clamp(1e-8)
            q_step = torch.clamp(ste.q_step, 1e-4, 1.0)
            H = torch.log2(sigma_x * math.sqrt(2 * math.pi * math.e)) - torch.log2(q_step)
            H = torch.clamp(H, min=0.01, max=16.0)
            total = total + w.numel() * H
            cnt += w.numel()
        if cnt == 0:
            return torch.zeros((), device=self.device)
        return total / max(cnt, 1)
    def get_q_steps(self):
        """返回训练好的量化步长 dict(供压缩时使用)。"""
        if hasattr(self, '_q_steps_trained') and self._q_steps_trained:
            return self._q_steps_trained
        out = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'parametrizations') and 'weight' in getattr(module, 'parametrizations', {}):
                for p in module.parametrizations.weight:
                    if isinstance(p, STEQuant):
                        out[(name + '.weight').replace('.', '_')] = p.q_step.detach().cpu()  # property 返回 softplus 值
        return out

    # ---------------------------------------------------------- 训练
    def train(self, frame_weights=None, init_state=None, ref_state=None,
              epochs=None, lr=None, verbose=True, inherit_mask=None, res_fn=None,
              ckpt_path=None, ckpt_every=20):
        """frame_weights: (n,) 帧权重(None=均匀); init_state: P-model 参照(反量化)权重;
        ref_state: 率项/差分 L1 参照(应与 init 相同); inherit_mask: 继承 I-model 剪枝掩码;
        res_fn: V3 残差分支 —— res_fn(global_ids)->(B,3,H,W) 冻结尾座输出, 分支学习其残差。
        ckpt_path: 每 ckpt_every 个 epoch 落盘一次断点(进程被杀后可从断点续训)。"""
        cfg = self.cfg
        epochs = epochs or cfg.epochs
        lr0 = lr or cfg.lr
        warmup = max(1, int(cfg.warmup * epochs / cfg.epochs)) if epochs != cfg.epochs else cfg.warmup
        warmup = min(warmup, epochs // 5)
        fw = torch.tensor(frame_weights if frame_weights is not None else np.ones(self.n),
                          dtype=torch.float32, device=self.device)
        # fw resource 模式: 逐帧优先级 = 当前残差的 EMA, 只影响采样, 不改损失。
        # 种子化: 提供 frame_weights(dynamicAL 探针训练动态)时以其初始化, 在线 EMA 随训练接管
        prio = fw.clone() if frame_weights is not None else \
            torch.ones(self.n, dtype=torch.float32, device=self.device)
        ref_sd = None
        if ref_state is not None:
            ref_sd = {k: v.to(self.device) for k, v in ref_state.items()}

        # ---- 断点续训: 恢复模型/优化器/掩码/采样优先级, 跳过 init ----
        start_ep = 0
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                ck = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(ck['model'], strict=False)
                try:
                    self.opt.load_state_dict(ck['opt'])
                except Exception:
                    pass
                prio = ck['prio'].to(self.device).float()
                self.prune_mask = {k: (m.to(self.device) if m is not None else None)
                                   for k, m in (ck.get('prune_mask') or {}).items()}
                self.delta_mask = {k: (m.to(self.device) if m is not None else None)
                                   for k, m in (ck.get('delta_mask') or {}).items()}
                if ck.get('qat_active') and self._n_qat > 0 and not self._qat_active:
                    set_qat_start(self.model, True)
                    self._qat_active = True
                start_ep = int(ck['ep'])
                if verbose:
                    print(f'  [clip {self.lo}-{self.hi}] 从断点续训: ep{start_ep}/{epochs}', flush=True)
            except Exception as e:
                print(f'  [clip {self.lo}-{self.hi}] 断点损坏({e}), 从头训练', flush=True)
                start_ep = 0

        # ---- 初始化: P2 II 插值 θ' = ε·θ_fresh + (1−ε)·θ_prev ----
        if start_ep == 0 and init_state is not None:
            own = self.model.state_dict()
            if cfg.ii_eps is not None and cfg.ii_eps > 0:
                eps = float(cfg.ii_eps)
                mixed = {}
                for k, v in init_state.items():
                    if k in own and own[k].shape == v.shape:
                        v = v.to(own[k].device)
                        fresh = own[k].detach().clone()
                        if 'frame_embed' in k:
                            mixed[k] = v      # 嵌入表逐帧新内容, 不混合
                        else:
                            mixed[k] = (eps * fresh + (1 - eps) * v.float()).to(v.dtype)
                self.model.load_state_dict(mixed, strict=False)
            else:
                init = {k: v for k, v in init_state.items() if k in own and own[k].shape == v.shape}
                fe_key = 'frame_embed'
                if fe_key in init_state and fe_key in own and init_state[fe_key].shape != own[fe_key].shape:
                    last_row = init_state[fe_key][-1:].clone()
                    init[fe_key] = last_row.repeat(own[fe_key].shape[0], 1, 1, 1)
                self.model.load_state_dict(init, strict=False)

        prune_steps = {}
        if cfg.prune and cfg.prune_keep < 1.0:
            # 渐进剪枝: 每一步保留"当前剩余"的 keep^(1/n), n 步连乘后总保留恰为 keep
            keep_step = cfg.prune_keep ** (1.0 / len(cfg.prune_fracs))
            for f in cfg.prune_fracs:
                prune_steps[int(f * epochs)] = keep_step
        if start_ep == 0 and inherit_mask is not None:
            self.prune_mask = {k: (m.to(self.device) if m is not None else None)
                               for k, m in inherit_mask.items()}
            self._apply_mask()
        # P5: 差分"先阈值后恢复"的时机
        delta_thr_ep = int(cfg.delta_thr_frac * epochs) if ref_sd is not None else None
        tcl_on = cfg.use_tcl and (cfg.tcl_enabled is None or cfg.tcl_enabled)

        rng = np.random.RandomState(cfg.seed + self.lo)
        model = self.model
        model.train()
        t0 = time.time()
        for ep in range(start_ep, epochs):
            ep_frac = (ep + 1) / epochs
            # 学习率: warmup + cosine
            if ep < warmup:
                mult = 0.1 + 0.9 * ep / warmup
            else:
                mult = 0.5 * (math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup)) + 1)
            for g in self.opt.param_groups:
                g['lr'] = lr0 * mult
            # QAT: warmup 结束后激活 STE 量化(模型前向开始使用量化后权重)
            if self._n_qat > 0 and ep == warmup and not self._qat_active:
                set_qat_start(self.model, True)
                self._qat_active = True
            # 渐进剪枝
            if ep in prune_steps:
                self.prune_mask = self._global_prune(prune_steps[ep])
                self._apply_mask()
                if verbose:
                    print(f'  [clip {self.lo}-{self.hi}] prune step @ep{ep}: keep={prune_steps[ep]:.3f}', flush=True)
            # P5: 到达阈值化时机 → 对差分(相对 ref)做幅值阈值, 生成并施加差分掩码
            if delta_thr_ep is not None and ep == delta_thr_ep:
                with torch.no_grad():
                    sd = model.state_dict()
                    # STE QAT 参数化给键加 .original 后缀 → 清洗后匹配 ref_sd
                    delta_sd = {}
                    for k in sd:
                        ck = k.replace('.original', '')
                        if (ck in ref_sd and 'frame_embed' not in ck
                                and sd[k].dim() >= 2 and ref_sd[ck].shape == sd[k].shape):
                            delta_sd[ck] = sd[k].float() - ref_sd[ck].float()
                    if delta_sd:
                        self.delta_mask = self._thr_mask(delta_sd, cfg.delta_keep)
                        self._apply_delta_mask(ref_sd)
                        if verbose:
                            kept = sum(m.sum().item() for m in self.delta_mask.values() if m is not None)
                            tot = sum(m.numel() for m in self.delta_mask.values() if m is not None)
                            print(f'  [clip {self.lo}-{self.hi}] delta threshold @ep{ep}: '
                                  f'keep {kept/tot:.3f} of {tot/1e6:.2f}M, recover for {epochs-ep} eps', flush=True)
                    elif verbose:
                        print(f'  [clip {self.lo}-{self.hi}] delta threshold @ep{ep}: '
                              f'no matching keys, skip', flush=True)

            # 帧对(或单帧)顺序: ours 每 epoch 抽 n-1 个相邻帧对。
            # fw resource 模式: 帧对起点概率 ∝ (ρ_t^α+ρ_{t+1}^α)/2(带下限), 末段退火回均匀。
            if cfg.pair_batch and not self.is_baseline:
                use_rs = (cfg.use_fw and getattr(cfg, 'fw_mode', 'resource') == 'resource'
                          and ep < epochs * (1 - cfg.fw_anneal_frac))
                if use_rs:
                    prio_s = torch.where(torch.isnan(prio) | torch.isinf(prio) | (prio < 0),
                                          torch.ones_like(prio), prio)
                    pw = (prio_s.clamp_min(1e-8) ** cfg.fw_alpha)[:-1] + \
                         (prio_s.clamp_min(1e-8) ** cfg.fw_alpha)[1:]
                    p = (pw / pw.sum()).clamp_min(cfg.fw_floor / max(1, self.n - 1))
                    p = (p / p.sum()).cpu().numpy()
                    pair_starts = rng.choice(self.n - 1, size=self.n - 1, replace=True, p=p)
                else:
                    pair_starts = rng.permutation(self.n - 1)
            else:
                single = rng.permutation(self.n)

            ep_loss, ep_psnr, nb = 0.0, 0.0, 0
            it = 0
            pairs = ([list((int(t), int(t + 1))) for t in pair_starts] if (cfg.pair_batch and not self.is_baseline)
                     else [(int(t),) for t in single])
            # P4: 辅助损失退火系数(最后 aux_off_frac 之后线性退火到 0)
            aux_w = 1.0
            if ep_frac > cfg.aux_off_frac:
                aux_w = max(0.0, 1.0 - (ep_frac - cfg.aux_off_frac) / max(1e-6, 1.0 - cfg.aux_off_frac))

            for ids in pairs:
                imgs = self.video.frame_tensor([i + self.lo for i in ids], self.device)
                norm_idx = self._norm_idx(ids)
                fid = torch.tensor(ids, device=self.device)
                x = pe_encode(norm_idx, self.lbase, self.levels)
                out_list = model(x, fid)
                if res_fn is not None:               # V3: 最终输出 = clamp(底座 + 分支残差)
                    with torch.no_grad():
                        base_out = res_fn([i + self.lo for i in ids])
                    out_list = out_list[:-1] + [torch.clamp(out_list[-1] + base_out, 0, 1)]
                target_list = self._targets(imgs, out_list)

                recon_loss = self._spatial(out_list, target_list, ep_frac)
                # 帧权重: 'loss'=v1 加性调制(尺度不变); 'resource'=只更新采样优先级, 损失不动
                if (not self.is_baseline) and cfg.use_fw:
                    per_l1 = torch.stack([torch.abs(out_list[-1][j] - target_list[-1][j]).mean()
                                          for j in range(len(ids))])
                    if getattr(cfg, 'fw_mode', 'resource') == 'loss':
                        w = fw[list(ids)]
                        w_c = w / (w.mean() + 1e-8)
                        recon_loss = recon_loss + ((w_c - 1) * per_l1).mean()
                    else:
                        with torch.no_grad():
                            for j, fid_ in enumerate(ids):
                                val = per_l1[j].item()
                                if not (val == val) or val > 1e6:  # NaN or Inf guard
                                    val = 1.0
                                prio[fid_] = cfg.fw_ema * prio[fid_] + (1 - cfg.fw_ema) * val

                loss = recon_loss
                # ---- FFT 频域损失(P4 退火) ----
                if (not self.is_baseline) and cfg.use_fft and aux_w > 0:
                    loss = loss + cfg.fft_weight * aux_w * fft_loss(out_list[-1], target_list[-1])
                # ---- TCL 时序对比(P4 退火 + 门控) ----
                if (not self.is_baseline) and tcl_on and aux_w > 0 and \
                        ep >= max(2, int(epochs * cfg.tcl_start_frac)):
                    emb_fn = self._embedder()
                    if self.bank is None:
                        self.bank = EmbeddingBank(self.n, 256, self.device)
                    re = emb_fn(out_list[-1])
                    self.bank.update(ids, re)
                    if self.bank.init_flag.sum() >= 8:
                        lt = 0.0
                        for j, aid in enumerate(ids):
                            lt = lt + tcl_loss(re[j], aid, self.bank, self.n)
                        loss = loss + cfg.tcl_weight * aux_w * lt / max(1, len(ids))
                # ---- 率项: R+λD(warmup 后生效) / v1 朴素 delta-L1(rate_weight=0 时) ----
                # warmup: 前 rate_warm_frac 纯失真, 之后线性 ramp(先学好再压缩)
                rate_w = 0.0
                if ep_frac > cfg.rate_warm_frac:
                    rate_w = min(1.0, (ep_frac - cfg.rate_warm_frac) / max(1e-6, 1.0 - cfg.rate_warm_frac))
                if (ref_sd is not None or cfg.use_rate_dense) and rate_w > 0:
                    if cfg.rate_weight > 0:
                        loss = loss + cfg.rate_weight * rate_w * self._rate_term(ref_sd)
                    else:
                        ramp = min(1.0, max(0.0, (ep_frac - 0.7) / 0.3))
                        dl = 0.0
                        cnt = 0
                        for kk, p in model.named_parameters():
                            if kk in ref_sd and 'frame_embed' not in kk and ref_sd[kk].shape == p.shape:
                                dl = dl + torch.abs(p - ref_sd[kk]).sum()
                                cnt += p.numel()
                        loss = loss + cfg.delta_l1 * ramp * (dl / max(cnt, 1))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if self.prune_mask is not None:
                    self._apply_mask()
                if self.delta_mask is not None:
                    self._apply_delta_mask(ref_sd)

                with torch.no_grad():
                    mse = F.mse_loss(out_list[-1], target_list[-1])
                    ep_psnr += (-10 * torch.log10(mse)).item()
                ep_loss += loss.item()
                nb += 1
                it += 1
                if verbose and it % cfg.log_freq == 0:
                    print(f'  [clip {self.lo}-{self.hi}] ep{ep+1}/{epochs} it{it}/{len(pairs)} '
                          f'lr={self.opt.param_groups[0]["lr"]:.2e} loss={ep_loss/nb:.4f} '
                          f'psnr~{ep_psnr/nb:.2f} ({time.time()-t0:.0f}s)', flush=True)

            if verbose and ((ep + 1) % cfg.eval_freq == 0 or ep == epochs - 1):
                psnr = self.quick_eval(res_fn=res_fn)
                print(f'  [clip {self.lo}-{self.hi}] epoch {ep+1}/{epochs} quick PSNR={psnr:.2f} '
                      f'({time.time()-t0:.0f}s)', flush=True)

            # 断点落盘(临时文件+原子重命名, 避免写到一半被杀留下损坏文件)
            if ckpt_path and ((ep + 1) % ckpt_every == 0 or ep == epochs - 1):
                _tmp = ckpt_path + '.tmp'
                torch.save(dict(ep=ep + 1,
                                model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                                opt=self.opt.state_dict(),
                                prio=prio.detach().cpu(),
                                prune_mask=self.prune_mask, delta_mask=self.delta_mask,
                                qat_active=self._qat_active), _tmp)
                os.replace(_tmp, ckpt_path)

        # QAT 收尾: 移除参数化 → state_dict 恢复标准键名
        if self._qat_active:
            set_qat_start(self.model, False)
            self._qat_active = False
            self._orig_w, self._q_steps_trained = remove_qat(self.model)
            own = self.model.state_dict()
            self.model.load_state_dict({**own, **self._orig_w}, strict=False)

        return self.model

    # ---------------------------------------------------------- 掩码
    def _global_prune(self, keep):
        from compress import compute_global_prune_mask
        return compute_global_prune_mask(self.model.state_dict(), keep, self.prune_mask)

    def _apply_mask(self):
        with torch.no_grad():
            sd = self.model.state_dict()
            for k, m in self.prune_mask.items():
                if m is not None:
                    sd[k].mul_(m)

    def _thr_mask(self, delta_sd, keep):
        """P5: 对差分做全局幅值阈值, 返回保留掩码。"""
        from compress import compute_global_prune_mask
        return compute_global_prune_mask(delta_sd, keep)

    def _apply_delta_mask(self, ref_sd):
        """把差分掩码外的参数恢复为 ref 值(等效于差分置零)。原地操作(兼容 QAT 键名)。"""
        with torch.no_grad():
            sd = self.model.state_dict()
            for k, m in self.delta_mask.items():
                if m is None:
                    continue
                for sd_key in [k, k + '.original']:
                    if sd_key in sd and k in ref_sd and ref_sd[k].shape == sd[sd_key].shape:
                        keep_back = ref_sd[k] * (~m)
                        sd[sd_key].mul_(m).add_(keep_back)
                        break

    # ---------------------------------------------------------- 评估
    @torch.no_grad()
    def quick_eval(self, sample_stride=4, res_fn=None):
        """当前权重下 clip 的抽样 PSNR(快速日志用, 流式, 不缓存整段帧)。"""
        return self.eval_stream(stride=sample_stride, res_fn=res_fn)[0]

    @torch.no_grad()
    def eval_stream(self, state_dict=None, batch=4, stride=1, fps_reps=0, msssim=True, res_fn=None):
        """流式评估: 逐批重建-对比-释放, 返回 (平均PSNR, MS-SSIM, fps); res_fn 同 train。"""
        model = self.model
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model.eval()
        psnrs, msssims = [], []
        tlist = []
        ids_all = list(range(0, self.n, stride))
        for i in range(0, len(ids_all), batch):
            ids = ids_all[i:i + batch]
            x = pe_encode(self._norm_idx(ids), self.lbase, self.levels)
            fid = torch.tensor(ids, device=self.device)
            if fps_reps:
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(fps_reps):
                    _ = model(x, fid)
                    if res_fn is not None:
                        _ = res_fn([j + self.lo for j in ids])
                torch.cuda.synchronize()
                tlist.append(fps_reps * len(ids) / (time.time() - t0))
            out = model(x, fid)[-1]
            if res_fn is not None:
                with torch.no_grad():
                    out = out + res_fn([j + self.lo for j in ids])
            out = out.clamp(0, 1)
            gt = self.video.frame_tensor([j + self.lo for j in ids], self.device)
            for j in range(len(ids)):
                mse = F.mse_loss(out[j].float(), gt[j].float())
                psnrs.append((-10 * torch.log10(mse)).item())
            if msssim:
                from video_data import evaluate_msssim
                msssims.append(evaluate_msssim(out.cpu(), gt.cpu(), batch=len(ids)))
        fps = float(np.mean(tlist)) if tlist else None
        psnr = float(np.mean(psnrs))
        ms = float(np.mean(msssims)) if msssims else None
        model.train()
        return psnr, ms, fps
