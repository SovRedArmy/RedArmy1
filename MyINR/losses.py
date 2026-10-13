# ==========================================================================================
# MyINR 损失函数集合
#  - spatial_loss      : 原版 NeRV 的 Fusion 系列重建损失(Fusion6 = 0.7·L1 + 0.3·(1-SSIM))
#  - scale_adaptive    : MSNeRV 风格的尺度自适应多尺度损失 α·MSE + β·L1 + γ·MS-SSIM
#  - maxpool_pyramid   : max-pooling 下采样 GT 金字塔(多尺度监督目标)
#  - high_freq_boost   : MSNeRV/FaNeRV 高频增强目标 V_boost = V_gt + HP(V_gt - X̂)
#  - fft_loss          : Scene Matters 频域 FFT 损失(L1 on FFT magnitude, 零码率)
#  - flow_loss         : Scene Matters 光流约束(RAFT 风格伪标签 + 可学习置信度 + 熵二值化)
#  - tcl_loss          : Scene Matters 时序对比损失(高斯先验 InfoNCE)
# ==========================================================================================
import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


# ---------------- 基础空间损失(与原版 NeRV utils.loss_fn 一致) ----------------
def spatial_loss(pred, target, loss_type='Fusion6'):
    target = target.detach()
    if loss_type == 'L2':
        return F.mse_loss(pred, target)
    elif loss_type == 'L1':
        return torch.mean(torch.abs(pred - target))
    elif loss_type == 'SSIM':
        return 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion6':
        return 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    else:
        raise KeyError(loss_type)


# ---------------- MSNeRV 风格多尺度 ----------------
def maxpool_down(x, size):
    """max-pooling 下采样到指定尺寸(非整数倍时先 floor 取池化窗口)。"""
    h, w = x.shape[-2:]
    th, tw = size
    if (h, w) == (th, tw):
        return x
    kh, kw = max(1, h // th), max(1, w // tw)
    # 若不整除, 先裁剪到能整除的最大尺寸
    h2, w2 = th * kh, tw * kw
    x = x[..., :h2, :w2]
    return F.max_pool2d(x.float(), kernel_size=(kh, kw), stride=(kh, kw))


def scale_adaptive_weights(h):
    """按目标分辨率返回 (α, β, γ): 低分辨率偏 MSE, 高分辨率加 MS-SSIM (MSNeRV 设定)。"""
    if h < 150:
        return 0.9, 0.1, 0.0
    elif h < 300:
        return 0.8, 0.2, 0.0
    elif h < 700:
        return 0.7, 0.3, 0.0
    else:
        return 0.6, 0.3, 0.1


def scale_adaptive_loss(pred, target):
    """单尺度: α·MSE + β·L1 + γ·(1-MS-SSIM); MS-SSIM 仅在尺寸足够时启用。"""
    a, b, c = scale_adaptive_weights(target.shape[-2])
    loss = a * F.mse_loss(pred, target) + b * torch.mean(torch.abs(pred - target))
    if c > 0 and target.shape[-2] >= 160:
        loss = loss + c * (1 - ms_ssim(pred.float(), target, data_range=1, size_average=True))
    return loss


def _box_blur(x, k=5):
    """深度可分离 box blur(高通滤波器的低通部分)。"""
    c = x.shape[1]
    ker = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    ker = ker.expand(c, 1, k, k)
    return F.conv2d(F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='reflect'), ker, groups=c)


def high_freq_boosted_target(gt, recon, k=5, beta=1.0):
    """V_boost = V_gt + β·HP(V_gt - X̂), HP(r) = r - blur(r)。recon 需 detach。"""
    r = gt - recon.detach()
    hp = r - _box_blur(r, k)
    return (gt + beta * hp).clamp(0, 2)


# ---------------- FFT 频域损失 (Scene Matters: L1 on FFT, 零码率) ----------------
def fft_loss(pred, target):
    fp = torch.fft.rfft2(pred.float(), norm='ortho')
    ft = torch.fft.rfft2(target.float().detach(), norm='ortho')
    mp = torch.sqrt(fp.real ** 2 + fp.imag ** 2 + 1e-12)
    mt = torch.sqrt(ft.real ** 2 + ft.imag ** 2 + 1e-12)
    return torch.mean(torch.abs(mp - mt))


# ---------------- 光流约束 (Scene Matters SFCM: 伪标签 + 可学习置信度) ----------------
def flow_loss(pred_flow, pseudo_flow, conf, flow_scale=16.0):
    """
    pred_flow/pseudo_flow: (B,2,h,w); conf: (B,2,h,w) softmax 输出, ch1=置信度。
    L_flow = L1(w⊙f̂/s, w⊙f/s)  (前向/后向各一份, 由调用方求和)
    L_ent  = -mean(w0·log w0 + w1·log w1)   置信度二值化正则
    """
    w = conf[:, 1:2]
    l_flow = torch.mean(torch.abs(w * pred_flow / flow_scale - w * pseudo_flow.detach() / flow_scale))
    eps = 1e-6
    l_ent = -torch.mean(conf[:, 0] * torch.log(conf[:, 0] + eps) + conf[:, 1] * torch.log(conf[:, 1] + eps))
    return l_flow, l_ent


# ---------------- 时序对比损失 (Scene Matters TCL: 高斯先验 InfoNCE) ----------------
class EmbeddingBank:
    """每帧重建特征的 EMA 记忆库(detached), 供 TCL 取正负样本。"""

    def __init__(self, n_frames, dim, device, ema=0.9):
        self.ema = ema
        self.v = torch.zeros(n_frames, dim, device=device)
        self.init_flag = torch.zeros(n_frames, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, frame_ids, feats):
        if not torch.is_tensor(frame_ids):
            frame_ids = torch.tensor(frame_ids)
        for fid, f in zip(frame_ids.tolist(), feats):
            f = F.normalize(f.detach().view(-1), dim=0)
            if not self.init_flag[fid]:
                self.v[fid] = f
                self.init_flag[fid] = True
            else:
                self.v[fid] = self.ema * self.v[fid] + (1 - self.ema) * f


def tcl_loss(anchor_embed, anchor_id, bank, ids_pool, sigma2=10.0, tau=0.1, window=32, n_neg=24):
    """
    anchor_embed: (D,) 归一化前的当前帧重建特征(带梯度)。
    正样本 = 时间窗内 bank 中已初始化的帧, 目标分布 = 高斯先验 Gau(|t-j|; σ²);
    负样本 = 窗外随机 n_neg 帧。
    L = -Σ_j w_j · log[exp(sim_tj/τ) / Σ_k exp(sim_tk/τ)]
    """
    device = anchor_embed.device
    n_frames = bank.v.shape[0]
    lo, hi = max(0, anchor_id - window), min(n_frames, anchor_id + window + 1)
    win_ids = torch.tensor([j for j in range(lo, hi) if j != anchor_id], device=device)
    win_ids = win_ids[bank.init_flag[win_ids]]
    if len(win_ids) == 0:
        return torch.zeros((), device=device)

    far_mask = torch.ones(n_frames, dtype=torch.bool, device=device)
    far_mask[lo:hi] = False
    far_ids = torch.nonzero(far_mask & bank.init_flag).flatten()
    if len(far_ids) > n_neg:
        sel = torch.randperm(len(far_ids), device=device)[:n_neg]
        far_ids = far_ids[sel]

    cand = torch.cat([win_ids, far_ids])
    sims = torch.mv(bank.v[cand], F.normalize(anchor_embed.detach() * 0 + anchor_embed, dim=0)) / tau

    dist = (win_ids - anchor_id).float()
    g = torch.exp(-dist ** 2 / (2 * sigma2))
    w = g / g.sum()

    logit_pos = sims[:len(win_ids)]
    log_all = torch.logsumexp(sims, dim=0)
    loss = -(w * (logit_pos - log_all)).sum()
    return loss


# ---------------- P-model 权重差稀疏化正则 ----------------
def delta_l1_loss(model, ref_state):
    """mean(|θ - θ_ref|), 鼓励 P-model 权重差稀疏(便于阈值化+熵编码)。"""
    total, cnt = 0.0, 0
    with torch.no_grad():
        pass
    ref = ref_state
    for k, p in model.named_parameters():
        if k in ref and p.requires_grad and 'frame_embed' not in k:
            total = total + torch.abs(p - ref[k].to(p.device)).sum()
            cnt += p.numel()
    return total / max(cnt, 1)
