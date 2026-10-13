# ==========================================================================================
# 训练动态(dynamicAL)与场景切分(TeNeRV 风格)
#
#  - probe_dynamics : 用一个小型"探针"NeRV 在整段视频上短训若干 epoch(batch=1 单帧,
#                     每步梯度范数即该帧的 per-sample 梯度范数 —— dynamicAL 的训练动态信号),
#                     返回每帧平均梯度范数 g_t。
#  - frame_weights  : 由 g_t 映射到 [w_min, w_max]、均值≈1 的帧权重(训练动态高的帧更受重视)。
#  - scene_split    : TeNeRV 风格场景边界 —— 结合 (a) 相邻帧内容特征余弦距离
#                     (b) 训练动态突变 |g_t - g_{t-1}|, 两个指标各自 robust z-score 后融合,
#                     贪心选取 top-K 边界(带最小段长约束)把视频切成 K 个 clip。
# ==========================================================================================
import numpy as np
import torch
import torch.nn.functional as F

from modules import NeRVGenerator
from video_data import VideoData


class _PE:
    """帧索引位置编码(与原版 NeRV utils.PositionalEncoding 相同)。"""

    def __init__(self, pe_embed='1.25_40'):
        self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
        self.levels = int(self.levels)
        self.embed_length = 2 * self.levels

    def __call__(self, pos):
        pe_list = []
        for i in range(self.levels):
            v = pos * self.lbase ** i * np.pi
            pe_list += [torch.sin(v), torch.cos(v)]
        return torch.stack(pe_list, 1)


PositionalEncoding = _PE


@torch.no_grad()
def _robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def probe_dynamics(video: VideoData, device='cuda', epochs=8, lr=1e-3, seed=1,
                   stem='128_1', fc_hw='9_16_8', lower=8, cache_dir=None):
    """训练探针模型并返回每帧平均梯度范数 (N,)。"""
    N = video.n_frames
    torch.manual_seed(seed)
    pe = PositionalEncoding()
    model = NeRVGenerator(embed_length=pe.embed_length, stem_dim_num=stem, fc_hw_dim=fc_hw,
                          lower_width=lower, n_frames=0, crspe_channels=0, multi_scale=False,
                          stride_list=_strides_for(video), act='swish').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gn = np.zeros((epochs, N), dtype=np.float64)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        order = torch.randperm(N, generator=g).tolist()
        model.train()
        for i in order:
            img = video.frame_tensor([i], device)
            idx = video.frame_norm_idx([i], N).to(device)
            out = model(pe(idx), torch.tensor([i], device=device))[0]
            loss = F.l1_loss(out, img)
            opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                s = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        s += p.grad.pow(2).sum().item()
            gn[ep, i] = np.sqrt(s)
            opt.step()
    return gn[-3:].mean(0)


def _strides_for(video):
    """按分辨率选择上采样倍率列表: 1080p→[5,3,2,2,2], 720p→[5,2,2,2,2]。"""
    h, w = video.H, video.W
    for strides in ([5, 3, 2, 2, 2], [5, 2, 2, 2, 2], [3, 2, 2, 2, 2], [2, 2, 2, 2]):
        p = 1
        for s in strides:
            p *= s
        if h % p == 0 and w % p == 0 and h // p >= 9:
            return strides
    return [5, 3, 2, 2, 2]


def frame_weights(grad_norms, w_min=0.8, w_max=1.2):
    """dynamicAL 帧权重: 训练动态(梯度范数)越高 → 权重越高。经验上范围要温和
    (大范围会让低权重帧欠拟合), 默认 [0.8, 1.2], 均值归一到 1。"""
    z = _robust_z(grad_norms)
    z = np.clip(z, -2, 2) / 2.0                      # [-1, 1]
    w = w_min + (z + 1) / 2 * (w_max - w_min)
    return w / w.mean()


def scene_split(cos_dist, grad_norms, n_frames, target_len=100, min_len=None):
    """
    cos_dist   : (N,) 相邻帧特征余弦距离 D_t = 1 - cos(h_t, h_{t-1}), 首帧无效置 NaN→0
    grad_norms : (N,) 每帧训练动态
    返回边界列表 [0, b1, ..., N](升序), 即 K = len-1 个 clip。
    """
    if min_len is None:
        min_len = max(16, target_len // 2)
    K = max(1, int(round(n_frames / target_len)))
    if K <= 1:
        return [0, n_frames]

    # 边界得分对齐: score[t] 表示"在第 t 帧前切开"(t ∈ [1, N-1])
    s1 = _robust_z(cos_dist[1:])                       # 内容差异: 帧间余弦距离
    dg = np.abs(np.diff(grad_norms))                   # 训练动态突变: |g_t - g_{t-1}|
    dg = np.concatenate([[0.0], dg])                   # 对齐到边界 t
    s2 = _robust_z(dg[1:])
    score = 0.5 * s1 + 0.5 * s2
    order = np.argsort(-score)

    bounds = []
    for t in order + 1:                                # 候选内部边界 t ∈ [1, N-1]
        if len(bounds) >= K - 1:
            break
        if any(abs(t - b) < min_len for b in bounds) or t < min_len or t > n_frames - min_len:
            continue
        bounds.append(int(t))
    bounds = sorted(bounds)
    return [0] + bounds + [n_frames]


@torch.no_grad()
def content_cos_dist(video: VideoData, embeds):
    """相邻帧内容特征的余弦距离 (N,) (帧 0 为 0)。embeds: (N,D) 已归一化的特征。"""
    e = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    cos = np.sum(e[1:] * e[:-1], axis=1)
    return np.concatenate([[0.0], 1.0 - cos])
