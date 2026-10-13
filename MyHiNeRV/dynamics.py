"""
训练动态（training dynamics）工具：帧权重采样与场景切分。

借鉴：
- dynamicAL（Deep Active Learning by Leveraging Training Dynamics, NeurIPS 2022）：
  其 acquisition 把"训练动态"（样本损失/输出的均值 μ 与方差 σ）组合为
  score = norm(μ) + λ·norm(σ)，方差大 = 训练动态高 = 该样本仍不稳定/难学。
  本实现把该思想迁移到 INR 视频过拟合：以每帧重建损失轨迹的 (μ_t, σ_t) 作为
  训练动态得分，动态得分高的帧获得更高的采样概率（更多训练资源）。
  注意：只改采样分布，绝不在损失函数里给帧加权（约束）。
- TeNeRV（Content-Aware GoP Partitioning）：以相邻帧特征的余弦散度
  D_t = 1 - cos(X_t, X_{t-1}) 检测内容变化，按散度降序贪心选择 K-1 个
  clip 边界并保证最小 clip 长度 L_min。本实现把"散度分数"扩展为
  内容余弦散度 + 训练动态得分 的加权和（场景突变帧往往损失动态也高）。
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def zscore(x: np.ndarray, eps=1e-8):
    """标准化到零均值单位方差（NaN → 0）。"""
    x = np.nan_to_num(x.astype(np.float64), nan=0.)
    return (x - x.mean()) / (x.std() + eps)


class FrameDynamicsTracker:
    """
    逐 epoch 记录每帧平均损失，维护滑动窗口内的均值 μ_t 与标准差 σ_t，
    计算动态AL 风格的帧得分 s_t = norm(μ_t) + λ·norm(σ_t)。
    """

    def __init__(self, num_frames, window=10, var_weight=2.0, ema=0.7):
        assert num_frames > 0
        self.num_frames = num_frames
        self.window = window        # 统计窗口（最近多少个 epoch）
        self.var_weight = var_weight  # dynamicAL 的 λ（默认 2，与其论文一致）
        self.ema = ema              # 得分 EMA 平滑系数
        self.history = []           # 每个 entry: [T] 该 epoch 每帧平均损失
        self.scores = np.zeros(num_frames, dtype=np.float64)

    def update(self, epoch, frame_losses):
        """
        frame_losses: {t: 该 epoch 累计的帧平均损失}（t 为 clip 内局部帧号）。
        训练循环每个 epoch 调用一次；窗口外的历史丢弃。
        """
        losses = np.full(self.num_frames, np.nan)
        for t, l in frame_losses.items():
            if 0 <= t < self.num_frames:
                losses[t] = float(l)
        # 尚未被采样到的帧用该 epoch 均值填充（防止 NaN 传播）
        valid = ~np.isnan(losses)
        fill = losses[valid].mean() if valid.any() else 0.
        losses = np.where(valid, losses, fill)
        self.history.append(losses)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

        h = np.stack(self.history, axis=0)  # [W, T]
        mu = h.mean(axis=0)
        sigma = h.std(axis=0)
        raw = zscore(mu) + self.var_weight * zscore(sigma)
        self.scores = self.ema * self.scores + (1. - self.ema) * raw
        return self.scores.copy()

    def get_scores(self):
        return self.scores.copy()


class PatchDynamicsTracker:
    """
    patch 级训练动态跟踪（dynamicAL 的 patch 版）：
    逐 epoch 记录每个 3D 补丁的平均损失（μ+λσ 得分），用于
    a) 选择性丢弃低动态 patch（MSNeRV 随机丢 10~15% 不掉点的动态版，默认丢 25%）；
    b) 聚合成帧级得分供场景切分（帧得分 = 该帧所有 patch 得分的均值）。
    """
    def __init__(self, num_patches, window=10, var_weight=2.0, ema=0.7):
        assert len(num_patches) == 3
        self.num_patches = tuple(num_patches)
        self.num_frames = num_patches[0]
        self.n_hw = num_patches[1] * num_patches[2]
        self.window = window
        self.var_weight = var_weight
        self.ema = ema
        self.history = []  # 每 entry: [P] 该 epoch 每 patch 平均损失（未采样→NaN）
        self.scores = np.zeros(self.num_frames * self.n_hw, dtype=np.float64)

    def update(self, epoch, patch_sum: np.ndarray, patch_cnt: np.ndarray):
        """
        patch_sum/patch_cnt: [P] 该 epoch 各 patch 的损失累计与采样次数（未采样 cnt=0）。
        """
        P = self.num_frames * self.n_hw
        assert patch_sum.shape == (P,)
        losses = np.where(patch_cnt > 0, patch_sum / np.maximum(patch_cnt, 1), np.nan)
        valid = ~np.isnan(losses)
        fill = losses[valid].mean() if valid.any() else 0.
        losses = np.where(valid, losses, fill)
        self.history.append(losses)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

        h = np.stack(self.history, axis=0)  # [W, P]
        mu = h.mean(axis=0)
        sigma = h.std(axis=0)
        raw = zscore(mu) + self.var_weight * zscore(sigma)
        self.scores = self.ema * self.scores + (1. - self.ema) * raw
        return self.scores.copy()

    def get_scores(self):
        """patch 级得分 [P]（线性索引按 t 优先展开，与数据集一致）。"""
        return self.scores.copy()

    def get_frame_scores(self):
        """帧级得分 [T]（该帧所有 patch 得分的均值），供场景切分。"""
        return self.scores.reshape(self.num_frames, self.n_hw).mean(axis=1)


def compute_frame_weights(scores: np.ndarray, temperature=0.5, max_ratio=5.0, min_w=0.2):
    """
    由动态得分计算帧采样权重：
      w_t ∝ exp(temperature · z(s))，并截断到 [min_w, max_ratio]（相对均匀权重 1.0）。
    temperature=0 时退化为均匀采样。
    """
    T = len(scores)
    if temperature <= 0.:
        return np.ones(T)
    z = zscore(scores)
    w = np.exp(temperature * z)
    w = np.clip(w, min_w, max_ratio)
    return w


def frame_patch_weights(frame_weights: np.ndarray, num_patches, keep_mask: np.ndarray = None) -> torch.Tensor:
    """
    把帧权重展开成补丁级采样权重（帧 t 的每个补丁同权）。
    num_patches: (Pt_num, Ph_num, Pw_num) 每维补丁数。
    keep_mask: [P] bool 数组——dynamicAL patch 丢弃（False 的补丁权重置 0，被丢弃）。
    返回长度 = 总补丁数 的 DoubleTensor（顺序与数据集线性索引一致，见 datasets.__getitem__）。
    """
    n_t, n_h, n_w = num_patches
    w = np.asarray(frame_weights, dtype=np.float64)
    patch_w = np.repeat(w, n_h * n_w)  # 线性索引按 t 优先展开
    if keep_mask is not None:
        assert keep_mask.shape == patch_w.shape
        patch_w = patch_w * keep_mask.astype(np.float64)
    return torch.as_tensor(patch_w, dtype=torch.double)


def compute_patch_train_weights(patch_scores: np.ndarray, temp=1.0, w_min=0.5, w_max=2.0):
    """
    patch 级训练采样权重（激进版，不丢弃任何 patch）：
    w = clip(exp(temp · z_score), w_min, w_max)。
    难 patch（高动态得分）被采样更频繁，获得更多梯度步；
    易 patch 权重不低于 w_min，保证持续训练不"饿死"。
    """
    z = zscore(patch_scores)
    w = np.exp(temp * z)
    return np.clip(w, w_min, w_max)


def compute_patch_loss_weights(patch_scores: np.ndarray, alpha=0.15, clip_delta=0.2):
    """
    patch 级损失权重（保守版，降低 MSE 方差以缩小 Jensen 偏移）：
    w = clip(1 + alpha · z_score, 1-clip_delta, 1+clip_delta)。
    难 patch 的误差在损失中权重略高 → 梯度偏向压低其 MSE；
    与采样权重同源（同一 dynamics 得分），但映射温和（如 [0.85, 1.15]）。
    alpha=0 时全部权重为 1（关闭）。
    """
    if alpha <= 0.:
        return np.ones(len(patch_scores))
    z = zscore(patch_scores)
    return np.clip(1.0 + alpha * z, 1.0 - clip_delta, 1.0 + clip_delta)


def compute_patch_drop_mask(patch_scores: np.ndarray, drop_ratio=0.25):
    """
    dynamicAL 式 patch 丢弃（MSNeRV patch-dropout 的动态版）：
    丢掉训练动态得分最低的 drop_ratio 比例的 patch（低动态 = 已学好/稳定，
    继续训练收益最小），保留高动态 patch。
    返回 keep_mask（True=保留训练）。
    """
    P = len(patch_scores)
    n_drop = int(round(P * drop_ratio))
    keep = np.ones(P, dtype=bool)
    if n_drop <= 0:
        return keep
    order = np.argsort(patch_scores)  # 升序：最低动态在前
    keep[order[:n_drop]] = False
    return keep


@torch.no_grad()
def compute_content_divergence(dataset, size=64, device='cpu', batch_frames=32):
    """
    相邻帧内容特征的余弦散度（TeNeRV 的 D_t，作用在下采样 GT 内容特征上）：
      D_t = 1 - cos(X_t, X_{t-1})，X_t = resize(GT_t, size×size) 展平（L2 归一化）。
    返回 [T] 的 numpy 数组（D_0 = 0）。
    """
    T = dataset.video_size[0]
    feats = []
    for s in range(0, T, batch_frames):
        imgs = torch.stack([dataset.get_image(t).float() / 255. for t in range(s, min(s + batch_frames, T))])
        f = F.interpolate(imgs, size=(size, size), mode='bilinear', align_corners=False)
        f = f.flatten(1).to(device)
        feats.append(F.normalize(f, dim=1, eps=1e-8).cpu())
    feats = torch.cat(feats, dim=0)  # [T, D]
    cos = (feats[1:] * feats[:-1]).sum(dim=1)  # [T-1]
    div = 1. - cos.numpy()
    return np.concatenate([[0.], div])  # D_0 = 0


def segment_clips(divergence: np.ndarray, dynamics: np.ndarray = None, num_clips=None, clip_len=64,
                  min_len=8, max_len=None, dyn_weight=0.5):
    """
    场景切分（TeNeRV Content-Aware GoP Partitioning 的扩展版）：
    综合"内容余弦散度 + 训练动态得分"作为边界分数，按分数降序贪心选 K-1 个边界，
    迭代保证任意 clip 长度 ≥ min_len。

    - divergence: [T] 相邻帧内容余弦散度（边界候选分数 1）
    - dynamics:   [T] 训练动态得分（边界候选分数 2，可为 None）
    - num_clips:  目标 clip 数 K（None 时按 clip_len 自动估计 K = round(T / clip_len)）
    - clip_len:   仅用于估计 K 的目标长度
    - min_len:    最小 clip 长度约束
    - max_len:    最大 clip 长度约束（None 不限制；超长 clip 在内部融合分数峰处递归切分）
    - dyn_weight: 训练动态分数的融合权重 w（score = z(div) + w·z(dyn)）
    返回 clips: [(start, end), ...]（end 为开区间，即帧 [start, end)）
    """
    T = len(divergence)
    assert T >= 2 * min_len, 'video too short to split'

    score = zscore(divergence)
    if dynamics is not None and dyn_weight > 0.:
        score = score + dyn_weight * zscore(dynamics)

    if num_clips is None:
        num_clips = max(1, int(round(T / clip_len)))
    num_clips = max(1, min(num_clips, T // min_len))
    K = num_clips

    # TeNeRV Alg.1：按分数降序取候选边界，贪心满足最小长度约束
    # 候选边界 b 表示在帧 b-1 与 b 之间切开（b ∈ [1, T-1]）
    candidates = [int(b) + 1 for b in np.argsort(-score[1:T])]  # argsort 位置 → 帧号 +1
    boundaries = []
    for b in candidates:
        if len(boundaries) >= K - 1:
            break
        trial = sorted(boundaries + [b])
        lengths = np.diff([0] + trial + [T])
        if lengths.min() >= min_len:
            boundaries = trial

    bounds = [0] + sorted(boundaries) + [T]
    clips = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]

    # 超长 clip 递归切分：在其内部融合分数峰处一分为二（两侧仍满足 min_len）
    if max_len is not None and max_len > min_len:
        changed = True
        while changed:
            changed = False
            for i, (aa, bb) in enumerate(clips):
                if bb - aa > max_len and bb - aa >= 2 * min_len:
                    inner = np.arange(aa + min_len, bb - min_len + 1)
                    m = int(inner[np.argmax(score[inner])])
                    clips[i:i + 1] = [(aa, m), (m, bb)]
                    changed = True
                    break
    return clips


def set_dynamics_args(parser):
    """注册训练动态（帧权重采样/场景切分）相关参数。"""
    group = parser.add_argument_group('Training dynamics parameters (dynamicAL / TeNeRV)')
    group.add_argument('--dyn-enabled', type=lambda s: str(s).lower() == 'true', default=True,
                       help='enable dynamicAL-style frame-weighted sampling (never re-weights the loss).')
    group.add_argument('--dyn-update-epochs', type=int, default=5,
                       help='update the weighted/drop sampler every N epochs.')
    group.add_argument('--dyn-warmup-epochs', type=int, default=10,
                       help='uniform sampling for the first N epochs (dynamics warm-up).')
    group.add_argument('--dyn-temperature', type=float, default=0.5,
                       help='softmax temperature of frame sampling weights.')
    group.add_argument('--dyn-max-ratio', type=float, default=5.0,
                       help='max frame sampling weight ratio (relative to uniform 1.0).')
    group.add_argument('--dyn-window', type=int, default=10,
                       help='window (epochs) of per-frame loss statistics.')
    group.add_argument('--dyn-var-weight', type=float, default=2.0,
                       help='lambda of the dynamicAL score norm(mu)+lambda*norm(sigma).')
    group.add_argument('--patch-drop-start', type=float, default=0.0, help='progress to start patch dropout (cosine window with --patch-drop-end).')
    group.add_argument('--patch-drop-end', type=float, default=1.0, help='progress to stop patch dropout (default 1.0 = active throughout, legacy behavior).')
    group.add_argument('--dyn-sampling-temp', type=float, default=1.0,
                       help='temperature for patch-level training sampling weights (aggressive).')
    group.add_argument('--dyn-sampling-w-min', type=float, default=0.5,
                       help='min sampling weight for any patch (all patches participate, none starved).')
    group.add_argument('--dyn-sampling-w-max', type=float, default=2.0,
                       help='max sampling weight for any patch.')
    group.add_argument('--dyn-loss-alpha', type=float, default=0.0,
                       help='slope for patch-level loss weights from dynamics scores (conservative). 0: disabled.')
    group.add_argument('--dyn-loss-clip', type=float, default=0.2,
                       help='max deviation of loss weight from 1.0.')
    group.add_argument('--sampling-window', type=str, default='',
                       help='cosine window for sampling weights "p0,p1,p2,p3" (uniform at edges, differentiated in middle). Empty: always differentiated.')
    group.add_argument('--patch-drop-ratio', type=float, default=0.25,
                       help='drop the lowest-dynamics ratio of patches each update (MSNeRV patch-dropout, dynamicAL-selected). 0: disabled')
    group.add_argument('--seg-dyn-weight', type=float, default=0.5,
                       help='weight of the dynamics score in scene segmentation score fusion.')
    group.add_argument('--seg-feature-size', type=int, default=64,
                       help='resolution of the downsampled GT feature for cosine divergence.')
