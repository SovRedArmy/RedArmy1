"""
损失与指标库：所有函数均按帧(frame-wise)计算，输入输出为 [N, C, T, H, W]。
基础损失与指标沿用 HiNeRV（0.7·L1 + 0.3·MS-SSIM(5×5)，PSNR / MS-SSIM）。

新增损失（MyHiNeRV）：
- fft：频域 FFT 损失（Scene Matters L_freq = L1(FFT(x̂), FFT(x))，零码率），
  可选高频加权（high_freq_gamma>0 时按频率半径加权，进一步强调高频）；
- temporal_contrastive：时序对比损失（Scene Matters TCL）——InfoNCE + 时间高斯软标签，
  拉近相邻帧的 CRSPE 嵌入；
- flow_constraint：RAFT 伪标签光流约束 + 置信度熵正则（Scene Matters SFCM），
  可学习置信权重用于过滤伪标签噪声；
- maxpool_pyramid / highpass / highfreq_enhanced_target：多尺度金字塔与
  动态高频注入（FaNeRV / MSNeRV 频率尺度分解）的工具函数。
"""
from utils import *
import pytorch_msssim


def check_shape(x, y):
    """
    校验两个张量形状一致且为 5D（[N,C,T,H,W]）；步长仅在维度大小 >1 时比较
    （大小为 1 的维度步长不唯一——例如 T=1 的 permute 张量——不影响后续 view）。
    """
    assert x.shape == y.shape, 'shape of tensors must be the same!'
    assert all(x.stride()[d] == y.stride()[d] or x.shape[d] == 1 for d in range(x.ndim)), \
        'strides of tensors must be compatible!'
    assert x.ndim == y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'


def mse(x, y):
    """
    Compute the per-frame MSE loss
    逐帧 MSE：返回 [N, T]，每帧一个值。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.mse_loss(x, y, reduction='none').mean(dim=2)


def l1(x, y):
    """
    Compute the per-frame L1 loss
    逐帧 L1：返回 [N, T]。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.l1_loss(x, y, reduction='none').mean(dim=2)


def psnr(x, y, v_max=1.):
    """
    Compute the per-frame PSNR
    逐帧 PSNR = 10·log10(v_max² / MSE)。
    """
    return 10 * torch.log10((v_max ** 2) / (mse(x, y) + 1e-9))


def ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame SSIM
    逐帧 SSIM（基于 pytorch_msssim），返回 [N, T]。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


def ms_ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame MS-SSIM
    逐帧多尺度 SSIM，返回 [N, T]。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ms_ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


"""
Frequency-domain tools (Scene Matters / FaNeRV)
"""
_fftfreq_cache = {}


def _radial_weight(H, W, device, dtype, gamma):
    """
    构造 rFFT 幅度谱的径向频率权重 w = 1 + gamma * (r/r_max)^2（r=0 在直流处）。
    """
    key = (H, W, device, dtype, gamma)
    if key not in _fftfreq_cache:
        fy = torch.fft.fftfreq(H, device=device).to(dtype)  # [-0.5, 0.5)
        fx = torch.fft.rfftfreq(W, device=device).to(dtype)  # [0, 0.5]
        r = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
        r_max = (0.5 ** 2 + 0.5 ** 2) ** 0.5
        w = 1. + gamma * (r / r_max) ** 2
        _fftfreq_cache[key] = w
    return _fftfreq_cache[key]


def fft_loss(x, y, high_freq_gamma=0.):
    """
    频域 FFT 损失（Scene Matters Eq.8）：L1(FFT(x̂), FFT(x))，逐帧 2D FFT 的幅度谱。
    high_freq_gamma > 0 时按频率半径加权（1 + gamma·(r/r_max)²），进一步保高频。
    返回 [N, T]（与其它逐帧损失一致）。
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    xf = torch.fft.rfft2(x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W), norm='ortho')
    yf = torch.fft.rfft2(y.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W), norm='ortho')
    weight = _radial_weight(H, W, x.device, x.dtype, high_freq_gamma) if high_freq_gamma > 0. else 1.
    loss = F.l1_loss(xf.abs() * weight, yf.abs() * weight, reduction='none').mean(dim=(1, 2, 3))
    return loss.view(N, T)


_hp_kernel_cache = {}


def _laplacian_kernel(device, dtype):
    """3×3 拉普拉斯高通核（depthwise，按通道重复）。"""
    key = (device, dtype)
    if key not in _hp_kernel_cache:
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device, dtype=dtype)
        _hp_kernel_cache[key] = k
    return _hp_kernel_cache[key]


def highpass(x):
    """
    高通滤波（FaNeRV 动态高频注入的 H 算子）：3×3 拉普拉斯卷积（depthwise，same padding）。
    输入/输出均为 [N, C, T, H, W]。
    """
    N, C, T, H, W = x.shape
    k = _laplacian_kernel(x.device, x.dtype)
    xf = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
    out = F.conv2d(xf, k[None, None].expand(C, 1, 3, 3), padding=1, groups=C)
    return out.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)


def highfreq_enhanced_target(y, x_hat, beta=1.):
    """
    动态高频注入的目标构造（FaNeRV）：V_target = clamp(V + β·HP(V − sg(V̂)), 0, 1)。
    残差对当前重建 x_hat 做 stop-gradient（目标是"增强参考"而非改写损失尺度）；
    必须把目标裁剪回 [0,1]——输出头是 sigmoid（值域 (0,1)），界外目标会使模型
    饱和、梯度消失并训练崩塌（消融中在注入启动的 epoch 观察到 PSNR 34→5dB）。
    """
    residual = y - x_hat.detach()
    return (y + beta * highpass(residual)).clamp(0., 1.)


def maxpool_pyramid(x, ratios):
    """
    GT 金字塔（FaNeRV）：对 [N, C, T, H, W] 按各比率做空间 max-pooling 下采样。
    ratios: 每级的空间下采样倍率列表（相对输入分辨率）。
    返回 [按 ratios 顺序的小视频列表]。
    """
    pyramid = []
    N, C, T, H, W = x.shape
    for r in ratios:
        assert H % r == 0 and W % r == 0, f'spatial size {H}x{W} not divisible by ratio {r}'
        y = F.max_pool2d(x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W), kernel_size=r, stride=r)
        y = y.view(N, T, C, H // r, W // r).permute(0, 2, 1, 3, 4)
        pyramid.append(y.contiguous())
    return pyramid


def temporal_contrastive_loss(embed: torch.Tensor, frame_ids: torch.Tensor, sigma=10., tau=0.1):
    """
    时序对比损失（Scene Matters TCL 的嵌入版）：
    以时间距离的高斯分布作为软标签的 InfoNCE，拉近相邻帧嵌入。
    - embed: [T, C] 帧嵌入表（如 CRSPE weight，梯度可流回）；
    - frame_ids: 本 batch 出现的（clip 内）帧坐标张量（任意形状，会被展平去重）；
    - 候选集 = batch 内出现的帧；正样本权重 w_ij ∝ exp(-(i-j)²/2σ²)，逐行归一化；
    - 损失 = -Σ_j w_ij · log softmax_j(cos(z_i, z_j)/τ)，排除自身。
    batch 内帧数 < 2 时返回 None。
    """
    ids = frame_ids.detach().reshape(-1).unique().to(embed.device)
    if ids.numel() < 2:
        return None
    z = F.normalize(embed[ids], dim=1, eps=1e-8)  # [M, C]
    logits = z @ z.t() / tau  # [M, M]
    M = ids.shape[0]
    eye = torch.eye(M, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(eye, float('-inf'))
    log_prob = torch.log_softmax(logits, dim=1)  # 自身位置为 -inf → 0 概率
    dt = (ids[:, None] - ids[None, :]).float()
    w = torch.exp(-dt ** 2 / (2 * sigma ** 2))
    w = w.masked_fill(eye, 0.)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    loss = -(w * log_prob.masked_fill(torch.isinf(log_prob), 0.)).sum(dim=1)
    valid = w.sum(dim=1) > 0
    if valid.sum() == 0:
        return None
    return loss[valid].mean()


def flow_constraint_loss(pred_f, pred_b, conf, gt_f, gt_b, valid=None):
    """
    RAFT 伪标签光流约束（Scene Matters SFCM，Eq.9-11）：
      L_flow = L1(W1·O_f, W1·O_f_gt) + L1(W1·O_b, W1·O_b_gt)   （W1 可学习，梯度可流回置信头）
      L_ent  = -(W0·log W0 + W1·log W1)                          （置信度二值化正则）
    pred_f/pred_b: [N, 2, T, H, W]；conf: [N, 2, T, H, W]（softmax 已归一化）；
    gt_f/gt_b: 同形状伪标签；valid: [N, T] 帧级有效掩码（clip 首帧无伪标签）。
    返回 (L_flow, L_ent) 标量。
    """
    w1 = conf[:, 1:2]  # [N, 1, T, H, W] 置信权重（Scene Matters 的 W^(1)）
    if valid is not None:
        mask = valid[:, None, :, None, None].to(pred_f.dtype)
    else:
        mask = torch.ones_like(w1)
    n = mask.sum().clamp(min=1.) * pred_f.shape[1] * pred_f.shape[3] * pred_f.shape[4]  # 每方向的元素数
    loss_flow = ((w1 * mask * (pred_f - gt_f).abs()).sum() / n) + \
                ((w1 * mask * (pred_b - gt_b).abs()).sum() / n)
    ent_map = -(conf * torch.log(conf.clamp_min(1e-8))).sum(dim=1, keepdim=True)  # [N, 1, T, H, W]
    loss_ent = (ent_map * mask).sum() / n
    return loss_flow, loss_ent


def compute_loss(name, x, y):
    """
    损失分发：按名称调用对应损失函数。
    SSIM 类损失返回 (1 - SSIM)；窗口大小可通过名称后缀 _3x3/_5x5/_7x7 指定。
    新增 'fft'（纯 Scene Matters 频域损失）与 'fft_hf'（高频加权的 FFT 损失）。
    """
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'ms_ssim':
        name = 'ms-ssim'  # 兼容别名
    if name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'ssim':
        return 1. - ssim(x, y)
    elif name == 'ssim_3x3':
        return 1. - ssim(x, y, win_size=3)
    elif name == 'ssim_5x5':
        return 1. - ssim(x, y, win_size=5)
    elif name == 'ssim_7x7':
        return 1. - ssim(x, y, win_size=7)
    elif name == 'ms-ssim':
        return 1. - ms_ssim(x, y)
    elif name == 'ms-ssim_3x3':
        return 1. - ms_ssim(x, y, win_size=3)
    elif name == 'ms-ssim_5x5':
        return 1. - ms_ssim(x, y, win_size=5)
    elif name == 'ms-ssim_7x7':
        return 1. - ms_ssim(x, y, win_size=7)
    elif name == 'fft':
        return fft_loss(x, y, high_freq_gamma=0.)
    elif name == 'fft_hf':
        return fft_loss(x, y, high_freq_gamma=4.)
    else:
        raise ValueError


def compute_metric(name, x, y):
    """指标分发：按名称调用 mse/l1/psnr/ssim/ms-ssim。"""
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'ms_ssim':
        name = 'ms-ssim'  # 兼容别名
    if name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'psnr':
        return psnr(x, y)
    elif name == 'ssim':
        return ssim(x, y)
    elif name == 'ms-ssim':
        return ms_ssim(x, y)
    else:
        raise ValueError


def compute_regularization(name, model):
    """正则项分发（当前未实现任何正则）。"""
    raise ValueError
