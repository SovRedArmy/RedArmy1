# ==========================================================================================
# 伪标签光流生成器: GPU 批量化的金字塔 LK(Lucas-Kanade) 迭代光流估计。
#
# 说明: Scene Matters 用 RAFT 预训练模型生成光流伪标签; 本机无法下载预训练权重,
# 故实现了一个 RAFT 风格的"金字塔 + 多次迭代 + 中值滤波"稠密光流估计器作为伪标签来源:
#   - 高斯金字塔(6 级)由粗到细传播光流
#   - 每级多次 warp 后解 2x2 正规方程(亮度+梯度恒常)
#   - 中值滤波抑制孤立外点
# 训练端的可学习置信度头会自动降低低质量伪标签(大运动/遮挡)的权重 —— 这正是
# "带可学习置信权重过滤噪声标签"机制的作用。
# 输出: (N,2,h,w) float16 前向流(t→t+1) 与后向流(t+1→t), 存为 npz 缓存。
# ==========================================================================================
import os

import numpy as np
import torch
import torch.nn.functional as F


def _gauss_blur(x, k=5):
    c = x.shape[1]
    sigma = 1.0
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).to(x.dtype)
    kx = g.view(1, 1, 1, k).expand(c, 1, 1, k)
    ky = g.view(1, 1, k, 1).expand(c, 1, k, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='replicate'), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode='replicate'), ky, groups=c)
    return x


def _pyramid(x, levels):
    pyr = [x]
    for _ in range(levels - 1):
        x = _gauss_blur(x, 5)
        x = F.avg_pool2d(x, 2)
        pyr.append(x)
    return pyr


def _warp(x, flow):
    """按光流双线性后向 warp。flow: (B,2,h,w) (dx, dy), 像素单位。"""
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                            torch.arange(W, device=x.device, dtype=x.dtype), indexing='ij')
    xx = xx.unsqueeze(0) + flow[:, 0]
    yy = yy.unsqueeze(0) + flow[:, 1]
    xx = 2 * xx / (W - 1) - 1
    yy = 2 * yy / (H - 1) - 1
    grid = torch.stack([xx, yy], dim=-1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)


def lk_flow_pair(im0, im1, levels=6, iters=20, med_k=5):
    """
    im0, im1: (B,3,H,W) float [0,1] 同尺寸。估计 im0→im1 的前向流(以 im0 坐标为基准, 后向 warp 表示)。
    返回 (B,2,H,W): 每个像素的位移 (dx, dy), 使 im1(x+flow(x)) ≈ im0(x)。
    内部用亮度通道(通道均值)计算结构张量。
    """
    B, C, H, W = im0.shape
    lum = lambda x: x.mean(1, keepdim=True)
    pyr0, pyr1 = _pyramid(lum(im0), levels), _pyramid(lum(im1), levels)
    flow = torch.zeros(B, 2, H // (2 ** (levels - 1)), W // (2 ** (levels - 1)),
                       device=im0.device, dtype=im0.dtype)
    for lv in reversed(range(levels)):
        f0, f1 = pyr0[lv], pyr1[lv]
        h, w = f0.shape[-2:]
        # 上采样光流到当前层并按尺度放大
        if flow.shape[-2:] != (h, w):
            flow = F.interpolate(flow * 2.0, size=(h, w), mode='bilinear', align_corners=True)
        for _ in range(iters):
            w1 = _warp(f1, flow)
            gx = _grad_x(w1)
            gy = _grad_y(w1)
            # 亮度误差: im0 - warp(im1);  解正规方程 G·Δf = b, G=[[Σgx²,Σgxgy],[·,Σgy²]], b=[Σgx·er,Σgy·er]
            er = f0 - w1
            gxx = _gauss_blur(gx * gx, 5)
            gxy = _gauss_blur(gx * gy, 5)
            gyy = _gauss_blur(gy * gy, 5)
            ex = _gauss_blur(gx * er, 5)
            ey = _gauss_blur(gy * er, 5)
            det = gxx * gyy - gxy * gxy + 1e-6
            du = (ex * gyy - ey * gxy) / det
            dv = (ey * gxx - ex * gxy) / det
            step = torch.stack([du[:, 0], dv[:, 0]], dim=1)
            # 限制单步更新幅度避免发散
            step = torch.clamp(step, -8.0, 8.0)
            flow = flow + step
            flow = _median2(flow, med_k)
    return flow


def _grad_x(x):
    """replicate-pad 中心差分 (沿 W)。"""
    xp = F.pad(x, (1, 1, 0, 0), mode='replicate')
    return 0.5 * (xp[:, :, :, 2:] - xp[:, :, :, :-2])


def _grad_y(x):
    xp = F.pad(x, (0, 0, 1, 1), mode='replicate')
    return 0.5 * (xp[:, :, 2:, :] - xp[:, :, :-2, :])


def _median2(flow, k=5):
    """每个通道的 k×k 中值滤波(unfold 实现, torch2.4 无 median_pool2d)。"""
    pad = k // 2
    x = F.pad(flow, (pad,) * 4, mode='replicate')
    u = x.unfold(2, k, 1).unfold(3, k, 1)          # (B,C,H,W,k,k)
    B, C, H, W = u.shape[:4]
    return u.reshape(B, C, H, W, k * k).median(dim=-1).values


def compute_pseudo_flow(video_data, out_dir, flow_h=270, batch=4, device='cuda'):
    """对整个视频预计算 1/4 分辨率伪标签光流(t↔t+1), 缓存为 npz。"""
    os.makedirs(out_dir, exist_ok=True)
    name = video_data.name
    out_path = os.path.join(out_dir, f'{name}_flow_h{flow_h}.npz')
    if os.path.isfile(out_path):
        return out_path
    N = video_data.n_frames
    # 等比缩放到 flow_h 高度(≈ 1/4 分辨率); 输出光流即 flow_h 分辨率下的像素单位, 无需换算
    w = int(round(video_data.W * flow_h / video_data.H))
    fwd = np.zeros((N - 1, 2, flow_h, w), dtype=np.float16)
    bwd = np.zeros((N - 1, 2, flow_h, w), dtype=np.float16)
    with torch.no_grad():
        for t0 in range(0, N - 1, batch):
            idx = list(range(t0, min(t0 + batch, N - 1)))
            a = torch.cat([video_data.frame_tensor([i], device) for i in idx], 0)
            b = torch.cat([video_data.frame_tensor([i + 1], device) for i in idx], 0)
            a = F.interpolate(a, size=(flow_h, w), mode='bilinear')
            b = F.interpolate(b, size=(flow_h, w), mode='bilinear')
            f = lk_flow_pair(a, b, levels=5, iters=25)   # a→b
            r = lk_flow_pair(b, a, levels=5, iters=25)   # b→a (后向)
            for j, i in enumerate(idx):
                fwd[i] = f[j].cpu().numpy().astype(np.float16)
                bwd[i] = r[j].cpu().numpy().astype(np.float16)
    np.savez_compressed(out_path, fwd=fwd, bwd=bwd)
    return out_path


def load_pseudo_flow(path):
    z = np.load(path)
    return z['fwd'], z['bwd']
