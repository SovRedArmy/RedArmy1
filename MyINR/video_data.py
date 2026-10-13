# ==========================================================================================
# 视频数据读取:
#  - prepare_npy      : 把逐帧 PNG 目录打包成单个 uint8 .npy (N,3,H,W), 后续 np.memmap 共享页缓存
#  - VideoData        : 内存映射读取, 提供 帧张量 / 帧对采样 / PSNR / MS-SSIM 评估
# ==========================================================================================
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim


def prepare_npy(frame_dir, npy_path):
    if os.path.isfile(npy_path):
        return npy_path
    names = sorted(os.listdir(frame_dir))
    imgs = [n for n in names if n.lower().endswith(('.png', '.jpg'))]
    first = np.array(Image.open(os.path.join(frame_dir, imgs[0])).convert('RGB'))
    H, W = first.shape[0], first.shape[1]
    arr = np.lib.format.open_memmap(npy_path, mode='w+', dtype=np.uint8, shape=(len(imgs), H, W, 3))
    for i, n in enumerate(imgs):
        arr[i] = np.array(Image.open(os.path.join(frame_dir, n)).convert('RGB'))
    arr.flush()
    return npy_path


class VideoData:
    """uint8 memmap 视频读取器。约定: 全部帧都会同时用作训练与评估(单视频过拟合协议)。"""

    def __init__(self, frame_dir, cache_dir='./cache'):
        os.makedirs(cache_dir, exist_ok=True)
        name = os.path.basename(os.path.normpath(frame_dir))
        npy_path = os.path.join(cache_dir, f'{name}_{len(sorted(os.listdir(frame_dir)))}.npy')
        prepare_npy(frame_dir, npy_path)
        self.data = np.load(npy_path, mmap_mode='r')
        self.n_frames, self.H, self.W, _ = self.data.shape
        self.name = name

    def frame_tensor(self, ids, device='cuda'):
        """ids: int 或 list → (B,3,H,W) float in [0,1]"""
        if isinstance(ids, (int, np.integer)):
            ids = [ids]
        x = torch.from_numpy(np.ascontiguousarray(self.data[list(ids)])).to(device)
        return x.permute(0, 3, 1, 2).float() / 255.0

    def frame_norm_idx(self, ids, n_total=None):
        """clip 内归一化帧索引 [0,1): ids/(n_total)。"""
        if isinstance(ids, (int, np.integer)):
            ids = [ids]
        n = n_total or self.n_frames
        return torch.tensor([i / n for i in ids], dtype=torch.float32)


def evaluate_psnr(recon_frames, gt_frames):
    """逐帧 PSNR(dB, RGB, 峰值 1) 均值。recon_frames/gt_frames: (N,3,H,W) tensors。"""
    psnrs = []
    for r, g in zip(recon_frames, gt_frames):
        mse = F.mse_loss(r.float(), g.float())
        psnrs.append((-10 * torch.log10(mse)).item())
    return float(np.mean(psnrs)), psnrs


def evaluate_msssim(recon_frames, gt_frames, batch=8):
    vals = []
    for i in range(0, len(recon_frames), batch):
        r = recon_frames[i:i + batch].float()
        g = gt_frames[i:i + batch].float()
        vals.append(ms_ssim(r, g, data_range=1).item())
    return float(np.mean(vals))
