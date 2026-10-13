# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 质量评价指标：PSNR / SSIM / MS-SSIM（纯 numpy 实现，用于测试评估）

import numpy as np

from scipy import ndimage
from scipy import signal


def calc_psnr(img1, img2, data_range=255):
    # 计算 PSNR（dB）；MSE 异常时返回 -999.9，完美重建截断到 99.9
    '''
    img1 and img2 are arrays with same shape
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean(np.square(img1 - img2))
    if np.isnan(mse) or np.isinf(mse):
        return -999.9
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    psnr = min(psnr, 99.9)
    return psnr


def fspecial_gauss(size, sigma):
    # 生成 size×size 的高斯核（SSIM 的滑窗）
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def calc_ssim(img1, img2, data_range=255):
    # 单尺度 SSIM：11x11 高斯窗（sigma=1.5），
    # 返回 (ssim_map, cs_map)，cs_map 供 MS-SSIM 的各尺度加权使用
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2

    return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                         (sigma1_sq + sigma2_sq + C2)),
            (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))


def calc_msssim(img1, img2, data_range=255):
    # MS-SSIM：5 尺度（小图降为 4 尺度，权重与 HM 实现一致），
    # 各尺度 CS 均值加权积 × 末尺度 SSIM
    '''
    img1 and img2 are 2D arrays
    '''
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    height, width = img1.shape
    if height < 176 or width < 176:
        # according to HM implementation
        level = 4
        weight = np.array([0.0517, 0.3295, 0.3462, 0.2726])
    if height < 88 or width < 88:
        assert False
    downsample_filter = np.ones((2, 2)) / 4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(level):
        ssim_map, cs_map = calc_ssim(im1, im2, data_range=data_range)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = ndimage.convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level - 1]**weight[0:level - 1]) * (mssim[level - 1]**weight[level - 1]))


def calc_msssim_rgb(img1, img2, data_range=255):
    # RGB 三通道各自算 MS-SSIM 后取平均
    '''
    img1 and img2 are arrays with 3xHxW
    '''
    msssim = 0
    for i in range(3):
        msssim += calc_msssim(img1[i, :, :], img2[i, :, :], data_range)
    return msssim / 3
