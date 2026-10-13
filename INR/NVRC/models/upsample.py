"""
Upsample utils & layers（上采样工具与层：插值的稀疏矩阵乘法实现 + 张量裁剪/填充）
"""
from .utils import *
from .patch_utils import *


def crop_tensor_nthwc(x, size, contiguous=False):
    """（裁剪 [N,T,H,W,C] 的中间区域到 size）"""
    assert len(size) == 3
    if x.shape[1:4] != tuple(size):
        crop_offset = _compute_cropping_offset(x.shape[1:4], size)
        x = x[:, crop_offset[0]:crop_offset[0] + size[0], crop_offset[1]:crop_offset[1] + size[1],
              crop_offset[2]:crop_offset[2] + size[2], :]
    if contiguous:
        x = x.contiguous()
    return x


def crop_tensor_ncthw(x, size, contiguous=False):
    """（裁剪 [N,C,T,H,W] 的中间区域）"""
    assert len(size) == 3
    if x.shape[2:5] != tuple(size):
        crop_offset = _compute_cropping_offset(x.shape[2:5], size)
        x = x[:, :, crop_offset[0]:crop_offset[0] + size[0], crop_offset[1]:crop_offset[1] + size[1],
              crop_offset[2]:crop_offset[2] + size[2]]
    if contiguous:
        x = x.contiguous()
    return x


def crop_tensor_nchw(x, size, contiguous=False):
    """（裁剪 [N,C,H,W] 的中间区域）"""
    assert len(size) == 2
    if x.shape[2:4] != tuple(size):
        crop_offset = _compute_cropping_offset(x.shape[2:4], size, 2)
        x = x[:, :, crop_offset[0]:crop_offset[0] + size[0], crop_offset[1]:crop_offset[1] + size[1]]
    if contiguous:
        x = x.contiguous()
    return x


def pad_tensor_nchw(x, size, contiguous=False, value=0.):
    """（对称填充 [N,C,H,W] 到 size）"""
    assert len(size) == 2
    if x.shape[2:4] != tuple(size):
        padding = _compute_padding_offset(x.shape[2:4], size, 2)
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]), value=value)
    if contiguous:
        x = x.contiguous()
    return x


def _compute_cropping_offset(size, target, ndims=3):
    """（居中裁剪的起始偏移）"""
    return [(size[d] - target[d]) // 2 for d in range(ndims)]


def _compute_padding_offset(size, target, ndims=3):
    """（对称填充的单侧大小）"""
    return _compute_cropping_offset(target, size, ndims)


def _interpolate(x, scale_factor, mode, align_corners):
    """（标准 F.interpolate 的封装（4D/5D））"""
    if x.ndim == 4:
        _, _, H, W = x.shape
        size = (scale_factor[0] * H, scale_factor[1] * W)
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    elif x.ndim == 5:
        _, _, T, H, W = x.shape
        size = (scale_factor[0] * T, scale_factor[1] * H, scale_factor[2] * W)
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    else:
        raise NotImplementedError


def _convert_index(x, in_size, out_size, align_corners: bool):
    # See https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/UpSample.h
    """（把输出坐标映射回输入坐标系（与 PyTorch 上采样公式一致））"""
    if align_corners:
        y = (out_size - 1.) / (in_size - 1.) * x
    else:
        y = out_size / in_size * (x + 0.5) - 0.5
    return y


class FastTrilinearInterpolation(nn.Module):
    """
    A module for switching implementaion of the trilinear interpolation.
    It also combines the interpolation with the cropping.

    （快速三线性插值：patch 模式下用稀疏矩阵乘法（matmul）实现插值，
      避免先上采样整块再裁剪的巨大中间张量；帧模式回退到 F.interpolate。）
    """
    def __init__(self, cfg):
        super().__init__()
        self.method, self.align_corners = self.get_upsampling_options(cfg)
        self.patch_spec = None

    def extra_repr(self):
        s = 'method={method}, align_corners={align_corners}'
        return s.format(**self.__dict__)

    def get_upsampling_options(self, cfg):
        """（解析配置串：method[,align]）"""
        upsample_options = cfg.split(',')
        assert len(upsample_options) <= 2
        assert len(upsample_options) < 2 or 'align' in upsample_options
        method = upsample_options[0]
        align_corners = 'align' in upsample_options
        return method, align_corners

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int],
                patch_mode: bool=True):
        """（上采样一个 patch：'interpolate' 走 F.interpolate；'matmul*' 构造插值权重矩阵分步相乘）"""
        assert x.ndim == 5
        assert all(size[d] % scale[d] == 0 for d in range(3))
        in_sizes = [size[d] // scale[d] for d in range(3)]
        out_sizes = size
        in_patch_sizes = x.shape[1:4]
        out_patch_size = [out_sizes[d] // idx_max[d] + 2 * padding[d] for d in range(3)]
        in_padding = [(in_patch_sizes[d] - in_sizes[d] // idx_max[d]) // 2 for d in range(3)]
        out_padding = padding

        N, T1, H1, W1, C = x.shape
        T2, H2, W2 = out_patch_size

        method = self.method if patch_mode else 'interpolate'

        if method == 'interpolate':
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = _interpolate(x, scale_factor=scale, mode='trilinear', align_corners=self.align_corners)
            x = x.permute(0, 2, 3, 4, 1)
            x = crop_tensor_nthwc(x, out_patch_size).contiguous()
        elif method in ['matmul', 'matmul-th-w', 'matmul-t-h-w']:
            idx_in, _ = compute_pixel_idx_3d(idx, idx_max, in_sizes, in_padding, clipped=False)
            idx_out, idx_out_mask = compute_pixel_idx_3d(idx, idx_max, out_sizes, out_padding, clipped=False)

            idx_out_p = [_convert_index(idx_out[d], out_sizes[d], in_sizes[d], align_corners=self.align_corners) \
                                       .clip_(0, in_sizes[d] - 1) \
                         for d in range(3)]
            diff = [torch.abs(idx_out_p[d][:, :, None] - idx_in[d][:, None, :]) for d in range(3)]
            weights = [(1. - diff[d]) * (diff[d] <= 1.) * idx_out_mask[d][:, :, None] for d in range(3)]

            if method == 'matmul':
                M = (weights[0][:, :, None, None, :, None, None] * \
                     weights[1][:, None, :, None, None, :, None] * \
                     weights[2][:, None, None, :, None, None, :]) \
                    .view(N, T2 * H2 * W2, T1 * H1 * W1)
                x = torch.matmul(M, x.view(N, T1 * H1 * W1, C)).view(N, T2, H2, W2, C)
                x = x.view(N, T2, H2, W2, C)
            elif method == 'matmul-th-w':
                M1 = (weights[0][:, :, None, :, None] * weights[1][:, None, :, None, :]).view(N, 1, T2 * H2, T1 * H1)
                M2 = weights[2].view(N, 1, W2, W1)
                x = torch.matmul(M1, x.view(N, 1, T1 * H1, W1 * C))
                x = torch.matmul(M2, x.view(N, T2 * H2, W1, C))
                x = x.view(N, T2, H2, W2, C)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return x


class FastNearestInterpolation(nn.Module):
    """
    A module for switching implementaion of the nearest interpolation. See FastTrilinearInterpolation for more details.
    （最近邻插值的对应实现（当前只提供帧模式 interpolate 路径））
    """    
    def __init__(self, cfg):
        super().__init__()
        self.method = self.get_upsampling_options(cfg)
        self.patch_spec = None

    def extra_repr(self):
        s = 'method={method}'
        return s.format(**self.__dict__)

    def get_upsampling_options(self, cfg):
        upsample_options = cfg.split(',')
        assert len(upsample_options) <= 2
        assert len(upsample_options) < 2 or 'align' in upsample_options
        method = upsample_options[0]
        return method

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int],
                patch_mode: bool=True):
        assert x.ndim == 5
        assert all(size[d] % scale[d] == 0 for d in range(3))
        in_sizes = tuple(size[d] // scale[d] for d in range(3))
        out_sizes = size
        in_patch_sizes = tuple(x.shape[1:4])
        out_patch_size = tuple(out_sizes[d] // idx_max[d] + 2 * padding[d] for d in range(3))
        in_padding = tuple((in_patch_sizes[d] - in_sizes[d] // idx_max[d]) // 2 for d in range(3))
        out_padding = tuple(padding)

        N, T1, H1, W1, C = x.shape
        T2, H2, W2 = out_patch_size

        method = self.method if patch_mode else 'interpolate'

        if method == 'interpolate':
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = F.interpolate(x, scale_factor=scale, mode='nearest')
            x = x.permute(0, 2, 3, 4, 1)
            x = crop_tensor_nthwc(x, out_patch_size).contiguous()
        else:
            raise NotImplementedError
        return x