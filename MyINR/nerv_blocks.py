# ==========================================================================================
# 原版 NeRV 的基础模块(与 INR/NeRV/model_nerv.py 保持一致), 供 NeRVGenerator 使用。
# 复制到本目录以保持 MyINR 自包含, 不改动原仓库代码。
# ==========================================================================================
import torch
import torch.nn as nn


def ActivationLayer(act_type):
    if act_type == 'relu':
        return nn.ReLU(True)
    elif act_type == 'leaky':
        return nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        return nn.ReLU6()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'sin':
        return torch.sin
    elif act_type == 'swish':
        return nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        return nn.Softplus()
    elif act_type == 'hardswish':
        return nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")


def NormLayer(norm_type, ch_width):
    if norm_type == 'none':
        return nn.Identity()
    elif norm_type == 'bn':
        return nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError


class CustomConv(nn.Module):
    """可切换上采样方式的卷积块: conv(PixelShuffle) / deconv / bilinear."""

    def __init__(self, **kargs):
        super().__init__()
        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2 * stride + 1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        return self.up_scale(self.conv(x))


def MLP(dim_list, act='relu', bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i + 1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.conv = CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'],
                               bias=kargs['bias'], conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
