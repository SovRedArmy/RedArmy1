# -*- coding: utf-8 -*-
"""原始 NeRV 基线模型及其网络组件。

论文: NeRV: Neural Representations for Videos (NeurIPS 2021, arXiv:2110.13903)
NeRV 的思路：帧号 t -> 位置编码 -> MLP 直接回归出整帧的初始特征图 [C,h,w] -> 逐级上采样卷积 -> 图像。

本文件同时提供被 E-NeRV 复用的组件：
    NeRV_MLP()       通用 MLP 堆叠（Linear + 激活交替）
    NeRV_CustomConv  三种上采样卷积实现（conv+PixelShuffle / 转置卷积 / 双线性上采样+卷积）
    NeRVBlock        标准 NeRV 上采样块（CustomConv -> Norm -> Act）
    Conv_Up_Block    “升级版”首阶段块（conv -> PixelShuffle -> conv，E-NeRV 论文 3.4 节，省约 3/4 参数）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import ActivationLayer, NormLayer, PositionalEncoding


def NeRV_MLP(dim_list, act='relu', bias=True):
    """按维度列表构建多层感知机：Linear(dim_list[i], dim_list[i+1]) + 激活函数交替堆叠。

    例：dim_list=[160, 512, 256] -> Linear(160,512) + act + Linear(512,256) + act
    E-NeRV 中用作时间支路 stem_t、空间支路 stem_xy、投影层 toconv、temporal IN 支路 t_branch/t_layers。
    """
    act_fn = ActivationLayer(act)  # 由字符串配置选择激活函数（默认配置为 gelu）
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class NeRV_CustomConv(nn.Module):
    """可切换类型的“卷积 + 上采样”组合，按 conv_type 三选一：
        'conv'     3x3 卷积扩张到 new_ngf*stride*stride 通道，再 PixelShuffle 重排上采样（默认，权重最省）
        'deconv'   步长为 stride 的转置卷积（反卷积）
        'bilinear' 双线性插值上采样后接 (2*stride+1)x(2*stride+1) 卷积
    """
    def __init__(self, **kargs):
        super(NeRV_CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']  # 输入通道 / 输出通道 / 上采样倍数
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])  # 通道扩张 stride^2 倍
            self.up_scale = nn.PixelShuffle(stride)  # 把 stride^2 个通道重排为空间分辨率放大 stride 倍
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)  # 转置卷积直接上采样
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)  # 双线性插值
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])  # 插值后卷积融合

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)


class NeRVBlock(nn.Module):
    """标准 NeRV 上采样块：CustomConv(上采样卷积) -> Norm -> Act。"""
    def __init__(self, **kargs):
        super().__init__()

        self.conv = NeRV_CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'], 
            conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])  # 归一化层（默认配置 norm=none，即 Identity）
        self.act = ActivationLayer(kargs['act'])                # 激活层（默认 gelu）

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv_Up_Block(nn.Module):
    """升级版上采样块（E-NeRV 论文 3.4 节）。

    原始 NeRVBlock 在大上采样倍数 s 时参数量为 C1*C2*s^2*3*3（s=5 时爆炸，首块可占模型 65%）。
    升级版重构为：
        conv(C1, C0*s*s) -> PixelShuffle(s) -> conv(C0, C2)，其中 C0 = min(C1,C2)/4
    参数量约降为原来的 1/4 以下，性能几乎无损。
    E-NeRV 仅在第一阶段（s=5）使用该块（此时 ngf <= new_ngf，走 factor=4 分支）。
    """
    def __init__(self, **kargs):
        super().__init__()
        ngf = kargs['ngf']      # 输入通道 C1
        new_ngf = kargs['new_ngf']  # 输出通道 C2

        if ngf <= new_ngf:  # 通道扩张的场景（第一阶段 expansion>1）
            factor = 4
            self.conv1 = NeRV_CustomConv(ngf=ngf, new_ngf=ngf // factor, stride=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'])  # conv+PixelShuffle：C1 -> (C1/4) 通道 + 上采样
            self.conv2 = nn.Conv2d(ngf // factor, new_ngf, 3, 1, 1, bias=kargs['bias'])  # 3x3 卷积恢复到目标通道 C2
        else:  # 通道收缩的场景：先 3x3 卷积降通道，再上采样
            self.conv1 = nn.Conv2d(ngf, new_ngf, 3, 1, 1, bias=kargs['bias'])
            self.conv2 = NeRV_CustomConv(ngf=new_ngf, new_ngf=new_ngf, stride=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv2(self.conv1(x))))


class NeRV_Generator(nn.Module):
    """原始 NeRV 生成器（作为 E-NeRV 的对比基线，配置 NeRV-bunny.yaml）。

    数据流（bunny 配置，fc_hw_dim='9_16_112'）：
      帧号 t [B] --pe--> [B,160] --stem MLP--> [B, 9*16*112] --reshape--> [B,112,9,16]
      逐级 NeRVBlock 上采样（stride 5,2,2,2,2 -> 720x1280），末阶段 1x1 卷积 + sigmoid 输出整帧。
    与 E-NeRV 的本质区别：空间信息与时间信息全部耦合在 stem MLP 的输出里
    （最后一层 Linear 需输出约 1.6 万维，占模型大部分参数）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.pe = PositionalEncoding(  # 帧号位置编码 gamma(t)，b=1.25, l=80 -> 维度 160
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        stem_dim, stem_num = [int(x) for x in cfg['stem_dim_num'].split('_')]  # stem 隐藏层宽度与层数，如 '512_1' -> (512, 1)
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in cfg['fc_hw_dim'].split('_')]  # 初始特征图高/宽/通道
        mlp_dim_list = [self.pe.embed_length] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]  # [160, 512, 9*16*112]
        self.stem = NeRV_MLP(dim_list=mlp_dim_list, act=cfg['act'])  # 由帧号直接回归整帧初始特征图的 MLP
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]  # 解码块列表 / 输出头列表
        ngf = self.fc_dim
        for i, stride in enumerate(cfg['stride_list']):  # [5,2,2,2,2]
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * cfg['expansion'])  # 第一阶段通道扩张
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])  # 后续阶段通道减半且不低于 lower_width

            for j in range(cfg['num_blocks']):  # 每个阶段 num_blocks 个块，仅第 0 个块做上采样，其余 stride=1
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if cfg['sin_res']:
                if i == len(cfg['stride_list']) - 1:  # 单分辨率模式：仅最后一个阶段接输出头
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias']) 
                    # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias']) 
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])  # 多分辨率模式：每阶段都接输出头
                # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias']) 
            self.head_layers.append(head_layer)
        self.sigmoid = cfg['sigmoid']

    def forward(self, data):
        """输入 data['img_id']（归一化帧号），输出多尺度图像列表（sin_res=true 时仅最后一项）。"""
        input_id = data['img_id']  # [B]
        input_emb = self.pe(input_id) # [B, L]  位置编码
        output = self.stem(input_emb)  # [B, fc_h*fc_w*fc_dim]  MLP 直接回归初始特征图
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)  # 展平向量重排为特征图 [B, C, h, w]

        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output)   # 上采样卷积块
            if head_layer is not None:
                img_out = head_layer(output)  # 1x1 卷积输出头
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5  # 压缩到 [0,1]
                out_list.append(img_out)

        return  out_list
