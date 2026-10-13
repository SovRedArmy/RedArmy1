import torch
import torch.nn as nn
import numpy as np
import einops
from math import ceil

from models import register
from .layers import batched_conv
from utils import split_str, get_sincos_pos_embed, ActLayer, ClampTensor


@register('nerv_model')
class NeRVModel(nn.Module):
    """
    NeRV 子网络（hyponet）：即论文中的 f_theta，也是 NeRV-Dec 的解码器。

    与普通 NeRV 的区别：本类自身没有任何可训练权重（self.params 仅是占位），
    每层卷积权重由外部（NeRVEnc.forward_hyponet）以分组形式传入（nerv_params），
    通过 layers.batched_conv 的分组卷积实现"一次前向同时解码 B 个视频的所有 T 帧"，
    这正是论文 Fig.5(b) 中 NeRV-Dec 的并行解码方式（B=1 时退化为普通 NeRV 解码）。

    结构：帧索引的时间位置编码 -> [1x1 映射 + PixelShuffle 上采样] x depth 层 -> 输出 RGB 帧。
    每层输出的通道数为 cur_dim * stride^2，恰好被 PixelShuffle 重排为空间分辨率（与 HNeRV 一致）。
    """
    def __init__(self, frame_num, in_dim, out_dim, hid_dim,
        strds, ks, pos_embed, pe_dim, learn_pe, act='relu', out_bias=0.5, out_range='no'):
        super().__init__()
        self.frame_num, self.pos_embed, self.pe_dim = frame_num, pos_embed, pe_dim
        # import pdb; pdb.set_trace; from IPython import embed; embed()
        # 帧时间位置编码（T, D）：可学习（learn_pe=True）或按 NeRV 惯例固定
        self.pe_embed = nn.Parameter(self.get_pos_embed(), requires_grad=learn_pe)      # T D
        strds_list, ks_list = split_str(strds), split_str(ks)
        depth = len(strds_list)
        ks_list += ks_list[-1:] * (depth - len(ks_list))
        self.ps_layers = nn.ModuleList()
        # conv_shape_list 记录每层 (ch_out, ch_in, ks, pad)，param_shapes 记录每层权重形状 (ch_out, ch_in, ks)，
        # 供 NeRVEnc 构建 base_params / wtoken_postfc 使用（键名 wb0/wb1/...）
        self.conv_shape_list, self.param_shapes = [], {}
        last_dim, ch = in_dim*pe_dim, hid_dim
        for i, cur_strd in enumerate(strds_list):
            cur_dim = ch if i < depth - 1 else out_dim     # 最后一层输出通道为 3（RGB）
            cur_ks = ks_list[i]
            ch_out = cur_dim*cur_strd**2                    # PixelShuffle 要求：输出通道 = 目标通道 * stride^2
            self.param_shapes[f'wb{i}'] = (ch_out, last_dim, cur_ks)
            cur_pad = 0 if (cur_ks == cur_strd and i == 0) else ceil(cur_ks - 1) // 2
            self.conv_shape_list.append((ch_out, last_dim, cur_ks, cur_pad))
            self.ps_layers.append(nn.PixelShuffle(cur_strd))
            last_dim = cur_dim
        self.act = ActLayer(act)
        self.out_bias = out_bias
        self.out_range = out_range
        self.depth = depth
        self.params = None


    def get_pos_embed(self, ):
        """按配置生成帧时间位置编码 (T, D)：'sincos' 用固定 sin-cos；'inr_<sigma>' 用指数频率基（NeRV 惯例）。"""
        if 'sincos' in self.pos_embed:
            pos_embed = get_sincos_pos_embed(self.pe_dim, self.frame_num)
        else:
            pe_sigma = float(self.pos_embed.split('_')[-1])
            if 'inr' in self.pos_embed:
                w = torch.exp(torch.linspace(0, np.log(pe_sigma), self.pe_dim // 2))
            else:
                w = pe_sigma ** torch.arange(self.pe_dim // 2)
            pe_range = [0, 1] if 'positive' in self.pos_embed else [-1, 1]
            x = torch.linspace(start=pe_range[0], end=pe_range[1], steps=self.frame_num)
            x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0))
            pos_embed = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)

        return pos_embed


    def OutImg(self, x):
        """输出激活：sigmoid / tanh（映射到 [0,1]）或加常数偏置，并可按 out_range 截断。"""
        if self.out_bias == 'sigmoid':
            out = torch.sigmoid(x)
        elif self.out_bias == 'tanh':
            out = (torch.tanh(x) * 0.5) + 0.5
        else:
            out = x + float(self.out_bias)
        out = ClampTensor(out, self.out_range)

        return out


    def forward(self, nerv_params, batch_size):
        """
        用传入的每层权重 nerv_params（含 bias 行）解码视频：返回 (T, B, C, H, W)。
        输入 (B, in_ch*ks*ks+1, out_ch) 的权重被 batched_conv 展开成分组卷积核，
        因此 B 个视频的 NeRV 可以在同一次 conv2d 中并行前向（NeRV-Dec 的核心）。
        """
        B, T = batch_size, self.frame_num
        if 'embed' in nerv_params:
            # x = self.params['embed']
            NotImplementedError
        else:
            # 把时间嵌入 (T, D) 扩展为 (T, B*D, 1, 1)：所有 B 个视频共享同一时间输入（论文 Fig.5 中的 Repeat）
            x = einops.repeat(self.pe_embed, 't d-> t (b d) h w', b=B, h=1, w=1)

        for i in range(self.depth):
            x = batched_conv(x, nerv_params[f'wb{i}'], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = x.view((T, B, -1) + x.shape[-2:])
        return x

