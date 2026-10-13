# -*- coding: utf-8 -*-
"""E-NeRV 生成器（论文核心模型）。

论文: E-NeRV: Expedite Neural Video Representation with Disentangled
      Spatial-Temporal Context (ECCV 2022, arXiv:2207.08132)

核心思想：把 NeRV 中“由帧号直接解码整帧”的耦合式建模解耦为两条支路——
    f_t = F_theta( F(gamma(t)) ⊙ S )
    1) 时间支路 F(gamma(t))：帧号 t 的位置编码 gamma(t) 经 MLP 得到紧凑的时间向量 [B, block_dim]；
    2) 空间支路 S：可学习的 xy 坐标网格经位置编码 + 单头自注意力 Transformer(Phi) 得到空间上下文 [B, h*w, block_dim]；
    3) 二者逐元素相乘(⊙)后，经多头 Transformer F_theta 融合，投影回卷积解码器所需的 [B, fc_dim, fc_h, fc_w] 特征图；
    4) 解码器每个阶段先做 InstanceNorm，再用 fuse_t()（FiLM/时间实例归一化，temporal IN）把时间调制参数
       (gamma, beta) 注入特征，最后经上采样卷积块逐级放大到目标分辨率。

本文件还包含自用的 Transformer 基础组件：PreNorm / FeedForward / Attention / TransformerBlock。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from .model_utils import ActivationLayer, NormLayer, PositionalEncoding, gradient
from .NeRV import NeRV_MLP, NeRVBlock, Conv_Up_Block
from einops import rearrange

class PreNorm(nn.Module):
    """Pre-LayerNorm 包装器：先归一化再送入子层 fn（Pre-LN 结构，训练更稳定）"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 对特征最后一维做 LayerNorm
        self.fn = fn                   # 被包装的子层（Attention 或 FeedForward）
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """Transformer 的前馈网络 (FFN)：Linear -> GELU -> Dropout -> Linear -> Dropout"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """多头自注意力 (Multi-Head Self-Attention)。

    输入 x: [B, N, dim]（N 为 token 数，这里 token 是空间网格位置）
    输出:   [B, N, dim]
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head          # 多头拼接后的总维度
        project_out = not(heads==1 and dim_head==dim)  # 若输入输出维度一致且单头则可省去输出投影

        self.heads = heads
        self.scale = dim_head ** -0.5         # 注意力分数缩放因子 1/sqrt(d_k)

        self.attend = nn.Softmax(dim = -1)    # 对最后一维（key 维）做 softmax
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 一次性生成 Q、K、V

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 切分为 q/k/v，各 [B, N, inner_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # 分头: [B, heads, N, dim_head]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 注意力分数 QK^T/sqrt(d_k): [B, heads, N, N]
        attn = self.attend(dots)                                   # softmax 归一化
        out = torch.matmul(attn, v)                                # 加权求和 V: [B, heads, N, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')               # 各头拼接回 [B, N, inner_dim]
        return self.to_out(out)                                    # 输出投影回 [B, N, dim]

class TransformerBlock(nn.Module):
    """标准 Transformer 块：自注意力 + FFN，各带残差连接。

    论文中的两个实例：
      trans1 -> 单头 (heads=1)，充当空间上下文变换 Phi；
      trans2 -> 8 头，充当时空融合 Transformer F_theta 的注意力部分。
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:  # Pre-LN 变体：子层前先做 LayerNorm
            self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:        # Post-LN 变体（本仓库默认配置使用 prenorm=False，即不加 LayerNorm）
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x = self.attn(x) + x  # 注意力残差连接
        x = self.ffn(x) + x   # FFN 残差连接
        return x


class E_NeRV_Generator(nn.Module):
    """E-NeRV 模型主体（生成器）。

    数据流（以 bunny 配置为例，fc_h=9, fc_w=16, block_dim=256, fc_dim=196）：
      帧号 t --pe_t--> [B,160] --stem_t--> t_emb [B,256]                （时间上下文）
      t     --pe_t_manipulate--> [B,160] --t_branch--> t_manipulate [B,128]（供解码器 temporal IN 使用）
      xy 网格 [2,h*w] --pe_xy--> [h*w,320] --stem_xy--> [h*w,256]        （空间上下文）
                --trans1(单头自注意力 Phi)--> [B,h*w,256]
      逐元素乘 emb = xy_emb ⊙ broadcast(t_emb)                          （时空解耦融合 ⊙）
                --trans2(8头 F_theta)--> --toconv--> [B,h*w,196]
      reshape -> [B,196,9,16]，随后 5 个解码阶段（stride 5,2,2,2,2 -> 720x1280）：
        每阶段 = InstanceNorm -> fuse_t(FiLM 调制) -> 上采样卷积块 -> (末阶段)1x1卷积+sigmoid 输出图像
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # yaml 中 model 段的配置字典

        # t mapping
        self.pe_t = PositionalEncoding(  # 帧号 t 的位置编码 gamma(t)，b=1.25, l=80 -> 维度 2*80=160
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        stem_dim_list = [int(x) for x in cfg['stem_dim_num'].split('_')]  # 时间支路 MLP 的隐藏层宽度列表，如 '512' -> [512]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in cfg['fc_hw_dim'].split('_')]  # 初始特征图高/宽/通道，如 '9_16_196'
        self.block_dim = cfg['block_dim']  # 时空融合阶段的特征维度（论文中的 d）

        mlp_dim_list = [self.pe_t.embed_length] + stem_dim_list + [self.block_dim]  # [160, 512, 256]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, act=cfg['act'])  # 时间支路 MLP F：[B,160] -> [B,256]

        # xy mapping
        xy_coord = torch.stack(   # 归一化到 [0,1) 的 xy 网格坐标，形状 [2, h*w]（2 个通道分别是 x 与 y 坐标）
            torch.meshgrid(
                torch.arange(self.fc_h) / self.fc_h, torch.arange(self.fc_w) / self.fc_w
            ), dim=0
        ).flatten(1, 2)  # [2, h*w]
        self.xy_coord = nn.Parameter(xy_coord, requires_grad=False)  # 固定不参与训练的坐标常量
        self.pe_xy = PositionalEncoding(  # xy 坐标的位置编码（与时间支路同结构），输出维度 2*80=160
            pe_embed_b=cfg['xypos_b'], pe_embed_l=cfg['xypos_l']
        )
        
        self.stem_xy = NeRV_MLP(dim_list=[2 * self.pe_xy.embed_length, self.block_dim], act=cfg['act'])  # 空间支路 MLP：x/y 编码拼接 [h*w,320] -> [h*w,256]
        self.trans1 = TransformerBlock(  # 论文中的 Phi：单头自注意力，把空间先验编码进网络权重，输出仍为 [.,h*w,256]
            dim=self.block_dim, heads=1, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        self.trans2 = TransformerBlock(  # 论文中的 F_theta 注意力部分：8 头自注意力，融合相乘后的时空特征
            dim=self.block_dim, heads=8, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        if self.block_dim == self.fc_dim:  # 融合维度与解码器输入通道一致时无需投影
            self.toconv = nn.Identity()
        else:
            self.toconv = NeRV_MLP(dim_list=[self.block_dim, self.fc_dim], act=cfg['act'])  # [256] -> [196]
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers, self.t_layers, self.norm_layers = [nn.ModuleList() for _ in range(4)]  # 解码块/输出头/时间调制支路/IN 层
        ngf = self.fc_dim  # 当前通道宽度，从 fc_dim=196 开始
        for i, stride in enumerate(cfg['stride_list']):  # stride_list=[5,2,2,2,2]，9x16 -> 45x80 -> ... -> 720x1280
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * cfg['expansion'])  # 第一阶段通道扩张（expansion=3 -> 588）
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])  # 后续阶段通道减半，但不低于 lower_width
            
            self.t_layers.append(NeRV_MLP(dim_list=[128, 2*ngf], act=cfg['act']))  # 论文中的 M_i：把 t_manipulate[128] 映射为当前阶段的 (gamma,beta)，共 2*ngf 维
            self.norm_layers.append(nn.InstanceNorm2d(ngf, affine=False))  # 逐通道 InstanceNorm（仿射参数由 fuse_t 提供，故 affine=False）
            
            if i == 0:
                self.layers.append(Conv_Up_Block(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))  # 第一阶段用“升级版”上采样块（论文 3.4 节，大幅省参数）
            else:
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))  # 其余阶段用标准 NeRV 块
            ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if cfg['sin_res']:
                if i == len(cfg['stride_list']) - 1:  # 单分辨率模式：只在最后一个阶段接输出头
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])  # 多分辨率模式：每个阶段都接输出头（多尺度监督）
            self.head_layers.append(head_layer)
        self.sigmoid = cfg['sigmoid']  # 输出激活：True 用 sigmoid，False 用 (tanh+1)/2

        self.T_num = 20  # 常量（推理/插值实验相关，前向中未使用）
        self.pe_t_manipulate = PositionalEncoding(pe_embed_b=cfg['pos_b_tm'], pe_embed_l=cfg['pos_l_tm'])  # temporal IN 支路专用位置编码（论文中降低其频率 b 可改善未见帧插值）
        self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, 128, 128], act=cfg['act'])  # 论文 3.3 节的微型 MLP：[B,160] -> [B,128]，供各阶段 t_layers 生成调制参数

        self.loss = cfg['additional_loss'] if cfg.__contains__('additional_loss') else None  # 是否启用辅助损失（默认 false 关闭）
        self.loss_w = cfg['additional_loss_weight'] if cfg.__contains__('additional_loss_weight') else 1.0  # 辅助损失权重
        self.mse = nn.MSELoss()  # 预留的 MSE 损失（默认配置未用到）
    
    def fuse_t(self, x, t):
        """时间条件调制（论文 3.3 节，FiLM / temporal instance normalization）。

        把时间特征 t 拆成逐通道的缩放 gamma 与偏移 beta，对 (InstanceNorm 后的) 特征图做仿射变换：
            out = x * gamma + beta
        等价于 AdaIN：IN(f) = sigma * (f - mu(f))/sigma(f) + mu，其中 (sigma, mu) 由帧号决定。
        """
        # x: [B, C, H, W], normalized among C
        # t: [B, 2* C]
        f_dim = t.shape[-1] // 2
        gamma = t[:, :f_dim]   # 前一半通道作为缩放系数 [B, C]
        beta = t[:, f_dim:]    # 后一半通道作为偏移量 [B, C]

        gamma = gamma[..., None, None]  # [B, C, 1, 1]，便于广播到空间维度
        beta = beta[..., None, None]
        out = x * gamma + beta
        return out

    def forward_impl(self, input_id):
        """前向计算核心：帧号 -> 多尺度图像列表。"""
        t = input_id  # 归一化帧号 [B]（首帧 0，末帧 1）

        t_emb = self.stem_t(self.pe_t(t)) # [B, L]  时间上下文：gamma(t) -> MLP -> [B, 256]
        t_manipulate = self.t_branch(self.pe_t_manipulate(t))  # [B, 128] 供解码器各阶段 temporal IN 使用

        xy_coord = self.xy_coord
        x_coord = self.pe_xy(xy_coord[0])    # [h*w, C]  x 坐标位置编码 [h*w, 160]
        y_coord = self.pe_xy(xy_coord[1])    # [h*w, C]  y 坐标位置编码 [h*w, 160]
        xy_emb = torch.cat([x_coord, y_coord], dim=1)  # [h*w, 320] 拼接
        xy_emb = self.stem_xy(xy_emb).unsqueeze(0).expand(t_emb.shape[0], -1, -1)  # [B, h*w, L] 空间上下文 MLP 后按 batch 复制

        xy_emb = self.trans1(xy_emb)  # 空间自注意力 Phi：建模像素间空间关系（权重编码了该视频的空间结构）
        # fuse t into xy map
        t_emb_list = [t_emb for i in range(xy_emb.shape[1])]
        t_emb_map = torch.stack(t_emb_list, dim=1)  # [B, h*w, L] 把时间向量复制到每个空间位置
        emb = xy_emb * t_emb_map  # 论文核心：时空特征逐元素相乘（解耦表示的融合 ⊙）
        emb = self.toconv(self.trans2(emb))  # 8 头 Transformer F_theta 融合，再投影到解码器通道数 [B, h*w, fc_dim]

        emb = emb.reshape(emb.shape[0], self.fc_h, self.fc_w, emb.shape[-1])  # [B, fc_h, fc_w, fc_dim]
        emb = emb.permute(0, 3, 1, 2)  # [B, fc_dim, fc_h, fc_w] 转成卷积要求的 NCHW
        output = emb

        out_list = []
        for layer, head_layer, t_layer, norm_layer in zip(self.layers, self.head_layers, self.t_layers, self.norm_layers):
            # t_manipulate
            output = norm_layer(output)          # InstanceNorm：对每个通道做空间维度的标准化
            t_feat = t_layer(t_manipulate)       # M_i：为当前阶段生成 (gamma, beta) [B, 2*ngf]
            output = self.fuse_t(output, t_feat) # FiLM 调制：注入帧号相关的分布偏移（temporal IN）
            # conv
            output = layer(output)               # 上采样卷积块（Conv_Up_Block / NeRVBlock）
            if head_layer is not None:
                img_out = head_layer(output)     # 1x1 卷积输出头
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5  # 压缩到 [0,1]
                out_list.append(img_out)

        return  out_list  # 各尺度输出列表（sin_res=true 时仅最后一个，720x1280）
    
    def forward(self, data):
        """模型对外接口。data 为 DataLoader 返回的字典，含 'img_id'（帧号）。"""
        input_id = data['img_id']  # [B]
        batch_size = input_id.shape[0]

        output_list = self.forward_impl(input_id)  # a list containing [B or 2B, 3, H, W]

        if self.loss and self.training:  # 辅助损失分支（默认配置 additional_loss=false，不启用）
            b, c, h, w = output_list[-1].shape
            # NO USE
            grad_loss = 0.0  # 占位：原意是梯度差损失，当前实现未使用
            return {
                "loss": grad_loss * self.loss_w,
                "output_list": output_list,
            }
        
        return output_list
