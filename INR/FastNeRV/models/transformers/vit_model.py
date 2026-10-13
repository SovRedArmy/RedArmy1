# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import einops
import torch.nn.functional as F

from timm.models.vision_transformer import Attention, Mlp

class AttentionFull(Attention):
    """标准多头自注意力（基于 timm 的 Attention，改用 F.scaled_dot_product_attention 实现）。
    在 NeRV-Enc 中它同时作用于视频 patch tokens 与权重 tokens，让视频信息写入权重 token。"""
    def forward(self, x, attn_mask=None):
        assert self.fused_attn == True, 'fused_attn not supported'
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#                                 Core ViT Model                                #
#################################################################################

class ViTBlock(nn.Module):
    """
    A ViT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    标准 Pre-Norm Transformer 块：LN + 自注意力 + 残差，LN + FFN + 残差。
    输入允许是 token 列表（如 [patch_tokens, wtokens]），forward 内先沿序列维拼接。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, proj_drop=0, attn_drop=0, mlp_drop=0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionFull(hidden_size, num_heads=num_heads, qkv_bias=True, proj_drop=proj_drop,attn_drop=attn_drop, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_drop)

    def forward(self, x, attn_mask=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)       # LN + Self-attention
        x = x + self.mlp(self.norm2(x))     # LN + FFN
        return x


class ViT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    Transformer 超网络骨干（论文中的 g_phi）：由 depth 个 ViTBlock 堆叠而成。
    NeRV-Enc（nerv_enc）走 z_token_list is None 的分支：把 [视频 patch tokens, 权重 tokens]
    拼接后整体过各层，末尾取出权重 tokens 生成视频特定权重。
    T-NeRV（tnerv_enc/param_enc）走另一分支：z_tokens 逐层注入（sequential/展开解码，用于对比实验）。
    """
    def __init__( 
        self,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        proj_drop=0.,
        attn_drop=0.,
        mlp_drop=0.,
    ):
        super().__init__()
        self.hid_dim = hidden_size
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, proj_drop=proj_drop, 
                attn_drop=attn_drop, mlp_drop=mlp_drop) for _ in range(depth)
        ])
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


    def forward(self, x, z_token_list=None, sequential_decode=False, cumsum_z=False):
        """
        Forward pass of ViT.
        x: (N, T, D)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        分支 1（NeRV-Enc 用）：z_token_list 为空 —— 直接把输入列表拼接后过所有 block；
        分支 2（T-NeRV/ParamEnc 用）：z_token_list 非空 —— 把 z_tokens 按层拆分，
        前若干个 block 逐个注入一段 z_tokens（sequential_decode 决定注入方式），
        并对各中间层输出做 cumsum 或拼接后返回（用于多层渐进解码的对比实验）。
        """
        if z_token_list == None or sum(z_token_list)==0:
            if isinstance(x, list):
                x = torch.cat(x, dim=1)
            for block in self.blocks:
                x = block(x)                      # (N, T, D)
            return x
        else:
            assert len(x) == 3, f'input should have 3 parts: patch_tokens, z_tokens, d_tokens'
            patch_tokens, z_tokens, d_tokens = x
            x = torch.cat([patch_tokens, d_tokens], dim=1)
            z_tokens_list = z_tokens.split(z_token_list, dim=1)
            def get_valid_levels(z_token_list):
                valid_levels = []
                for z_idx, z_tokens in enumerate(z_token_list):
                    if z_tokens != 0:
                        valid_levels.append(z_idx)
                    else:
                        valid_levels[-1] = z_idx
                return valid_levels
            valid_levels = get_valid_levels(z_token_list)
            # import pdb; pdb.set_trace; from IPython import embed; embed()  
            f_list = []
            for block_id, block in enumerate(self.blocks):
                if block_id < len(z_tokens_list):
                    if sequential_decode:
                        x = [x, z_tokens_list[block_id]]
                    else:
                        x = [patch_tokens, d_tokens] + z_tokens_list[:block_id+1]
                x = block(x)                      # (N, T, D)
                if block_id < len(z_token_list):
                    f_list.append(x.clone())
                elif block_id == len(z_token_list):
                    f_list = [f_list[i][:,:patch_tokens.size(1)] for i in valid_levels]
                    if cumsum_z:
                        def deterministic_cumsum(f_list):
                            stacked_tensors = torch.stack(f_list, dim=0)  # This will have shape [n, b, c, l] where n is the number of tensors in f_list
                            cumsum_tensors = torch.zeros_like(stacked_tensors)
                            cumsum_tensors[0] = stacked_tensors[0]
                            for i in range(1, len(f_list)):
                                cumsum_tensors[i] = cumsum_tensors[i - 1] + stacked_tensors[i]

                            return cumsum_tensors
                        x = deterministic_cumsum(f_list).flatten(end_dim=1)
                    else:
                        x = torch.cat(f_list, 0)
            return x
        

#################################################################################
#                                   ViT Configs                                  #
#################################################################################
# 不同规模的 ViT 超网络预设（depth / hidden_size / num_heads）；配置中通过
# transformer_encoder.name: 'ViT-B' 等选择，对应论文中 47.6M ~ 251M 的超网络规模。
def ViT_H(**kwargs):
    return ViT(depth=32, hidden_size=1280, num_heads=16, **kwargs)

def ViT_XL(**kwargs):
    return ViT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def ViT_L(**kwargs):
    return ViT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def ViT_B(**kwargs):
    return ViT(depth=12, hidden_size=768, num_heads=12, **kwargs)

# [复现补充] 论文附录 0.A.5 的默认超网络：6 层 encoder、hidden 720、FFN 2800
# （mlp_ratio=4 -> 2880，与 2800 基本一致），对应约 47.6M 的 NeRV-Enc。
def ViT_720(**kwargs):
    return ViT(depth=6, hidden_size=720, num_heads=12, **kwargs)

def ViT_S(**kwargs):
    return ViT(depth=12, hidden_size=384, num_heads=6, **kwargs)

def ViT_T(**kwargs):
    return ViT(depth=6, hidden_size=384, num_heads=6, **kwargs)

# ViT_models
ViT_models = {'ViT-H': ViT_H, 'ViT-T': ViT_T,
    'ViT-XL': ViT_XL, 'ViT-L':  ViT_L,
    'ViT-B':  ViT_B,  'ViT-S':  ViT_S,
    'ViT-720': ViT_720,
    }
