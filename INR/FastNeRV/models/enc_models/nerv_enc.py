import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.transformers.vit_model import ViT_models
from utils import NormLayer, split_str, init_wb, DiagonalGaussianDistribution

@register('nerv_enc')
class NeRVEnc(nn.Module):
    """
    NeRV-Enc 模型（论文 Sec. 3.1 的核心实现）：
    用一个 Transformer 超网络 g_phi 直接从输入视频生成 NeRV 的"视频特定权重" theta_hat'，
    从而免去逐视频梯度下降拟合（传统 NeRV 的"编码"过程），实现约 10^4 倍的编码加速。

    整体前向流程 forward() = encode() + decode()：
      1. encode()：视频分块 -> patch tokens，与初始权重 tokens（wtokens，即论文中的 theta_0）
         一起送入 Transformer 超网络，取出最后的权重 token，经 quant_z_fc 得到瓶颈表示 z；
      2. decode()：z 经 dequant_z_fc 与每层的 wtoken_postfc 生成每层调制权重 theta_hat'；
      3. forward_hyponet()：theta_hat' 与视频无关的共享权重 base_params（theta_1）逐元素相乘并
         L2 归一化，得到最终 NeRV 权重 theta'，再由 hyponet（分组卷积）一次性并行解码出所有帧。

    其中 "layer-adaptive weight tokens"（层自适应权重 token）是本文相对 TransINR/GINR 的改进点：
    每层可配置不同的 token 数量与维度（token_nums / token_dims），用更小的视频表示体积取得更好效果。
    """
    def __init__(self, tokenizer, hyponet, nerv_tokens, transformer_encoder, kl_layer):
        super().__init__()
        # Build transformer backbone and video tokenizer
        # 构建 Transformer 超网络骨干（ViT-B/L/...），并按其隐层维度构建视频分块器 tokenizer
        self.transformer_encoder = self.create_transformer(transformer_encoder)
        hid_dim = self.transformer_encoder.hid_dim
        self.tokenizer = models.make(tokenizer, args={'dim': hid_dim})

        # build bottleneck layers
        # 构建瓶颈层（论文中的视频特定权重 theta_hat' 的紧凑表示）：
        # z_type 决定瓶颈形式：'nerv'/'patch'/'extra' 为低维 z（quant_z_fc 输出 z_dim*2，
        # 对角高斯的均值/方差，dequant_z_fc 再升维回来）；'prefc' 只做线性变换不降维；'no' 为恒等映射
        self.z_dim = nerv_tokens['z_dim']
        self.z_type = nerv_tokens['z_type'].lower()
        assert self.z_type in ['patch', 'nerv', 'extra', 'no', 'prefc']
        if self.z_type in ['no', 'prefc']:
            self.quant_z_fc = nn.Linear(hid_dim, hid_dim*2) if self.z_type=='prefc' else nn.Identity()
            self.dequant_z_fc = nn.Identity()
        else:
            self.quant_z_fc = nn.Linear(hid_dim, self.z_dim*2)
            self.dequant_z_fc = nn.Linear(self.z_dim, hid_dim)

        # build hypo_net to generate NeRV weights
        # 构建 NeRV 子网络 hyponet（解码用的 NeRV 模型 f_theta）。它本身无可训练权重，
        # 其权重由 base_params（视频无关权重 theta_1）与超网络输出的调制权重相乘得到
        self.hyponet = models.make(hyponet)
        # base_params：论文中的视频无关（video-agnostic）共享权重 theta_1，
        # 形状由 hyponet.param_shapes 给出，用卷积默认初始化（init_wb：Kaiming 均匀分布 + bias）
        self.base_params = nn.ParameterDict()
        # wtoken_postfc：每层一个 LayerNorm+Linear，把超网络输出的权重 token 映射为该层调制权重
        self.wtoken_postfc = nn.ModuleDict()
        self.repeat_dim = nerv_tokens['repeat_dim'] if 'repeat_dim' in nerv_tokens else -2
        self.wtoken_rng, self.postfc_dim_list = dict(), dict()
        self.ordered_hyponet_param_names = list(self.hyponet.param_shapes.keys())
        token_num_list, token_dim_list = [split_str(nerv_tokens[x]) for x in ['token_nums', 'token_dims']]
        self.moment_shape = dict()
        n_wtokens = 0
        self.num_modulate_w = {'modulate_w': 0, 'modulate_w_prefc': 0, 'modulate_w_postfc': 0 }
        # 遍历 NeRV 的每一层（wb0/wb1/...），为每层分配：
        #   base_params[name]   —— 共享基础权重 theta_1（最后一行是 bias）
        #   wtoken_postfc[name] —— token 后处理全连接层
        #   wtoken_rng[name]    —— 该层 tokens 在整条 wtokens 中的下标区间 [l, r)
        # token_num/token_dim 为 0 表示该层不使用调制（直接用 theta_1）
        for layer_i, name in enumerate(self.ordered_hyponet_param_names):
            base_w = init_wb(*self.hyponet.param_shapes[name])       # model-agnostic weights: out_ch * in_ch * ks * ks
            self.base_params[name] = nn.Parameter(base_w)
            token_num, token_dim = token_num_list[layer_i], token_dim_list[layer_i]
            if token_num and token_dim:
                self.wtoken_postfc[name] = nn.Sequential(
                    nn.LayerNorm(hid_dim),
                    nn.Linear(hid_dim, token_dim),   
                )
                self.wtoken_rng[name] = (n_wtokens, n_wtokens + token_num)
                self.num_modulate_w['modulate_w'] += token_num * min(hid_dim, token_dim)
                self.num_modulate_w['modulate_w_prefc']  += token_num * hid_dim
                self.num_modulate_w['modulate_w_postfc'] += token_num * token_dim     
                n_wtokens += token_num
                self.moment_shape[name] = (token_num , token_dim)
        # 初始权重 tokens theta_0：所有层共用的一条可学习 token 序列，训练时与超网络一同更新
        self.wtokens = nn.Parameter(torch.empty(n_wtokens, hid_dim))
        self.token_init_std = nerv_tokens['token_init_std']
        self.token_num_list, self.token_dim_list = token_num_list, token_dim_list

        # collect bottleneck info for logging
        # 统计瓶颈（视频特定权重）总大小，仅用于日志与 wandb 记录
        z_token_num = n_wtokens
        if self.z_type in ['prefc', 'no']:
            self.z_dim = hid_dim
        z_size = z_token_num * self.z_dim 
        self.z_token_num = z_token_num
        self.num_modulate_w['bottleneck_w'] = z_size
        self.bottleneck_str = f'{self.z_type}: (token_num: {z_token_num}, token_dim: {self.z_dim}) at {round(z_size/1e3, 1)}k'
        # z 的归一化层（BN/LN 等），z_norm_first 决定归一化在高斯采样之前还是之后
        self.z_norm_first = nerv_tokens['z_norm_first']
        self.z_norm_layer = NormLayer(nerv_tokens['z_norm'], self.z_dim*(2 if self.z_norm_first else 1), z_token_num)
        self.frame_num = tokenizer['args']['frame_num']

        # add kl layer
        # KL 层：deterministic=False 时把 z 建模为对角高斯（VAE 风格），以 KL 散度做正则
        self.deterministic = kl_layer['deterministic']
        self.kl_weight = float(kl_layer['kl_weight'])
        self.avg_kl = kl_layer['avg_kl']
        self.initialize_weights()


    def initialize_weights(self):
        """权重初始化：Linear 层 Xavier 均匀初始化、bias 置 0；初始权重 tokens 用正态分布初始化。"""
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize wtokens
        nn.init.normal_(self.wtokens, std=self.token_init_std)


    def create_transformer(self, transformer_encoder):
        """根据配置名（如 'ViT-B'）创建 Transformer 超网络骨干。"""
        model = ViT_models[transformer_encoder['name']](
            proj_drop = transformer_encoder['proj_drop'],
            attn_drop = transformer_encoder['attn_drop'],
            mlp_drop = transformer_encoder['mlp_drop'],
        )      
        return model  


    def forward(self, data, **kwargs):
        # 完整前向：先 encode（超网络生成 z），再 decode（生成 NeRV 权重并解码出所有帧）
        # import pdb; pdb.set_trace; from IPython import embed; embed()
        z_info = self.encode(data, **kwargs)
        out_dict = self.decode(z_info['z'])
        out_frames_dict = {'pred': out_dict['pred_frames']}

        return {**out_dict, **z_info, 'out_frames': out_frames_dict, 
                 'gt_frames': data}


    def encode(self, data, **kwargs):
        """视频"编码"（论文 Fig.2 上半部分）：视频 -> patch tokens -> Transformer 超网络 -> 权重 token -> 每视频潜在表示 z。"""
        # Video tokenization and go through transformer encoder
        dtokens = self.tokenizer(data)                                    # 视频分块得到的 patch tokens (B, M, D)
        wtokens = self.repeat_input_tokens(self.wtokens, dtokens.size(0)) # 初始权重 tokens theta_0 复制到 batch 维
        trans_out = self.transformer_encoder([dtokens, wtokens], **kwargs)  # 拼接后过 Transformer（patch 与权重 token 交互）
        z_tokens = self.quant_z_fc(trans_out[:, -len(self.wtokens):])    # 取序列末尾的权重 token 做瓶颈量化投影

        if self.z_type in ['no']:
            # 无瓶颈：超网络输出直接作为 z
            return {'z': z_tokens}
            #  z = z_tokens
        else:
            # 有瓶颈：可选先归一化 -> 对角高斯采样（或确定性取均值）-> 可选后归一化，同时返回 KL 正则项
            if self.z_norm_first:
                z_tokens = self.z_norm_layer(z_tokens)
            z_posterior = DiagonalGaussianDistribution(z_tokens, self.deterministic)
            z = z_posterior.sample()      
            base_kl = z_posterior.kl(avg_kl=self.avg_kl)
            if not self.z_norm_first:
                z = self.z_norm_layer(z)
            return {'z': z, 'kl_loss': base_kl*self.kl_weight, 'base_kl': base_kl}
        # import pdb; pdb.set_trace; from IPython import embed; embed() 


    def decode(self, z):
        """由潜在表示 z 生成各层调制权重 theta_hat'（论文 Fig.4 右），再调用 forward_hyponet 解码出视频帧。"""
        z = self.dequant_z_fc(z)                    # 恢复到超网络隐层维度
        params_dict = dict()
        for name, (l,r) in self.wtoken_rng.items():
            modulate_w = self.wtoken_postfc[name](z[:, l: r, :])        # (B, token_num, total_fc_out_dim*2), where total_fc_out_dim = sum(in_ch*ks*ks*token_dim_list[i])
            params_dict[name] = modulate_w          # 该层的视频特定（调制）权重 theta_hat'
        pred_frames = self.forward_hyponet(params_dict, z.size(0))

        return {'params': params_dict, 'pred_frames': pred_frames}


    def forward_hyponet(self, modulate_params, batch_size):
        """
        生成最终 NeRV 权重并前向解码（论文 Eq.3 与附录 0.B）：
        theta' = L2_normalize( theta_1 * repeat(theta_hat') )，随后交给 hyponet 用分组卷积并行解码所有帧。
        附录指出三个关键点：a) 必须做 L2 归一化（否则 PSNR 掉约 3dB）；b) theta_1 必须按卷积默认方式
        初始化；c) theta_hat' 的重复扩展维度选 -2 时效果最好（均对应本函数内的实现）。
        """
        hyponet_params = {}
        for name in self.ordered_hyponet_param_names:
            wb = self.repeat_input_tokens(self.base_params[name], batch_size)
            # einops.repeat(self.base_params[name], 'n m -> b n m', b=batch_size)
            init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]       # (B, in_ch*ks*ks, out_ch), (B, 1, out_ch)
            if name in modulate_params:
                # 把低维调制权重 theta_hat' 沿 repeat_dim 维重复扩展到与 theta_1 相同的形状
                w = modulate_params[name]
                repeat_num = init_w.nelement() // w.nelement()
                assert repeat_num >= 1
                if self.repeat_dim == -1:
                    w = einops.repeat(w, 'B n m -> B n m d', d=repeat_num)
                elif self.repeat_dim == -2:
                    w = einops.repeat(w, 'B n m -> B n d m', d=repeat_num)
                elif self.repeat_dim == -3:
                    w = einops.repeat(w, 'B n m -> B d n m', d=repeat_num)
                else:
                    NotImplementedError(f'Unsupported repeat_dim {self.repeat_dim}')
                # import pdb; pdb.set_trace; from IPython import embed; embed()
                final_w = w.reshape_as(init_w) * init_w   # 逐元素相乘：theta' = theta_1 * theta_hat'
            else:
                final_w = init_w
            final_w = F.normalize(final_w, dim=1)         # L2 归一化（论文附录 0.B 的关键结论之一）
            hyponet_params[name] = torch.cat([final_w, init_b], dim=1)

        pred = self.hyponet(hyponet_params, batch_size)     # T B C H W

        return pred


    def repeat_input_tokens(self, in_tokens, batch_size):
        # repeat batch_size at the begining of dimension
        """把 (N, D) 的 token 张量在第 0 维复制出 batch 份，得到 (B, N, D)。"""
        return einops.repeat(in_tokens, 'n d -> b n d', b=batch_size)


    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True):
        """从 checkpoint（路径或 dict）重建 NeRVEnc；load_state_dict=True 时同时载入权重。"""
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f'checkpoint {ckpt} does not exist'
            ckpt = torch.load(ckpt, map_location= lambda storage, loc: storage)
        else:
            assert isinstance(ckpt, dict), f'checkpoint must be a dict or a path to a checkpoint'
        model = cls(**ckpt['model']['args'])
        if load_state_dict:
            model.load_state_dict(ckpt['model']['sd'])
        return model



    # deprecated
