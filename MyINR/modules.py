# ==========================================================================================
# MyINR: 在 NeRV 基础上的改进实现 —— 模型结构
#
# NeRVGenerator 在原版 NeRV (Generator) 基础上增加:
#   1. CRSPE 风格的每帧内容嵌入 (learnable per-frame spatial code, 注入 MLP 输出特征图)
#   2. 多尺度输出头 (每个 stage 都接 1x1 conv head, 供 max-pool 金字塔多尺度监督)
#   3. 辅助光流头 + 置信度头 (1/4 分辨率, 仅训练期使用, 不计入码率)
# 与原版 NeRV 的层命名 (stem / layers / head_layers) 保持一致, 便于加载官方 checkpoint。
# ==========================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from nerv_blocks import ActivationLayer, NormLayer, CustomConv, MLP, NeRVBlock  # noqa: F401


class NeRVGenerator(nn.Module):
    """NeRV++ 生成网络: 位置编码向量 + 每帧内容嵌入 → MLP → 逐级上采样卷积 → 多尺度 RGB。

    参数相对原版新增:
        n_frames       : 视频总帧数(clip 内), 用于为每帧分配内容嵌入
        crspe_channels : 每帧内容嵌入通道数 Ce; 0 = 关闭(CRSPE), 退化为原版 NeRV
        multi_scale    : True=每个 stage 接 head(多尺度监督); False=仅最后一个 stage(原版 single_res)
        flow_head      : 是否构建辅助光流头与置信度头(预测相邻帧间 1/4 分辨率光流)
    """

    def __init__(self, embed_length, stem_dim_num='512_1', fc_hw_dim='9_16_26', expansion=1,
                 reduction=2, lower_width=96, num_blocks=1, stride_list=(5, 3, 2, 2, 2),
                 norm='none', act='swish', conv_type='conv', sigmoid=False,
                 n_frames=0, crspe_channels=0, multi_scale=True, residual_mode=False, **kargs):
        super().__init__()
        stem_dim, stem_num = [int(x) for x in stem_dim_num.split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in fc_hw_dim.split('_')]
        self.n_frames = n_frames
        self.crspe_channels = crspe_channels

        mlp_dim_list = [embed_length] + [stem_dim] * stem_num + [self.fc_h * self.fc_w * self.fc_dim]
        self.stem = MLP(dim_list=mlp_dim_list, act=act)

        # ---- CRSPE: 每帧内容嵌入 (N, Ce, fc_h, fc_w), 经 1x1 conv 投影后与 MLP 输出特征融合 ----
        if crspe_channels > 0:
            assert n_frames > 0
            self.frame_embed = nn.Parameter(0.01 * torch.randn(n_frames, crspe_channels, self.fc_h, self.fc_w))
            self.embed_proj = nn.Conv2d(self.fc_dim + crspe_channels, self.fc_dim, 1, 1, 0, bias=True)

        # ---- 逐级上采样卷积主体 (与原版 NeRV 完全一致) ----
        self.layers = nn.ModuleList()
        self.head_layers = nn.ModuleList()
        self.stage_strides = []           # 记录每个 stage 的累计上采样倍率(用于多尺度分辨率)
        self.stage_channels = []          # 每个 stage 输出的通道数
        ngf = self.fc_dim
        up = 1
        for i, stride in enumerate(stride_list):
            if i == 0:
                new_ngf = int(ngf * expansion)
            else:
                new_ngf = max(ngf // (1 if stride == 1 else reduction), lower_width)
            for j in range(num_blocks):
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                                             bias=True, norm=norm, act=act, conv_type=conv_type))
                ngf = new_ngf
            up *= stride
            self.stage_strides.append(up)
            self.stage_channels.append(ngf)
            if multi_scale or i == len(stride_list) - 1:
                self.head_layers.append(nn.Conv2d(ngf, 3, 1, 1, bias=True))
            else:
                self.head_layers.append(nn.Identity())

        # v2: 不再构建辅助光流头/置信度头(显式光流估计模块已按需求移除)
        self.flow_stage_idx = None
        self.flow_head = None
        self.conf_head = None

        self.sigmoid = sigmoid
        self.residual_mode = residual_mode  # True=不做输出激活(输出原始残差,可为负)

    def fuse_embed(self, feat, frame_ids):
        """CRSPE 融合: feat (B,fc_dim,h,w) ⊕ 每帧内容嵌入 → 1x1 conv 投影回 fc_dim 通道."""
        if self.crspe_channels <= 0:
            return feat
        e = self.frame_embed[frame_ids]                     # (B, Ce, h, w)
        return self.embed_proj(torch.cat([feat, e], dim=1))

    def forward(self, embed_input, frame_ids, return_feats=False):
        """
        embed_input: (B, embed_length) 位置编码后的帧索引向量
        frame_ids  : (B,) long, clip 内帧编号(用于索引内容嵌入)
        返回 out_list: 各 head 输出的 (B,3,H,W) 列表(Identity head 输出特征跳过)
        """
        output = self.stem(embed_input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)
        output = self.fuse_embed(output, frame_ids)

        out_list, feat_list = [], []
        for i, (layer, head_layer) in enumerate(zip(self.layers, self.head_layers)):
            output = layer(output)
            feat_list.append(output)
            if not isinstance(head_layer, nn.Identity):
                img_out = head_layer(output)
                if self.residual_mode:
                    pass  # 残差模式: 不做激活, 输出原始值(可为负), 由外层 clamp
                else:
                    img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)
        if return_feats:
            return out_list, feat_list
        return out_list

# ---- 加载原版 NeRV checkpoint 的工具: 原版 multi-res 模式 head 挂在每个 stage ----
def load_nerv_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    # 原版 single_res: 只有最后一个 stage 有 Conv2d head, 其余位置没有模块; 我们的 multi_scale
    # 版本每个 stage 都有 head —— 逐一匹配参数名
    own = model.state_dict()
    new_sd = {}
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            new_sd[k] = v
    missing = [k for k in own if k not in new_sd]
    print(f'loaded {len(new_sd)}/{len(own)} tensors; missing: {missing[:6]}...')
    model.load_state_dict(new_sd, strict=False)
    return model


# ==========================================================================================
# V3: 宽底座 → 轻量分支 的确定性权重投影(解码端可由底座码流复算, 分支差分因此自包含)
# 同族架构(fc_dim/lower_width 按整数倍缩小)逐张量做"通道组平均";
# PixelShuffle 卷积的输出通道按 (new_ngf, s^2) 两级布局分组, 保持 PS 语义。
# ==========================================================================================
import torch.nn.functional as _F


def _pool_dim0(w, ratio, ps_stride=1):
    """对张量第 0 维按 ratio 分组平均; ps_stride>1 时按 (new_ngf, s^2) 布局分组保持 PixelShuffle 语义。"""
    if ratio <= 1 or w.dim() < 1:
        return w
    if ps_stride > 1:
        s2 = ps_stride * ps_stride
        if w.shape[0] % s2:
            return w
        new_ngf = w.shape[0] // s2
        ngf_small = new_ngf // ratio
        if ngf_small < 1 or new_ngf % ratio:
            return w
        rest = list(w.shape[1:])
        w2 = w.view(new_ngf, s2, *rest)
        w2 = w2.view(ngf_small, ratio, s2, *rest).mean(dim=1)
        return w2.reshape(ngf_small * s2, *rest)
    out = w.shape[0]
    out_small = out // ratio
    if out_small < 1 or out % ratio:
        return w
    rest = list(w.shape[1:])
    return w.view(out_small, ratio, *rest).mean(dim=1).reshape(out_small, *rest)


def _pool_dim1(w, ratio):
    """对张量第 1 维(输入通道)按 ratio 分组平均 —— 用于 1x1 head 卷积 (3, ngf, 1, 1)。"""
    if ratio <= 1 or w.dim() < 2:
        return w
    c = w.shape[1]
    c_small = c // ratio
    if c_small < 1 or c % ratio:
        return w
    shape = list(w.shape)
    w2 = w.view(shape[0], c_small, ratio, *shape[2:])
    return w2.mean(dim=2)


def _redim(t, out_shape):
    """把张量 t 的每一维自适应平均到 out_shape(确定可复算; 目标维更大时为重复)。"""
    if tuple(t.shape) == tuple(out_shape):
        return t.detach().clone()
    v = t.detach().float()
    for d in range(v.dim()):
        si, oi = int(v.shape[d]), int(out_shape[d])
        if si == oi:
            continue
        front, rest = list(v.shape[:d]), list(v.shape[d + 1:])
        m = v.reshape(-1, si)                                   # [其余维乘积, si] — 池化目标维放末尾
        m = torch.nn.functional.adaptive_avg_pool1d(m, oi)      # adaptive_avg_pool1d 只作用最后一维
        v = m.reshape(front + [oi] + rest)
    return v.to(t.dtype)


def project_state_to(base_sd, template_shapes):
    """由底座 state_dict 投影出分支初始权重, 目标形状取自分支模板(template_shapes: key->shape)。
    适配 V3 固定分支(256_1/fc8/lw24): 底座无论 256/384/512, 均逐层自适应平均到分支形状。
    frame_embed 保留底座原表(由 trainer 按帧数重采样), 不参与投影。"""
    ps = {}
    for k, want in template_shapes.items():
        if k not in base_sd:
            continue
        v = base_sd[k]
        if not torch.is_tensor(v) or k == 'frame_embed':
            ps[k] = v
            continue
        try:
            ps[k] = _redim(v, tuple(want))
        except Exception:
            ps[k] = v
    return ps


def project_state(base_sd, ratio, strides):
    """由底座 state_dict 投影出分支初始权重(确定性, 解码端可复算)。
    分支宽度 = 底座宽度 / ratio(fc_dim 与 lower_width 同比缩小, stem 隐层不变)。"""
    ps = {}
    for k, v in base_sd.items():
        if not torch.is_tensor(v):
            ps[k] = v
            continue
        # 逐层分派
        stride = 1
        if k.startswith('layers.'):
            si = int(k.split('.')[1])
            stride = strides[si]
        if k.startswith('stem.0.'):
            ps[k] = v                                   # Linear(embed, stem_dim) 两版相同, 不动
        elif k.startswith('stem.'):
            ps[k] = _pool_dim0(v.reshape(-1, *([1] * (v.dim() - 1))), ratio, 1).reshape(-1) \
                if v.dim() == 1 else _pool_dim0(v, ratio, 1)             # stem.2: 输出 144*C 按 fc 池化
        elif k.startswith('head_layers.') and k.endswith('.weight') and v.dim() == 4:
            ps[k] = _pool_dim1(v, ratio)                # 1x1 head: 池化输入通道
        elif '.conv.conv.' in k and k.endswith('.weight'):
            ps[k] = _pool_dim1(_pool_dim0(v, ratio, stride), ratio)   # PS 卷积: 输出(ngf,s^2)池化 + 输入池化
        elif '.conv.conv.' in k and k.endswith('.bias'):
            ps[k] = _pool_dim0(v.view(-1, 1), ratio, stride).view(-1)
        else:
            ps[k] = v                                   # head bias / 嵌入表等
    return ps


# ==========================================================================================
# V3 QAT: STE 权重量化参数化(UAR-NVC QuantEntropy 的精简实现)
# 注册为 Conv2d/Linear 的 weight 参数化 → 模型前向自动用量化后权重
# 梯度通过 STE 同时流向原始权重和 q_step:
#   d(out)/d(w)      = 1        (恒等, STE 直通)
#   d(out)/d(q_step) = round_err (量化误差越大梯度越强 → 推 q_step 向下)
# 与率项(推 q_step 向上)对抗 → 达到真实 R-D 平衡点
# ==========================================================================================
import torch.nn as nn


class STEQuant(nn.Module):
    """单个权重张量的 STE 量化参数化。
    q_step 用 softplus 参数化保证恒正(避免 clamp 的零梯度盲区):
        实际步长 = softplus(raw) + min_step, 梯度处处非零。"""
    def __init__(self, q_step_init=0.001):
        super().__init__()
        # softplus(x) > 0 恒成立; softplus⁻¹(0.001) ≈ ln(e^0.001 - 1) ≈ -6.9
        import math
        self.q_step_raw = nn.Parameter(torch.tensor(
            math.log(math.expm1(q_step_init)), dtype=torch.float32))
        self.min_step = 1e-4
        self.start = False

    @property
    def q_step(self):
        return torch.nn.functional.softplus(self.q_step_raw) + self.min_step

    def forward(self, w):
        if not self.start:
            return w
        qs = self.q_step  # 恒正, 无需 clamp
        w_scaled = w / qs
        w_q = (w_scaled.round() - w_scaled).detach() + w_scaled   # STE
        return w_q * qs


def register_qat(model):
    """为模型的所有 Conv2d/Linear 权重注册 STE 量化参数化。"""
    import torch.nn.utils.parametrize as P
    n = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            P.register_parametrization(module, 'weight', STEQuant())
            n += 1
    return n


def set_qat_start(model, start=True):
    """激活/关闭量化前向。"""
    for _, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in getattr(module, 'parametrizations', {}):
            for p in module.parametrizations.weight:
                if isinstance(p, STEQuant):
                    p.start = start


def remove_qat(model):
    """移除参数化, 恢复标准 state_dict 键名。返回 (原始权重 dict, q_step dict)。"""
    import torch.nn.utils.parametrize as P
    orig_w, q_steps = {}, {}
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in getattr(module, 'parametrizations', {}):
            key = name + '.weight'
            orig_w[key] = module.parametrizations.weight.original.detach().cpu().clone()
            for p in module.parametrizations.weight:
                if isinstance(p, STEQuant):
                    q_steps[key.replace('.', '_')] = p.q_step.detach().cpu().clone()
            P.remove_parametrizations(module, 'weight', leave_parametrized=False)
    return orig_w, q_steps


# ==========================================================================================
# V4 LoRA 低秩差分: 分支只训练每层 ΔW = U·V (rank r), 原权重冻结为投影初始值
#   - V 零初始化 → 初始 ΔW = 0 → 分支初始输出与(投影+head清零)语义完全一致
#   - 码流只编码 (U,V) 小因子(密集对称量化+熵编码), 解码端 W=底座投影(可复算) + U·V 精确重建
#   - 相比稀疏阈值化: 无"硬零+恢复不足"的 0.3~0.5dB 损失, 因子数 ~rank·(m+n) 远小于 0.25·m·n
# ==========================================================================================
class LoRADelta(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        out_f = weight.shape[0]
        in_f = weight.numel() // out_f
        self.rank = rank
        self.shape = tuple(weight.shape)
        dev = weight.device          # 注册时模型可能已在 GPU, 参数须同设备
        self.U = nn.Parameter(torch.randn(out_f, rank, dtype=weight.dtype, device=dev) / (rank ** 0.5))
        self.V = nn.Parameter(torch.zeros(rank, in_f, dtype=weight.dtype, device=dev))

    def forward(self, w):
        return w + (self.U @ self.V).reshape(self.shape)


def register_lora(model, rank):
    """为所有 Conv2d/Linear 注册 LoRA 差分并冻结原权重(须在加载初始权重之后调用)。返回层数。"""
    import torch.nn.utils.parametrize as P
    n = 0
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            P.register_parametrization(module, 'weight', LoRADelta(module.weight, rank))
            module.parametrizations.weight.original.requires_grad_(False)
            n += 1
    return n


def lora_params(model):
    """所有 LoRA U/V 参数(供优化器 add_param_group, ClipTrainer 构造时还没有它们)。"""
    return [p for name, p in model.named_parameters()
            if name.endswith('.U') or name.endswith('.V')]


def export_lora(model):
    """导出因子 dict {'<weight键>.U': U, '<weight键>.V': V} 用于压缩。"""
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in getattr(module, 'parametrizations', {}):
            for p in module.parametrizations.weight:
                if isinstance(p, LoRADelta):
                    out[name + '.weight.U'] = p.U.detach().cpu().clone()
                    out[name + '.weight.V'] = p.V.detach().cpu().clone()
    return out


def load_lora(model, dec_factors):
    """把反量化后的因子写回参数化(压缩后评估用)。"""
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'parametrizations') and 'weight' in getattr(module, 'parametrizations', {}):
                for p in module.parametrizations.weight:
                    if isinstance(p, LoRADelta):
                        ku, kv = name + '.weight.U', name + '.weight.V'
                        if ku in dec_factors:
                            p.U.copy_(dec_factors[ku].to(p.U.device))
                            p.V.copy_(dec_factors[kv].to(p.V.device))


def remove_lora(model):
    """移除 LoRA 参数化, 就地保留 W+ΔW 合成权重(恢复标准键名)。"""
    import torch.nn.utils.parametrize as P
    for _, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in getattr(module, 'parametrizations', {}):
            for p in module.parametrizations.weight:
                if isinstance(p, LoRADelta):
                    P.remove_parametrizations(module, 'weight', leave_parametrized=True)
                    break
