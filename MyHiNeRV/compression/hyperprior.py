"""
DCVC-RT 风格的迷你权重超先验（hyperprior）—— 无需外部数据版。

DCVC-RT 用数百万视频预训练的分析/合成网络来逐元素估计 μ 和 σ，从而把 y 的熵编码
变成对 (y − μ) 的零均值高斯编码（σ 小 → 0 附近极便宜）。我们没有外部数据与算力，
但 INR 视频压缩本身就是逐视频过拟合：让一个极小的超先验网络与 INR 一起在当前
模型的权重上过拟合训练（COOL-CHIC 对图像熵模型的思路），即可获得同样结构的
逐元素 μ/σ 预测：

    残差权重 r（= θ − θ_ref）的 2D 视图 [N, C]
        --g_a: adaptive_avg_pool 下采样--> z --round(z/q_z)--> z_hat   （少量侧信息）
        --g_s: z_hat --MLP(层嵌入+位置)--> 低分辨率 (μ_z, σ_z) --双线性上采样--> 全分辨率 μ, σ
    熵编码：s = q − round(μ)（q = round(r/q_step)），逐元素高斯 N(0, σ) 的 CDF 行
    做算术编码（σ 逐元素不同 → 每行 CDF 不同）；z_hat 用每层标量高斯编码；
    超网络参数（几 KB）量化存入码流头。

解码端：加载（half 固化的）超网络参数 → 解码 z_hat → g_s 在 CPU 上重建与编码端
完全一致的 μ/σ → 解码 s → θ̂ = (s + round(μ))·q_step + θ_ref。
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import unwrap_model
from .quant_utils import _iter_quant_parametrizations, QuantEntropy, gaussian_cdf

# 逐元素 CDF 行的分块大小（控制 torchac 的内存峰值）
_CHUNK = 4 * 1024 * 1024
_DOWN = 8  # z 相对 r 的下采样倍率


class WeightHyperprior(nn.Module):
    """
    所有量化目标张量共享的迷你超先验网络：
    - mlp：输入 [z_hat 值, 行/列归一化坐标, 层嵌入] → 输出 (μ_z, σ_z)（低分辨率）；
    - embeddings / q_z / q_steps：每张量一个层嵌入、z 步长与权重步长（可学习）。
    MLP 只在 z 分辨率上运行（元素数 ~ numel/64，训练代价可忽略），
    上采样到全分辨率用无参数双线性插值。
    """
    def __init__(self, embed_dim=16, hidden=64, down=_DOWN):
        super().__init__()
        self.down = down
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.embeddings = nn.ParameterDict()
        self.q_z = nn.ParameterDict()
        self.q_steps = nn.ParameterDict()
        self.tensor_keys = []
        self.tensor_shapes = {}
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _safe_key(key: str) -> str:
        return key.replace('.', '_')

    def register_tensor(self, key: str, shape):
        """为一个量化目标张量注册层嵌入与步长（构建时调用一次）。"""
        skey = self._safe_key(key)
        if skey not in self.embeddings:
            self.embeddings[skey] = nn.Parameter(torch.randn(self.embed_dim) * 0.1)
            self.q_z[skey] = nn.Parameter(torch.tensor(1e-3))
            self.q_steps[skey] = nn.Parameter(torch.tensor(1e-3))
            self.tensor_keys.append(skey)
            self.tensor_shapes[skey] = tuple(shape)

    def _to_2d(self, r: torch.Tensor):
        """任意形状张量 → 2D 视图 [N, C]（最后一维作为列）。"""
        if r.ndim == 1:
            return r.reshape(1, -1)
        return r.reshape(-1, r.shape[-1])

    @staticmethod
    def _z_shape(n, c, down):
        return (max(1, n // down), max(1, c // down))

    def analyze(self, key, r: torch.Tensor):
        """g_a：残差 r → 连续 z（下采样池化）。"""
        r2 = self._to_2d(r).float()
        zn, zc = self._z_shape(r2.shape[0], r2.shape[1], self.down)
        return F.adaptive_avg_pool2d(r2[None], (zn, zc))[0]

    def synthesize(self, key, z_hat: torch.Tensor, out_shape):
        """
        g_s：z_hat → 与 out_shape 同形状的 (μ, σ)。全部在 CPU float32 上计算
        （保证编码端与解码端逐位一致）。
        """
        skey = self._safe_key(key)
        z_hat = z_hat.float().cpu()
        zn, zc = z_hat.shape
        py = torch.linspace(0., 1., zn)[:, None].expand(zn, zc)
        px = torch.linspace(0., 1., zc)[None, :].expand(zn, zc)
        emb = self.embeddings[skey].detach().float().cpu()[None, None, :].expand(zn, zc, self.embed_dim)
        feat = torch.cat([z_hat[:, :, None], py[:, :, None], px[:, :, None], emb], dim=-1)
        out = self.mlp.cpu()(feat)  # [zn, zc, 2]
        mu_z, sig_z = out[..., 0], F.softplus(out[..., 1]).clamp_min(1e-3)

        n2 = int(np.prod(out_shape)) // out_shape[-1] if len(out_shape) > 1 else 1
        c2 = out_shape[-1] if len(out_shape) > 1 else int(np.prod(out_shape))
        mu = F.interpolate(mu_z[None, None], size=(n2, c2), mode='bilinear', align_corners=False)[0, 0]
        sig = F.interpolate(sig_z[None, None], size=(n2, c2), mode='bilinear', align_corners=False)[0, 0]
        return mu.reshape(out_shape).clone(), sig.reshape(out_shape).clamp_min(1e-3).clone()

    def quantize_z(self, key, z: torch.Tensor):
        """z 量化（STE）：z_hat = round(z / q_z)。"""
        skey = self._safe_key(key)
        q_z = torch.clamp_min(self.q_z[skey], 1e-3)
        return (z / q_z).round()

    def rate_bits(self, key, r: torch.Tensor, q: torch.Tensor):
        """
        训练期的率估计（比特数，可微）：
        s = q − μ 的离散高斯概率（σ 逐元素，来自 g_s(z_hat)），
        z 的侧信息按每元素固定比特估计（z 只有 ~numel/64 个元素）。
        """
        z = self.analyze(key, r)
        z_hat = self.quantize_z(key, z)
        # g_s 的可微版本（在 z 的设备上、用带梯度的参数；与 synthesize 结构一致）
        skey = self._safe_key(key)
        zn, zc = z_hat.shape
        py = torch.linspace(0., 1., zn, device=z.device, dtype=z.dtype)[:, None].expand(zn, zc)
        px = torch.linspace(0., 1., zc, device=z.device, dtype=z.dtype)[None, :].expand(zn, zc)
        emb = self.embeddings[skey][None, None, :].expand(zn, zc, self.embed_dim)
        feat = torch.cat([z_hat[:, :, None], py[:, :, None], px[:, :, None], emb], dim=-1)
        out = self.mlp(feat)
        mu_z, sig_z = out[..., 0], F.softplus(out[..., 1]).clamp_min(1e-3)
        n2, c2 = self._to_2d(r).shape
        mu = F.interpolate(mu_z[None, None], size=(n2, c2), mode='bilinear', align_corners=False)[0, 0].reshape(r.shape)
        sig = F.interpolate(sig_z[None, None], size=(n2, c2), mode='bilinear', align_corners=False)[0, 0].reshape(r.shape).clamp_min(1e-3)

        s_soft = q - mu  # q 为 STE 量化值（保持对权重的梯度）
        prob = gaussian_cdf(s_soft + 0.5, 0., sig) - gaussian_cdf(s_soft - 0.5, 0., sig)
        rate_r = -torch.log2(prob.clamp_min(2 ** -32)).sum()
        rate_z = torch.tensor(float(z_hat.numel()) * 6., device=z.device)
        return rate_r + rate_z

    @torch.no_grad()
    def freeze_transmitted(self):
        """
        把超网络参数按"传输精度"（half → float）固化：
        编码端从此用与解码端完全相同的参数值计算 μ/σ，保证熵编解码一致。
        """
        for d in (self.embeddings, self.q_z, self.q_steps):
            for k in d:
                d[k].copy_(d[k].detach().half().float())
        for p in self.mlp.parameters():
            p.copy_(p.detach().half().float())

    def transmitted_state(self):
        """导出（half 精度的）超网络参数，写入码流头。"""
        state = {}
        for name, p in self.named_parameters():
            state[name] = p.detach().half().cpu().clone()
        return state

    def load_transmitted_state(self, state):
        """从码流头恢复超网络参数（half → float）。"""
        with torch.no_grad():
            for name, p in self.named_parameters():
                p.copy_(state[name].float())


def attach_hyperprior(args, logger, model):
    """
    为模型构建并挂载 WeightHyperprior（在 initial_parametrizations 之后、
    accelerator.prepare 之前调用，使其参数进入优化器）。
    仅 quant_mode='hyper' 时生效；模块挂在 model.hyper（不属于 bitstream_prefix，
    其参数作为侧信息写入码流头）。
    """
    model = unwrap_model(model)
    if getattr(args, 'quant_mode', 'noise') != 'hyper':
        return None
    hyper = WeightHyperprior(embed_dim=getattr(args, 'hyper_embed_dim', 16),
                             hidden=getattr(args, 'hyper_hidden', 64),
                             down=getattr(args, 'hyper_down', _DOWN))
    n = 0
    for owner_path, weight_name, p in _iter_quant_parametrizations(model):
        if isinstance(p, QuantEntropy):
            key = f'{owner_path}.{weight_name}'
            w_shape = tuple(model.get_submodule(owner_path).parametrizations.weight.original.shape)
            hyper.register_tensor(key, w_shape)
            n += 1
    model.hyper = hyper
    if logger is not None:
        n_params = sum(pp.numel() for pp in hyper.parameters())
        logger.info(f'Attached WeightHyperprior: {n} tensors, {n_params} side-info params.')
    return hyper


def compute_hyper_rate_bpp(model, num_pixels: float):
    """
    训练期的率估计（bpp）：遍历所有已启动的 QuantEntropy 层，用超先验估计比特数
    （权重 q 取与失真路径一致的 STE 量化值）。
    """
    model = unwrap_model(model)
    hyper = getattr(model, 'hyper', None)
    if hyper is None:
        return None
    bits = None
    for owner_path, weight_name, p in _iter_quant_parametrizations(model):
        if not isinstance(p, QuantEntropy) or not bool(p.start):
            continue
        key = f'{owner_path}.{weight_name}'
        r = p.get_residual()
        if r is None:
            continue
        q_step = torch.clamp_min(hyper.q_steps[hyper._safe_key(key)], 1e-3)
        q = (r / q_step).round()
        b = hyper.rate_bits(key, r, q)
        bits = b if bits is None else bits + b
    if bits is None:
        return None
    return bits / float(num_pixels)


# ============================ 实际的熵编解码（离线，一次） ============================

def _torchac():
    import torchac
    return torchac


@torch.no_grad()
def hyper_compress_tensor(s_int: torch.Tensor, sigma: torch.Tensor, output_dir, name,
                          chunk=_CHUNK):
    """
    对整数残差 s = q − round(μ) 做逐元素高斯熵编码：
    每个元素的 CDF 行由其 σ 决定（N(0, σ_i)），符号表 [-R, R]。
    分块编码以控制 CDF 表内存。返回 meta（R 与每块字节数）。
    """
    assert s_int.dtype == torch.int32
    R = int(s_int.abs().max().item()) if s_int.numel() > 0 else 0
    if R == 0:
        return {'R': 0, 'chunks': [], 'shape': tuple(s_int.shape)}
    sym_all = (s_int.reshape(-1) + R).to(torch.int16).cpu()
    sig_all = sigma.reshape(-1).float().clamp_min(1e-3)
    k = torch.arange(-R, R + 1, dtype=torch.float32)

    def cdf_rows(sig):
        cdf = gaussian_cdf(k[None, :] - 0.5, 0., sig[:, None])
        cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
        return torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=1).clamp_(max=1.)

    n = sym_all.numel()
    bits_all = bytearray()
    chunk_sizes = []
    for st in range(0, n, chunk):
        sl = slice(st, min(st + chunk, n))
        bits = _torchac().encode_float_cdf(cdf_rows(sig_all[sl]), sym_all[sl],
                                           check_input_bounds=True, needs_normalization=True)
        bits_all.extend(bits)
        chunk_sizes.append(len(bits))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, name + '.bit'), 'wb') as f:
        f.write(bytes(bits_all))
    return {'R': R, 'chunks': chunk_sizes, 'shape': tuple(s_int.shape)}


@torch.no_grad()
def hyper_decompress_tensor(meta, sigma: torch.Tensor, output_dir, name,
                            chunk=_CHUNK):
    """hyper_compress_tensor 的逆操作（σ 由解码端 g_s 重建，与编码端一致）。"""
    R = int(meta['R'])
    shape = tuple(meta['shape'])
    numel = int(np.prod(shape)) if len(shape) else 0
    if R == 0 or numel == 0:
        return torch.zeros(list(shape), dtype=torch.int32)
    sig_all = sigma.reshape(-1).float().clamp_min(1e-3)
    k = torch.arange(-R, R + 1, dtype=torch.float32)

    def cdf_rows(sig):
        cdf = gaussian_cdf(k[None, :] - 0.5, 0., sig[:, None])
        cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
        return torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=1).clamp_(max=1.)

    with open(os.path.join(output_dir, name + '.bit'), 'rb') as f:
        byte_stream = f.read()
    out = []
    pos = 0
    st = 0
    for cs in meta['chunks']:
        m = min(chunk, numel - st)
        sig = sig_all[st:st + m]
        sym = _torchac().decode_float_cdf(cdf_rows(sig), byte_stream[pos:pos + cs]).int()
        out.append(sym - R)
        pos += cs
        st += m
    return torch.cat(out).view(list(shape)).to(torch.int32)
