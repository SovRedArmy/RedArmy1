"""
小模型权重推导（derive）：高动态 GOP 小模型的参数从低动态宽（base）模型"计算得来"。

设计要点（区别于 UAR-NVC 的 D-init / 插值 init）：
- 所有小模型都从同一个 base 宽模型推导，彼此之间互不依赖（避免 UAR-NVC 最差变体
  D-init 的顺序依赖与误差累积）；
- 推导是确定性的结构映射：
    * 通道维（宽度）：取前 C_s 个通道（子网络切片，类似结构化宽度剪枝）；
    * 深度维：每 stage 只保留前 D_s 个 block（base 多出的 block 直接丢弃）；
    * 时间维（网格/CRSPE 的 T 轴，clip 长度与全视频不同）：线性插值重采样；
- 推导结果同时充当：a) 小模型初始化（热启动）；b) 残差编码的参考 θ_ref。
  编码端与解码端对 base 的解码重建权重做同样推导 → 小模型只需传输
  θ_s − derive(θ_base_dec) 的量化残差，base 的通用信息"编码一次、全帧共享"。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _interp_time(x: torch.Tensor, t_out: int) -> torch.Tensor:
    """把张量的第 0 维（时间维）线性插值到 t_out。x: [T, ...]。"""
    t_in = x.shape[0]
    if t_in == t_out:
        return x
    rest = x.shape[1:]
    y = x.reshape(t_in, -1).t()[None].float()  # [1, M, T]
    y = F.interpolate(y, size=t_out, mode='linear', align_corners=False)
    return y[0].t().reshape((t_out,) + rest)


def _map_tensor(dst: torch.Tensor, src: torch.Tensor, t_first: bool) -> torch.Tensor:
    """
    把 src 映射到 dst 的形状：时间维（t_first 时）插值，其余维取前 d 个通道/深度。
    """
    if tuple(src.shape) == tuple(dst.shape):
        return src.to(dtype=dst.dtype, device=dst.device)
    out = src.float()
    if t_first and out.shape[0] != dst.shape[0]:
        out = _interp_time(out, dst.shape[0])
    for d in range(out.ndim):
        if out.shape[d] > dst.shape[d]:
            out = out.narrow(d, 0, dst.shape[d])
    assert tuple(out.shape) == tuple(dst.shape), \
        f'derive shape mismatch: src {tuple(src.shape)} -> {tuple(out.shape)} vs dst {tuple(dst.shape)}'
    return out.to(dtype=dst.dtype, device=dst.device)


@torch.no_grad()
def derive_state_dict(small_model: nn.Module, base_state: dict, logger=None) -> dict:
    """
    从 base 宽模型的（规范化键名）state dict 推导小模型的权重。

    键名分三类处理：
    - 'encoding.grids.*'（基础特征网格，2D 存储 [T*H*W, C]）：先还原为 [T, H*W, C]，
      时间维插值 + 通道切片（H/W 网格两级模型一致）；
    - 含 '.enc.grids.*'（时间局部网格，2D 存储 [T, C]）与 'crspe.*'（[T, C]）：
      时间维插值 + 通道切片；
    - 其余（Linear/Conv 权重与 bias）：纯通道/深度切片（前 C_s 个通道）。
    base 中缺失或形状无法映射的键保持小模型自身初始化。
    返回可直接 load_state_dict 的推导权重表。
    """
    derived = {}
    n_loaded, n_skipped = 0, 0
    small_state = small_model.state_dict()

    # 基础编码网格的空间分辨率积（两级模型共享同一空间配置）
    enc_hw = 1
    enc = getattr(small_model, 'encoding', None)
    if enc is not None and hasattr(enc, 'grid_sizes') and len(enc.grid_sizes) > 0:
        _, H, W, _ = enc.grid_sizes[0]
        enc_hw = H * W

    for k, dst in small_state.items():
        src = base_state.get(k, None)
        if src is None:
            n_skipped += 1
            continue
        try:
            if k.startswith('encoding.grids.'):
                src3 = src.float().reshape(-1, enc_hw, src.shape[1])  # [T, H*W, C]
                dst3 = dst.reshape(-1, enc_hw, dst.shape[1])
                out = _map_tensor(dst3, src3, t_first=True).reshape(dst.shape)
            elif '.enc.grids.' in k or k.startswith('crspe'):
                out = _map_tensor(dst, src, t_first=True)
            else:
                out = _map_tensor(dst, src, t_first=False)
        except AssertionError:
            n_skipped += 1
            continue
        derived[k] = out
        n_loaded += 1

    if logger is not None:
        logger.info(f'Derive: mapped {n_loaded} tensors from the base model, skipped {n_skipped} '
                    f'(missing or shape-incompatible).')
    return derived
