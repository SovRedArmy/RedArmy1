import torch


def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    """批处理线性层：在 x 末尾拼接 1 以实现 bias，再与权重矩阵 wb 做批量矩阵乘。"""
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb.to(x))

def batched_conv(x, wb, conv_shape, ps_layer):
    """
    分组卷积实现"B 个 NeRV 并行前向"（论文 Fig.5(b)，NeRV-Dec 的核心算子）：
    把每个视频的卷积权重 wb[b] 展开为一组卷积核，再将 B 组拼接成一个 (B*out_ch, in, k, k)
    的大卷积核，用 groups=B 的 conv2d 一次性计算 B 个视频；最后经 PixelShuffle 上采样。
    输入 x 的 batch 维实际是 (T*B)（T 帧与 B 个视频共享分组），因此 B 个视频需整除。
    """
    B = wb.size(0)
    ch_out, ch_in, cur_ks, pad = conv_shape
    conv_weigts = wb[:,:-1].view((B, ch_in, cur_ks, cur_ks, ch_out))
    conv_weigts = conv_weigts.permute(0,-1,1,2,3).flatten(end_dim=1)
    # import pdb; pdb.set_trace; from IPython import embed; embed()
    conv_bias = wb[:,-1:].flatten()
    x = torch.nn.functional.conv2d(x, conv_weigts, conv_bias, stride=1, padding=pad, groups=B)
    return ps_layer(x)