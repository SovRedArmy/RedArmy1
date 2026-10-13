# 基准测试: DCVC-RT P 帧模型一次训练迭代(前向+反向)的耗时。
# 论文未公开训练代码, 这里按 DCVC 家族通用训练设置模拟一个 P 帧训练 step:
#   batch_size=8, patch=256x256, 前向走 compress 等价路径, 反向取一个简单
#   MSE 风格损失 (loss 的具体形式只影响极小的反向量, 不影响量级)。
# 输出每次迭代秒数, 供训练总时长推算。
import sys
import time

import torch

sys.path.insert(0, '.')
from src.models.video_model import DMC

device = 'cuda'
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

B, S = 8, 256          # batch, patch size
N_FRAMES = 7           # Vimeo-90k 七帧序列
WARMUP, ITERS = 3, 10

net = DMC().to(device)
net.train()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
x0 = torch.rand(B, 3, S, S, device=device)

def step_amp(use_amp):
    """一个 P 帧训练 step: 主帧带梯度前向+反向。

    DPB 先用随机参考特征播种(等价于 I 帧后的初始特征), 之后每步把本步
    解码特征 detach 后存回 DPB 作下一步参考(与训练时 reference 处理一致)。"""
    qp = torch.randint(0, 64, (1,)).item()
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        feature = net.apply_feature_adaptor()
        ctx, ctx_t = net.feature_extractor(feature, net.q_feature[qp:qp+1])
        y = net.encoder(x0, ctx, net.q_encoder[qp:qp+1])
        hyper_inp = net.pad_for_y(y)
        z = net.hyper_encoder(hyper_inp)
        # 训练时用加噪模拟量化; 计时上与 round 等价
        z_hat = z + torch.randn_like(z) * 0.5
        params = net.res_prior_param_decoder(z_hat, ctx_t)
        q_dec, scales, means = params.chunk(3, 1)
        q_dec = torch.clamp_min(q_dec, 0.5)
        y_hat = y * torch.reciprocal(q_dec) + means
        f_dec = net.decoder(y_hat, ctx, net.q_decoder[qp:qp+1])
        x_hat = net.recon_generation_net(f_dec, net.q_recon[qp:qp+1])
        loss = torch.mean((x_hat - x0) ** 2)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    net.add_ref_frame(f_dec.detach(), None)
    return loss.item()

def seed_dpb():
    net.clear_dpb()
    net.add_ref_frame(torch.rand(B, 256, S // 8, S // 8, device=device), None)

for use_amp in (False, True):
    seed_dpb()
    tag = 'fp16-autocast' if use_amp else 'fp32'
    for i in range(WARMUP):
        step_amp(use_amp)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(ITERS):
        step_amp(use_amp)
    torch.cuda.synchronize()
    per_iter = (time.time() - t0) / ITERS
    print(f'[{tag}] per-P-frame-step (batch={B}, {S}x{S}): {per_iter*1000:.1f} ms | '
          f'7-frame iter: {per_iter*N_FRAMES*1000:.0f} ms')
print('done')
