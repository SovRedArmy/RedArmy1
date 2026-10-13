# 剖析 DCVC-RT P 帧编码/解码耗时构成: GPU 网络推理 vs CPU rANS 熵编码。
# 只做计时, 不改仓库代码逻辑。
import sys
import time

import torch

sys.path.insert(0, '.')
from src.models.video_model import DMC

dev = 'cuda'
net = DMC().to(dev).eval()
net.update(None)
net.half()
net.set_use_two_entropy_coders(True)

H, W = 1088, 1920   # 1080p padding 后
x = (torch.rand(1, 3, H, W, device=dev) * 0.5 + 0.25).half()
net.add_ref_frame(torch.rand(1, 256, H // 8, W // 8, device=dev).half(), None)

QP = 21
N = 20

# ---- 编码: 整帧 ----
for _ in range(3):
    net.compress(x, QP)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    net.compress(x, QP)
torch.cuda.synchronize()
enc_ms = (time.time() - t0) / N * 1000

# ---- 纯 GPU 部分(跳过熵编码, 复刻 compress 的网络部分) ----
with torch.no_grad():
    for _ in range(3):
        q = net.q_encoder[QP:QP+1]
        feature = net.apply_feature_adaptor()
        ctx, ctx_t = net.feature_extractor(feature, net.q_feature[QP:QP+1])
        y = net.encoder(x, ctx, q)
        z = net.hyper_encoder(net.pad_for_y(y))
        z_hat, _ = __import__('src.layers.cuda_inference', fromlist=['x']).round_and_to_int8(z)
        params = net.res_prior_param_decoder(z_hat, ctx_t)
        out = net.compress_prior_2x(y, params, net.y_spatial_prior)
        f = net.decoder(out[-1], ctx, net.q_decoder[QP:QP+1])
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        feature = net.apply_feature_adaptor()
        ctx, ctx_t = net.feature_extractor(feature, net.q_feature[QP:QP+1])
        y = net.encoder(x, ctx, q)
        z = net.hyper_encoder(net.pad_for_y(y))
        z_hat, _ = __import__('src.layers.cuda_inference', fromlist=['x']).round_and_to_int8(z)
        params = net.res_prior_param_decoder(z_hat, ctx_t)
        out = net.compress_prior_2x(y, params, net.y_spatial_prior)
        f = net.decoder(out[-1], ctx, net.q_decoder[QP:QP+1])
    torch.cuda.synchronize()
    gpu_ms = (time.time() - t0) / N * 1000

# ---- 纯 rANS 编码(用一次真实 compress 产生的符号, 只重复 CPU 编码) ----
# 通过反复调用完整 compress 得到的 enc_ms 与 gpu_ms 之差近似熵编码+调度开销
print(f'encode total : {enc_ms:7.2f} ms/frame')
print(f'network only : {gpu_ms:7.2f} ms/frame (GPU, 无熵编码)')
print(f'entropy+sync : {enc_ms - gpu_ms:7.2f} ms/frame (CPU rANS + 同步)')
print(f'=> 复现帧率 enc = {1000/enc_ms:.1f} fps | 论文A6000 = 70.4 fps | 论文A100 = 125.2 fps')
