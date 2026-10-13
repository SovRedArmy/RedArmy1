# 剖析解码耗时: GPU 网络推理 vs CPU rANS 熵解码。
import sys
import time

import torch

sys.path.insert(0, '.')
from src.models.video_model import DMC
from src.layers.cuda_inference import round_and_to_int8

net = DMC().to('cuda').eval()
net.update(None)
net.half()
net.set_use_two_entropy_coders(True)
H, W = 1088, 1920
x = (torch.rand(1, 3, H, W, device='cuda') * 0.5 + 0.25).half()
net.add_ref_frame(torch.rand(1, 256, H // 8, W // 8, device='cuda').half(), None)

QP = 21
sps = {'sps_id': 0, 'height': H, 'width': W, 'ec_part': 1, 'use_ada_i': 0}

# 生成一条真实码流
net.set_curr_poc(0)
enc = net.compress(x, QP)
stream = enc['bit_stream']
print(f'bitstream: {len(stream)} bytes ({len(stream)*8/(H*W):.4f} bpp)')

# ---- 解码整帧 ----
for _ in range(3):
    net.dpb[0].feature = torch.rand(1, 256, H // 8, W // 8, device='cuda').half()
    net.set_curr_poc(1)
    net.decompress(stream, sps, QP)
torch.cuda.synchronize()
t0 = time.time()
N = 20
for _ in range(N):
    net.dpb[0].feature = torch.rand(1, 256, H // 8, W // 8, device='cuda').half()
    net.set_curr_poc(1)
    net.decompress(stream, sps, QP)
torch.cuda.synchronize()
dec_ms = (time.time() - t0) / N * 1000

# ---- 解码的网络部分(跳过熵解码, 用 z_hat 直接走后续) ----
dtype = torch.float16
device = 'cuda'
q_decoder = net.q_decoder[QP:QP+1]
q_feature = net.q_feature[QP:QP+1]
q_recon = net.q_recon[QP:QP+1]
z_size = net.get_downsampled_shape(H, W, 64)
with torch.no_grad():
    for _ in range(3):
        feature = net.apply_feature_adaptor()
        c1, ctx_t = net.feature_extractor.forward_part1(feature, q_feature)
        z_hat = torch.zeros((1, net.z_channel, z_size[0], z_size[1]), device=device, dtype=dtype)
        params = net.res_prior_param_decoder(z_hat, ctx_t)
        ctx = net.feature_extractor.forward_part2(c1)
        y_hat = torch.zeros((1, 128, H // 16, W // 16), device=device, dtype=dtype)
        x_hat, feature = net.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        feature = net.apply_feature_adaptor()
        c1, ctx_t = net.feature_extractor.forward_part1(feature, q_feature)
        z_hat = torch.zeros((1, net.z_channel, z_size[0], z_size[1]), device=device, dtype=dtype)
        params = net.res_prior_param_decoder(z_hat, ctx_t)
        ctx = net.feature_extractor.forward_part2(c1)
        y_hat = torch.zeros((1, 128, H // 16, W // 16), device=device, dtype=dtype)
        x_hat, feature = net.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)
    torch.cuda.synchronize()
    gpu_ms = (time.time() - t0) / N * 1000

# ---- matmul 折损同步测量 ----
a = torch.rand(8192, 8192, device='cuda').half()
b = torch.rand(8192, 8192, device='cuda').half()
for _ in range(3):
    a @ b
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    a @ b
torch.cuda.synchronize()
avail = (2 * 8192**3 / ((time.time() - t0) / 20) / 1e12) / 155.2

print(f'decode total : {dec_ms:7.2f} ms/frame')
print(f'network only : {gpu_ms:7.2f} ms/frame (GPU)')
print(f'entropy+sync : {dec_ms - gpu_ms:7.2f} ms/frame (CPU rANS)')
print(f'GPU 可用率(matmul 同步测量): {avail*100:.0f}%')
print(f'折算空闲 A6000: 网络 {gpu_ms*avail:.1f} ms + 熵 {dec_ms-gpu_ms:.1f} ms '
      f'≈ {gpu_ms*avail + (dec_ms-gpu_ms):.1f} ms/frame = {1000/(gpu_ms*avail+(dec_ms-gpu_ms)):.1f} fps')
print(f'论文 A6000 解码 = 63.8 fps (15.7 ms)')
