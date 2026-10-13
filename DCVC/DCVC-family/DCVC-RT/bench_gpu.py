# 量化 GPU 共享折损 + 确定性算法开销:
# 1) fp16 matmul 实测 TFLOPS vs A6000 规格(155.2 TFLOPS dense fp16)
# 2) set_torch_env 的确定性设置对 compress 耗时的影响
import sys
import time

import torch

sys.path.insert(0, '.')

print('==== matmul fp16 (GPU 共享折损测量) ====')
for tag, setup in (('as-is', lambda: None),
                   ('deterministic(set_torch_env)', None)):
    if setup:
        from src.utils.common import set_torch_env
        set_torch_env()
    torch.cuda.synchronize()
    a = torch.rand(8192, 8192, device='cuda').half()
    b = torch.rand(8192, 8192, device='cuda').half()
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    tflops = 2 * 8192**3 / (ms / 1000) / 1e12
    print(f'[{tag}] {ms:.2f} ms/matmul -> {tflops:.1f} TFLOPS '
          f'(A6000 峰值 155.2 -> 可用率 {tflops/155.2*100:.0f}%)')

print('==== compress 全流程(含确定性设置的影响) ====')
from src.models.video_model import DMC
net = DMC().to('cuda').eval()
net.update(None)
net.half()
net.set_use_two_entropy_coders(True)
H, W = 1088, 1920
x = (torch.rand(1, 3, H, W, device='cuda') * 0.5 + 0.25).half()
net.add_ref_frame(torch.rand(1, 256, H // 8, W // 8, device='cuda').half(), None)
for _ in range(3):
    net.compress(x, 21)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    net.compress(x, 21)
torch.cuda.synchronize()
print(f'compress (deterministic on): {(time.time()-t0)/20*1000:.2f} ms/frame')

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True
for _ in range(3):
    net.compress(x, 21)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    net.compress(x, 21)
torch.cuda.synchronize()
print(f'compress (benchmark  on)   : {(time.time()-t0)/20*1000:.2f} ms/frame')
