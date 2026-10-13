# 为速度复现生成随机初始化的权重文件。
# 编码/解码速度与权重的具体数值无关(计算量、kernel 调度、符号数量均固定),
# 因此在拿不到官方 OneDrive 断点时, 可用随机权重复现论文的速度实验。
# 注意: RD 质量(PSNR/bpp)不可用随机权重复现。
import sys
import torch

sys.path.insert(0, '.')
from src.models.video_model import DMC
from src.models.image_model import DMCI

torch.manual_seed(0)
v = DMC()
torch.save(v.state_dict(), 'checkpoints/repro_random_video.pth.tar')
print('video params: %.2fM' % (sum(p.numel() for p in v.parameters()) / 1e6))

i = DMCI()
torch.save(i.state_dict(), 'checkpoints/repro_random_image.pth.tar')
print('image params: %.2fM' % (sum(p.numel() for p in i.parameters()) / 1e6))
