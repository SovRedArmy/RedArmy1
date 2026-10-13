# 复现测速用：从随机初始化导出 4 个 checkpoint（结构与官方一致，数值为随机初始化）。
# 说明：GPU 前向/CUDA Graph 耗时与权重数值无关；仅熵编码符号分布与真实模型不同，
# 因此测得的编/解码时间中熵编码部分会略偏慢（码率偏高的上界）。结果仅供速度复现。
import sys
import torch

sys.path.insert(0, '.')

from src.models.image_model import DMCI
from src.models.video_model_ht import DMC as DMC_HT
from src.models.video_model_ld import DMC as DMC_LD
from src.utils.common import ModelStructure

torch.manual_seed(0)

i_net = DMCI()
torch.save({'state_dict': i_net.state_dict()}, 'checkpoints/cvpr2026_image.pth.tar')
del i_net

for name, cls, kw in [
    ('cvpr2026_video_htl.pth.tar', DMC_HT, {'model_structure': ModelStructure.HTL}),
    ('cvpr2026_video_hts.pth.tar', DMC_HT, {'model_structure': ModelStructure.HTS}),
    ('cvpr2026_video_ld.pth.tar', DMC_LD, {}),
]:
    net = cls(**kw)
    torch.save({'state_dict': net.state_dict()}, f'checkpoints/{name}')
    del net
    print('saved', name)
print('done')
