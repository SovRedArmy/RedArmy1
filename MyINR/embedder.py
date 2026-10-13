# ==========================================================================================
# 固定特征嵌入器: ImageNet 预训练 AlexNet(本机缓存 ~/.cache/torch, 无需联网)
# 用途:
#   1. 场景切分 —— 相邻帧内容特征的余弦距离(TeNeRV 的 D_t)
#   2. TCL 时序对比损失 —— 重建帧的固定编码器嵌入(代替 Scene Matters 的预训练 ResNet)
# ==========================================================================================
import os

import numpy as np
import torch
import torch.nn.functional as F

CKPT = os.path.expanduser('~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class AlexnetEmbedder(torch.nn.Module):
    """返回 AlexNet 分类头前的 256 维 avgpool 特征(归一化)。"""

    def __init__(self, device='cuda'):
        super().__init__()
        import torchvision
        m = torchvision.models.alexnet(weights=None)
        sd = torch.load(CKPT, map_location='cpu')
        m.load_state_dict(sd)
        self.feat = m.features
        self.avg = m.avgpool
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', _MEAN.to(device))
        self.register_buffer('std', _STD.to(device))

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        x = (x - self.mean) / self.std
        f = self.avg(self.feat(x)).mean((2, 3))    # (B, 256) 通道平均池化
        return f


@torch.no_grad()
def video_content_embeds(video, device='cuda', batch=16):
    """整段视频的 GT 内容特征 (N,256)。"""
    emb = AlexnetEmbedder(device)
    out = []
    for i in range(0, video.n_frames, batch):
        x = video.frame_tensor(list(range(i, min(i + batch, video.n_frames))), device)
        out.append(emb(x).cpu())
    return torch.cat(out).numpy()
