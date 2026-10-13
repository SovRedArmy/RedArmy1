# 预计算全部数据集缓存(npy / alexnet 嵌入 / 探针梯度 / 伪标签光流), 一次性完成避免并发写冲突
import os
import sys

import numpy as np

from dynamics import content_cos_dist, probe_dynamics
from embedder import video_content_embeds
from pseudo_flow import compute_pseudo_flow
from video_data import VideoData

VIDEOS = sys.argv[1:] or ['beauty', 'bosphorus', 'honeybee', 'jockey',
                          'readysetgo', 'shakendry', 'yachtride']

for name in VIDEOS:
    print(f'=== {name} ===', flush=True)
    v = VideoData(f'data/{name}', cache_dir='./cache')
    print(f'  {v.n_frames} frames {v.W}x{v.H}', flush=True)
    ep = './cache/%s_alex.npy' % v.name
    if not os.path.isfile(ep):
        np.save(ep, video_content_embeds(v))
    print('  alexnet embeds done', flush=True)
    gp = './cache/%s_probegrads.npy' % v.name
    if not os.path.isfile(gp):
        np.save(gp, probe_dynamics(v))
    print('  probe grads done', flush=True)
    fh = 270 if v.H >= 1080 else v.H // 4
    compute_pseudo_flow(v, './cache', flow_h=fh)
    print('  pseudo flow done', flush=True)
print('ALL CACHES READY')
