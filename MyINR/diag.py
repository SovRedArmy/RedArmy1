# 诊断: 在 bunny clip0 (0:41) 上用同一小模型架构, 逐个开关模块, 30 epoch 对比 PSNR 轨迹
import sys

import numpy as np
import torch

from trainer import Cfg, ClipTrainer
from video_data import VideoData


def run(name, epochs=30, **over):
    v = VideoData('data/bunny', cache_dir='./cache')
    cfg_kw = dict(mode='ours', epochs=epochs, stem_dim_num='128_1', fc_hw_dim='9_16_55',
                  lower_width=32, strides=[5, 2, 2, 2, 2], pair_batch=True, eval_freq=15,
                  prune=False, seed=1)
    cfg_kw.update(over)
    cfg = Cfg(**cfg_kw)
    tr = ClipTrainer(v, cfg, (0, 41), device='cuda')
    n_par = sum(p.numel() for p in tr.model.parameters()) / 1e6
    tr.train(frame_weights=None, flow_pseudo=None, verbose=False)
    psnr = tr.quick_eval()
    print(f'{name:32s} params={n_par:.2f}M  PSNR@{epochs}ep={psnr:.2f}', flush=True)
    return psnr


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'A'):
        run('A ours-full(pair)')
    if which in ('all', 'B'):
        run('B ms-only(pair)', use_fft=False, use_flow=False, use_tcl=False, use_fw=False, use_hfb=False)
    if which in ('all', 'C'):
        run('C ms-only(single-frame)', use_fft=False, use_flow=False, use_tcl=False,
            use_fw=False, use_hfb=False, pair_batch=False)
    if which in ('all', 'D'):
        run('D single-res(pair)', use_fft=False, use_flow=False, use_tcl=False,
            use_fw=False, use_hfb=False, use_ms=False)
    if which in ('all', 'E'):
        run('E arch512_26lw64(ms-only,pair)', use_fft=False, use_flow=False, use_tcl=False,
            use_fw=False, use_hfb=False, stem_dim_num='512_1', fc_hw_dim='9_16_26', lower_width=64)
    if which in ('all', 'F'):
        run('F baseline-style(single-res,single-frame)', mode='baseline',
            stem_dim_num='512_1', fc_hw_dim='9_16_26', lower_width=96)

def isolate():
    import numpy as np
    from dynamics import frame_weights
    v = VideoData('data/bunny', cache_dir='./cache')
    grads = np.load('./cache/bunny_probegrads.npy')
    fw = frame_weights(grads)
    variants = {
        'I1 ms+fft': dict(use_flow=False, use_tcl=False, use_fw=False, use_hfb=False),
        'I2 ms+flow': dict(use_fft=False, use_tcl=False, use_fw=False, use_hfb=False),
        'I3 ms+tcl': dict(use_fft=False, use_flow=False, use_fw=False, use_hfb=False),
        'I4 ms+hfb': dict(use_fft=False, use_flow=False, use_tcl=False, use_fw=False),
    }
    for name, over in variants.items():
        cfg_kw = dict(mode='ours', epochs=30, stem_dim_num='128_1', fc_hw_dim='9_16_55',
                      lower_width=32, strides=[5, 2, 2, 2, 2], pair_batch=True, eval_freq=30,
                      prune=False, seed=1)
        cfg_kw.update(over)
        cfg = Cfg(**cfg_kw)
        tr = ClipTrainer(v, cfg, (0, 41), device='cuda')
        tr.train(frame_weights=fw[0:41] if not over.get('use_fw', True) else None,
                 flow_pseudo=None, verbose=False)
        print(f'{name:20s} PSNR@30ep={tr.quick_eval():.2f}', flush=True)
