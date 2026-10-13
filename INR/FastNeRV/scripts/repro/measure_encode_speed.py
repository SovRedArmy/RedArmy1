"""
补充实验工具：
1) 从训练完成的 checkpoint 加载 NeRV-Enc，测量"编码一个视频"（encode: 视频 -> z/θ̂'）的耗时；
2) 测量 NeRV-Dec 解码耗时（decode: z -> 所有序列帧）；
3) 输出视频表示大小（z 元素数与 8bit 量化后的比特数）。
"""
import sys, time, warnings, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, '/remote-home/gdzj21/VideoCompression/INR/FastNeRV')

import torch
import models
from models.enc_models.nerv_enc import NeRVEnc

def main(ckpt_path, n_videos=50, device='cuda:1'):
    model = NeRVEnc.from_checkpoint(ckpt_path).to(device).eval()

    # 构造与训练一致的输入（从真实视频读取）
    import datasets
    ds = datasets.make({'name': 'video_dataset', 'args': dict(
        frame_num=8, t_duration=1.0, cls_vid_num='400_25', crop_size=256,
        csv_file='k400_val_trim.csv',
        root_path='/remote-home/gdzj21/VideoCompression/INR/FastNeRV/data/dataset_meta/meta',
        split='val', resize=False, rand_flip=False, decode_type='torchvision')}, args={'zero_center': False})
    videos = torch.stack([ds[i]['video'] for i in range(n_videos)]).to(device)

    # ---- 编码速度（batch=1，逐视频）----
    with torch.no_grad():
        for i in range(5):
            model.encode(videos[i:i+1])  # 预热
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n_videos):
            z_info = model.encode(videos[i:i+1])
        torch.cuda.synchronize()
        enc_ms = (time.time() - t0) / n_videos * 1000

        # ---- 批量编码（数据加载器场景）----
        t0 = time.time()
        z_info = model.encode(videos)
        torch.cuda.synchronize()
        enc_batch_ms = (time.time() - t0) / n_videos * 1000
        z = z_info['z']

        # ---- 解码速度（z -> 8 帧 256x256）----
        for i in range(3):
            model.decode(z[:1])
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n_videos):
            model.decode(z[i:i+1])
        torch.cuda.synchronize()
        dec_ms = (time.time() - t0) / n_videos * 1000

    n_elements = z[0].numel()
    print(f'encode per-video (batch=1): {enc_ms:.1f} ms')
    print(f'encode per-video (batch={n_videos}): {enc_batch_ms:.1f} ms')
    print(f'decode per-video (batch=1): {dec_ms:.1f} ms')
    print(f'video representation: {n_elements} values -> 8bit = {n_elements} Bytes = {n_elements/1024:.1f} KB')
    return enc_ms, dec_ms


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='/remote-home/gdzj21/VideoCompression/INR/FastNeRV/outputs/repro_full/epoch-last.pth')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    main(a.ckpt, a.n, a.device)
