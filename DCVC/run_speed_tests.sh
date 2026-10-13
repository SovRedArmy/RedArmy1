#!/bin/bash
# 解码/编码速度复现（A6000）：对齐论文 Table 3c 设置——1080p、HEVC_B 5 序列、
# 97 帧、intra period -1（单 I 帧）、单卡、skip_thres 0.15（同 test_compress_time.py）。
set -e
cd /remote-home/gdzj21/VideoCompression/DCVC
P=/remote-home/gdzj21/miniconda3/envs/dcvcrt/bin/python
export CUDA_VISIBLE_DEVICES=0

for struct in ld hts htl; do
  echo "==== video model: $struct ===="
  $P test_video.py --verbose 1 --rate_num 1 --qp_i 30 --qp_p 30 \
    --test_config test_cfg/hevcb_png_1seq.json \
    --force_frame_num 97 \
    -w 1 --skip_thres 0.15 \
    --model_path_i checkpoints/cvpr2026_image.pth.tar \
    --model_path_p checkpoints/cvpr2026_video_${struct}.pth.tar \
    --model_structure $struct \
    --output_path output_speed_${struct}.json 2>&1 | tail -4
done

echo "==== image model (force_intra) ===="
$P test_video.py --verbose 1 --rate_num 1 --qp_i 30 \
    --force_intra 1 \
    --test_config test_cfg/hevcb_png_1seq.json \
    --force_frame_num 97 \
    -w 1 --skip_thres 0.15 \
    --model_path_i checkpoints/cvpr2026_image.pth.tar \
    --model_path_p checkpoints/cvpr2026_video_hts.pth.tar \
    --model_structure hts \
    --output_path output_speed_intra.json 2>&1 | tail -4

echo "==== summary ===="
$P - <<'EOF'
import json
for name in ['ld', 'hts', 'htl', 'intra']:
    res = json.load(open(f'output_speed_{name}.json'))['HEVC_B']
    enc, dec, n = [], [], 0
    for seq in res:
        for qp in res[seq]:
            r = res[seq][qp]
            enc.append(r['avg_frame_encoding_time'])
            dec.append(r['avg_frame_decoding_time'])
    import statistics as st
    fd = 8 if name in ('hts', 'htl') else 1
    if name == 'intra':
        fd = 1
    e, d = st.mean(enc), st.mean(dec)
    print(f"{name:6s}: enc {e*1000:7.2f} ms ({fd/e:7.1f} FPS) | dec {d*1000:7.2f} ms ({fd/d:7.1f} FPS) [A6000, 1080p]")
EOF
