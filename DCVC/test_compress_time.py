# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 编/解码速度基准测试脚本：调用 test_video.py 在 HEVC_B（runtime_avg.json 配置）
# 上跑 4 个码率点，然后从输出 json 汇总平均单 chunk 编/解码时间并换算 FPS。
# --img 时只测帧内模型（force_intra），否则测完整视频模型。

import argparse
import json
import os
import sys


def parse_args(argv):
    # 参数：--img 是否只测帧内模型；--model_structure 三种结构；输出 json 路径
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', action='store_true', help='test image codec')
    parser.add_argument('--model_structure', type=str, default='ld', choices=['htl', 'hts', 'ld'],
                        help='model structure to test')
    parser.add_argument('--output_path', type=str, default='temp.json')

    args = parser.parse_args(argv)

    return args


def main(argv):
    # 组装 test_video.py 命令行（4 个码率点、单 worker、skip_thres=0.15）并执行，
    # 再读取输出 json 汇总 HEVC_B 上所有序列所有码率的平均编/解码时间，
    # 按 chunk 大小（frame_delay）换算为每帧 FPS
    args = parse_args(argv)

    model_path_i = 'checkpoints/cvpr2026_image.pth.tar'
    model_path_p = f'checkpoints/cvpr2026_video_{args.model_structure}.pth.tar'

    dataset = 'HEVC_B'
    benchmark_config = 'test_cfg/runtime_avg.json'

    if args.img:
        img_arg = ' --force_intra 1'
    else:
        img_arg = ' --force_intra 0'

    command_line = (f' python test_video.py --verbose 2 --rate_num 4 {img_arg}'
                    f' --test_config {benchmark_config}'
                    f' --force_frame_num -1'
                    f' --cuda_idx 0 -w 1'
                    ' --skip_thres 0.15'
                    f' --output_path {args.output_path}'
                    f' --model_path_i {model_path_i}'
                    f' --model_path_p {model_path_p}'
                    f' --model_structure {args.model_structure}')

    print(command_line, flush=True)
    os.system(command_line)

    res = json.load(open(args.output_path, 'r'))
    res = res[dataset]
    encoding_time = []
    decoding_time = []
    for seq in res:
        for qp in res[seq]:
            enc_t = res[seq][qp]['avg_frame_encoding_time']
            dec_t = res[seq][qp]['avg_frame_decoding_time']
            encoding_time.append(enc_t)
            decoding_time.append(dec_t)
    avg_encoding_time = sum(encoding_time) / len(encoding_time)
    avg_decoding_time = sum(decoding_time) / len(decoding_time)
    if args.model_structure == 'ld':
        from src.models.video_model_ld import g_frame_delay as frame_delay
    else:
        from src.models.video_model_ht import g_frame_delay as frame_delay
    print(f'Average encoding time on {dataset}'
          f' = {avg_encoding_time * 1000:.4f} ms / {frame_delay / avg_encoding_time:.4f} fps')
    print(f'Average decoding time on {dataset}'
          f' = {avg_decoding_time * 1000:.4f} ms / {frame_delay / avg_decoding_time:.4f} fps')


if __name__ == '__main__':
    main(sys.argv[1:])
