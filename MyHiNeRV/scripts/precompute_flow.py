"""
离线预计算 RAFT 光流伪标签（Scene Matters SFCM 的教师信号）。
用法：
  python scripts/precompute_flow.py --dataset <root> --dataset-name <name> \
      --crop-size -1 -1 --input-size 1080 1920
输出：<root>/<name>_flow.pt（[T,2,H/2,W/2] fp16 双向流）
"""
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flow_utils


def str_to_list(s, cast):
    return [cast(x) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--crop-size', type=int, nargs='+', default=[-1, -1])
    parser.add_argument('--input-size', type=int, nargs='+', default=[-1, -1])
    parser.add_argument('--flow-cache-frames', type=int, default=600,
                        help='帧数不超过该值时把全部帧缓存在内存里加速。')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    logger = logging.getLogger('precompute_flow')

    flow_utils.precompute_flow(args, logger)


if __name__ == '__main__':
    main()
