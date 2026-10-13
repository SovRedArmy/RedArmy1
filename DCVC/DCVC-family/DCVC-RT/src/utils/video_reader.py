# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 视频源读取器: 支持 PNG 序列(RGB)与裸 YUV420 文件两种输入格式。

import os

import numpy as np
from PIL import Image


class PNGReader():
    """RGB PNG 序列读取器(目录内形如 im00001.png / im1.png 的帧序列)。"""

    def __init__(self, src_path, width, height, start_num=1):
        self.eof = False
        self.src_path = src_path
        self.width = width
        self.height = height
        pngs = os.listdir(self.src_path)
        # 根据文件名自动识别编号位数(vimeo 是 im00001.png, 其它可能是 im1.png)
        if 'im1.png' in pngs:
            self.padding = 1
        elif 'im00001.png' in pngs:
            self.padding = 5
        else:
            raise ValueError('unknown image naming convention; please specify')
        self.current_frame_index = start_num

    def read_one_frame(self):
        """读一帧, 返回 3xhxw 的 uint8 RGB; 文件不存在则置 eof 并返回 None。"""
        # rgb: 3xhxw uint8 numpy array
        if self.eof:
            return None

        png_path = os.path.join(self.src_path,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        if not os.path.exists(png_path):
            self.eof = True
            return None

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype(np.uint8).transpose(2, 0, 1)
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return rgb

    def close(self):
        """重置帧号(编码一遍后解码一遍时复用同一个 reader)。"""
        self.current_frame_index = 1


class YUV420Reader():
    """裸 YUV420 文件读取器(无封装的平面 Y + 交错 UV)。"""

    def __init__(self, src_path, width, height, skip_frame=0):
        self.eof = False
        if not src_path.endswith('.yuv'):
            src_path = src_path + '.yuv'
        self.src_path = src_path

        # 每帧的字节布局: Y 平面 w*h 字节 + UV 平面 w*h/2 字节(各为宽高的一半)
        self.y_size = width * height
        self.y_width = width
        self.y_height = height
        self.uv_size = width * height // 2
        self.uv_width = width // 2
        self.uv_height = height // 2
        # pylint: disable=R1732
        self.file = open(src_path, "rb")
        # pylint: enable=R1732
        skipped_frame = 0
        while not self.eof and skipped_frame < skip_frame:
            y = self.file.read(self.y_size)
            uv = self.file.read(self.uv_size)
            if not y or not uv:
                self.eof = True
            skipped_frame += 1

    def read_one_frame(self):
        """读一帧, 返回 (y: 1xhxw, uv: 2x(h/2)x(w/2)) 的 uint8 数组; EOF 返回 (None, None)。"""
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        if self.eof:
            return None, None
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)
        if not y or not uv:
            self.eof = True
            return None, None
        y = np.frombuffer(y, dtype=np.uint8).copy().reshape(1, self.y_height, self.y_width)
        uv = np.frombuffer(uv, dtype=np.uint8).copy().reshape(2, self.uv_height, self.uv_width)

        return y, uv

    def close(self):
        self.file.close()
