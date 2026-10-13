# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 重建帧写出器：与 video_reader 对应，支持 PNG 序列（RGB）与裸 YUV420 输出

import os

from PIL import Image


class PNGWriter():
    # PNG 序列写出器：按 im00001.png 命名逐帧保存 RGB
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb):
        # 写出一帧 RGB（3xHxW uint8 → HxWx3 后保存）
        # rgb: 3xhxw uint8 numpy array
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(self.dst_path,
                                f'im{str(self.current_frame_index).zfill(self.padding)}.png'
                                )
        Image.fromarray(rgb).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1


class YUV420Writer():
    # 裸 YUV420 写出器：按 Y 平面 + 交错 UV 平面顺序写文件
    def __init__(self, dst_path, width, height):
        if not dst_path.endswith('.yuv'):
            dst_path = dst_path + '/out.yuv'
        self.dst_path = dst_path
        self.width = width
        self.height = height

        self.file = open(dst_path, 'wb')

    def write_one_frame(self, y, uv):
        # 写出一帧：y (1xHxW) + uv (2x(H/2)x(W/2))，均为 uint8
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()
