# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 码流（bitstream）封装/解析工具：仿视频编码标准，把压缩数据组织为 NAL 单元。
# NAL 类型：SPS（序列参数集：高宽）、I 帧、P 帧（chunk）。
# 每个 I/P NAL 头部记录 [nal_type, sps_id, qp, ec_part(熵编码并行子流数),
# reset_feature_memory(记忆重置标志), 码长]，随后是 rANS 字节流。

import enum
import struct

from pathlib import Path


def filesize(filepath: str) -> int:
    # 返回文件大小（字节）
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def read_bytes(fd, n, fmt='>{:d}s'):
    # 从码流读 n 个字节
    sz = struct.calcsize('s')
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_uchars(fd, n, fmt='>{:d}B'):
    # 从码流读 n 个无符号字节
    sz = struct.calcsize('B')
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_header(f):
    # 读 NAL 头：高 4 位 nal_type；SPS/I/P 时低 4 位为 sps_id
    header = {}
    flag = read_uchars(f, 1)[0]
    nal_type = flag >> 4
    header['nal_type'] = NalType(nal_type)
    if nal_type < 3:
        header['sps_id'] = flag & 0x0f
        return header

    frame_num_minus1 = flag & 0x0f
    frame_num = frame_num_minus1 + 1
    header['frame_num'] = frame_num
    sps_ids = []
    for _ in range(0, frame_num, 2):
        flag = read_uchars(f, 1)[0]
        sps_ids.append(flag >> 4)
        sps_ids.append(flag & 0x0f)
    sps_ids = sps_ids[:frame_num]
    header['sps_ids'] = sps_ids
    return header


def read_uint_adaptive(f):
    # 自适应长度无符号整数读取：首字节高 2 位指示总长度（1/2/4 字节），小端序
    a3 = read_uchars(f, 1)[0]
    if (a3 >> 7) == 0:
        return a3

    a2 = read_uchars(f, 1)[0]

    if (a3 >> 6) == 0x02:
        a3 = a3 & 0x3f
        return (a3 << 8) + a2
    a3 = a3 & 0x3f
    a1 = read_uchars(f, 1)[0]
    a0 = read_uchars(f, 1)[0]
    return (a3 << 24) + (a2 << 16) + (a1 << 8) + a0


def read_ip_remaining(f):
    # 读 I/P NAL 的剩余部分：qp、ec_part（7bit 熵编码并行子流数）、
    # reset_feature_memory（1bit 记忆重置标志）、码长与码流本体
    flag = read_uchars(f, 1)[0]
    qp = flag
    flag = read_uchars(f, 1)[0]
    ec_part = (flag >> 1) & 0x7f
    reset_feature_memory = flag & 0x01
    stream_length = read_uint_adaptive(f)
    bit_stream = read_bytes(f, stream_length)
    return qp, ec_part, reset_feature_memory, bit_stream


def read_sps_remaining(f, sps_id):
    # 读 SPS NAL 的剩余部分：图像高、宽
    sps = {}
    sps['sps_id'] = sps_id
    sps['height'] = read_uint_adaptive(f)
    sps['width'] = read_uint_adaptive(f)
    return sps


def write_bytes(fd, values, fmt='>{:d}s'):
    # 向码流写字节串，返回写入字节数
    if len(values) == 0:
        return 0
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values)


def write_uchars(fd, values, fmt='>{:d}B'):
    # 向码流写无符号字节
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values)


def write_uint_adaptive(f, a):
    # 自适应长度无符号整数写入：按值大小选择 1/2/4 字节编码，小端序
    if a < (1 << 7):
        a0 = (a >> 0) & 0xff
        a0 = a0 | (0x00 << 7)
        write_uchars(f, (a0,))
        return 1

    if a < (1 << 14):
        a0 = (a >> 0) & 0xff
        a1 = (a >> 8) & 0xff
        a1 = a1 | (0x02 << 6)
        write_uchars(f, (a1, a0))
        return 2

    assert a < (1 << 30)
    a0 = (a >> 0) & 0xff
    a1 = (a >> 8) & 0xff
    a2 = (a >> 16) & 0xff
    a3 = (a >> 24) & 0xff
    a3 = a3 | (0x03 << 6)
    write_uchars(f, (a3, a2, a1, a0))
    return 4


def write_ip(f, is_i_frame, sps_id, qp, ec_part, reset_feature_memory, bit_stream):
    # 写一个 I/P NAL：[nal_type+sps_id(1B)][qp(1B)][ec_part(7b)+reset(1b)(1B)]
    # [自适应长度码长][码流]，返回写入字节数
    written = 0
    flag = (int(NalType.NAL_I if is_i_frame else NalType.NAL_P) << 4) + sps_id
    written += write_uchars(f, (flag,))
    assert 0 <= qp < 256
    written += write_uchars(f, (qp,))
    # ec_part(7), reset_feature_memory(1)
    flag = (ec_part << 1) + reset_feature_memory
    written += write_uchars(f, (flag,))
    written += write_uint_adaptive(f, len(bit_stream))
    written += write_bytes(f, bit_stream)
    return written


def write_sps(f, sps):
    # 写一个 SPS NAL：[nal_type(4b)+sps_id(4b)][height][width]（自适应长度整数）
    # nal_type(4), sps_id(4)
    # height (variable)
    # width (variable)
    assert sps['sps_id'] < 16
    written = 0
    flag = int((NalType.NAL_SPS << 4) + sps['sps_id'])
    written += write_uchars(f, (flag,))
    written += write_uint_adaptive(f, sps['height'])
    written += write_uint_adaptive(f, sps['width'])
    return written


class NalType(enum.IntEnum):
    # NAL 单元类型：SPS=0（序列参数集）、I=1（帧内帧）、P=2（帧间 chunk）
    NAL_SPS = 0
    NAL_I = 1
    NAL_P = 2


class SPSHelper():
    # SPS 管理器：编码时去重分配 sps_id，解码时按 id 查找已解析的 SPS
    def __init__(self):
        super().__init__()
        self.spss = []

    def add_sps_by_id(self, sps):
        # 解码侧登记一个 SPS（同 id 覆盖）
        for i, existing_sps in enumerate(self.spss):
            if existing_sps['sps_id'] == sps['sps_id']:
                self.spss[i] = sps.copy()
                return
        self.spss.append(sps.copy())

    def get_sps_by_id(self, sps_id):
        # 按 id 查找 SPS
        for sps in self.spss:
            if sps['sps_id'] == sps_id:
                return sps
        return None

    def get_sps_id(self, target_sps):
        # 编码侧：若已有同分辨率的 SPS 则复用其 id（sps_new=False），
        # 否则分配新 id（最多 16 个）并标记需要写出到码流
        min_id = -1
        for sps in self.spss:
            if sps['height'] == target_sps['height'] and sps['width'] == target_sps['width']:
                return sps['sps_id'], False
            if sps['sps_id'] > min_id:
                min_id = sps['sps_id']
        assert min_id < 15
        sps = target_sps.copy()
        sps['sps_id'] = min_id + 1
        self.spss.append(sps)
        return sps['sps_id'], True
