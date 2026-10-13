# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# 码流(bitstream)封装: 参数集(SPS)与帧(I/P)NAL 单元的读写。
# 码流结构(每个序列一个文件):
#   [SPS][I/P 帧 x N]...
# SPS(序列参数集): nal_type(4bit)=0 + sps_id(4bit) + 高度(变长) + 宽度(变长)
#                  + ec_part(1bit, 是否双熵编码器) + use_ada_i(1bit)
# I/P 帧:          nal_type(4bit)=1/2 + sps_id(4bit) + qp(1B) + 码长(变长) + rANS 码流
# ============================================================================

import enum
import struct
from pathlib import Path


def filesize(filepath: str) -> int:
    """返回文件大小(字节)。"""
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    """以大端字节序写入若干 uint32, 返回写入字节数。"""
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    """以大端字节序写入若干 uint8, 返回写入字节数。"""
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values)


def read_uints(fd, n, fmt=">{:d}I"):
    """读出 n 个大端 uint32。"""
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    """读出 n 个大端 uint8。"""
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    """写入原始字节串(rANS 码流本体)。"""
    if len(values) == 0:
        return 0
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values)


def read_bytes(fd, n, fmt=">{:d}s"):
    """读出 n 字节原始数据。"""
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    """以大端字节序写入若干 uint16。"""
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 2


def read_ushorts(fd, n, fmt=">{:d}H"):
    """读出 n 个大端 uint16。"""
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_uint_adaptive(f, a):
    """自适应变长无符号整数: 按数值大小用 1/2/4 字节编码, 首字节高位标识长度。

    用于 SPS 的宽高与每帧码流长度等大多数情况较小的值, 节省头信息比特。"""
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


def read_uint_adaptive(f):
    """write_uint_adaptive 的解码端: 按首字节高位模式判断 1/2/4 字节并还原数值。"""
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


class NalType(enum.IntEnum):
    """NAL 单元类型: 0=SPS 参数集, 1=I 帧, 2=P 帧。"""
    NAL_SPS = 0
    NAL_I = 1
    NAL_P = 2


class SPSHelper():
    """SPS 管理器: 编码端负责去重分配 sps_id, 解码端负责按 id 存取 SPS。

    相邻帧的 SPS 通常相同(仅 use_ada_i 等标志偶发变化), 去重后不必每帧重写 SPS。"""

    def __init__(self):
        super().__init__()
        self.spss = []

    def get_sps_id(self, target_sps):
        """编码端: 若已存在内容相同的 SPS 则复用其 id(返回 (id, False));

        否则分配新 id(上限 15) 并返回 (id, True) 表示需要写出新 SPS。"""
        min_id = -1
        for sps in self.spss:
            if sps['height'] == target_sps['height'] and sps['width'] == target_sps['width'] and \
                    sps['use_ada_i'] == target_sps['use_ada_i'] and \
                    sps['ec_part'] == target_sps['ec_part']:
                return sps['sps_id'], False
            if sps['sps_id'] > min_id:
                min_id = sps['sps_id']
        assert min_id < 15
        sps = target_sps.copy()
        sps['sps_id'] = min_id + 1
        self.spss.append(sps)
        return sps['sps_id'], True

    def add_sps_by_id(self, sps):
        """解码端: 按 sps_id 覆盖或追加一条 SPS 记录。"""
        for i in range(len(self.spss)):
            if self.spss[i]['sps_id'] == sps['sps_id']:
                self.spss[i] = sps.copy()
                return
        self.spss.append(sps.copy())

    def get_sps_by_id(self, sps_id):
        """解码端: 按 id 查找 SPS。"""
        for sps in self.spss:
            if sps['sps_id'] == sps_id:
                return sps
        return None


def write_sps(f, sps):
    """写出一个 SPS NAL: 类型+id(1B) + 高度(变长) + 宽度(变长) + 标志字节。"""
    # nal_type(4), sps_id(4)
    # height (variable)
    # width (vairable)
    # 0(6), ec_part(1) use_ada_i(1)
    assert sps['sps_id'] < 16
    assert sps['use_ada_i'] < 2
    written = 0
    flag = int((NalType.NAL_SPS << 4) + sps['sps_id'])
    written += write_uchars(f, (flag,))
    written += write_uint_adaptive(f, sps['height'])
    written += write_uint_adaptive(f, sps['width'])
    flag = (sps['ec_part'] << 2) + sps['use_ada_i']
    written += write_uchars(f, (flag,))
    return written


def read_header(f):
    """读取一个 NAL 单元的头部(1 字节): 解析 nal_type 与 sps_id。

    若 nal_type>=3(本实现未用), 则按"多帧打包"格式继续读取各帧的 sps_id 列表。"""
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


def read_sps_remaining(f, sps_id):
    """读完 SPS 剩余部分(头部已读出 sps_id): 高度/宽度/标志字节。"""
    sps = {}
    sps['sps_id'] = sps_id
    sps['height'] = read_uint_adaptive(f)
    sps['width'] = read_uint_adaptive(f)
    flag = read_uchars(f, 1)[0]
    sps['ec_part'] = (flag >> 2) & 0x01
    sps['use_ada_i'] = flag & 0x01
    return sps


def write_ip(f, is_i_frame, sps_id, qp, bit_stream):
    """写出一个 I/P 帧条目: 类型+引用的 sps_id(1B) + qp(1B) + 码长(变长) + rANS 码流。"""
    written = 0
    flag = (int(NalType.NAL_I if is_i_frame else NalType.NAL_P) << 4) + sps_id
    written += write_uchars(f, (flag,))
    assert qp < 256 and qp >= 0
    flag = qp
    written += write_uchars(f, (flag,))
    # we write all the streams in the same file, thus, we need to write the per-frame length
    # if packed independently, we do not need to write it
    written += write_uint_adaptive(f, len(bit_stream))
    written += write_bytes(f, bit_stream)
    return written


def read_ip_remaining(f):
    """读完 I/P 帧条目剩余部分(头部已读出类型与 sps_id): qp + 码长 + 码流。"""
    flag = read_uchars(f, 1)[0]
    qp = flag
    stream_length = read_uint_adaptive(f)
    bit_stream = read_bytes(f, stream_length)
    return qp, bit_stream
