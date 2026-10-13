"""
模型通用小工具：布尔字符串解析、配置覆盖、取模、整除对齐、形状断言。
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Union, List, Optional
from functools import partial


def str_to_bool(s):
    """把命令行字符串 'true'/'false' 解析成 bool。"""
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError


def cfg_override(cfg, **kwargs):
    """复制 cfg 字典并用 kwargs 覆盖/新增键值（不修改原 dict）。"""
    cfg = cfg.copy()
    for k, v in kwargs.items():
        cfg[k] = v
    return cfg


def mod(a, b):
    """张量取模 a mod b（等价于 %，用于局部坐标计算）。"""
    return a - a.div(b, rounding_mode='floor') * b


def divide_to_multiple(a, b, factor):
    """把 a//b 向上取整到 factor 的整数倍（用于通道数对齐）。"""
    return int(math.ceil((a // b) / factor) * factor)


def assert_shape(x: torch.Tensor, shape: tuple):
    """断言张量形状与给定 shape 完全一致（调试用）。"""
    assert tuple(x.shape) == tuple(shape), f'shape: {x.shape}     expected: {shape}'