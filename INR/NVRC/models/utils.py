"""
Model utils（模型工具：配置覆盖、形状断言与多尺度尺寸推算）
"""
from utils import *


def str_to_bool(s):
    """（字符串转布尔）"""
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError


def cfg_override(cfg, **kwargs):
    """（复制配置字典并覆盖指定键）"""
    cfg = cfg.copy()
    for k, v in kwargs.items():
        cfg[k] = v
    return cfg


def assert_shape(x: torch.Tensor, shape: tuple):
    """（断言张量形状）"""
    assert tuple(x.shape) == tuple(shape), f'shape: {x.shape}     expected: {shape}'


def compute_multiscale_sizes(size, scales, divisible_only=False):
    """（从输出尺寸逐级除以各 stage 的 scale，得到每个 stage 的输入/输出尺寸链）"""
    sizes = [size]
    for scale in scales:
        assert not divisible_only or all(size[d] % scale[d] == 0 for d in range(len(size)))
        size = tuple(int(size[d] / scale[d]) for d in range(len(size)))
        sizes.append(size)
    return sizes