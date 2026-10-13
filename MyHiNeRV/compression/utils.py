"""压缩模块通用工具：模型解包、量化轴选择。"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def unwrap_model(model):
    """剥掉 torch.compile / DDP 包装，返回原始模型。"""
    model = model._orig_mod if hasattr(model, '_orig_mod') else model # For compiled models
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model # For DDP models
    return model

def compute_best_quant_axis(x, thres=0.05):
    """
    Compute the best quantization axis for a tensor. 
    Similar to the one used in HNeRV quantization: https://github.com/haochen-rye/HNeRV/blob/main/hnerv_utils.py#L26
    为张量选择逐通道量化的轴：在"该轴长度不使分组过碎"的约束下
    （每组元素数 ≥ 总元素数×(1-thres)），选择最长的那一轴；
    都不满足则返回 None（逐张量量化）。
    """
    best_axis = None
    best_axis_dim = 0
    for axis in range(x.ndim):
        dim = x.shape[axis]
        if x.numel() / dim >= x.numel() * thres:
            continue
        if dim > best_axis_dim:
            best_axis = axis
            best_axis_dim = dim
    return best_axis