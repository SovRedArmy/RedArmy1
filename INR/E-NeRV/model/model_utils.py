# -*- coding: utf-8 -*-
"""模型通用组件：位置编码、SIREN 激活及其初始化、激活/归一化层工厂函数、梯度工具。"""
from tkinter.messagebox import NO  # 误导入（IDE 自动补全引入的无关依赖，未使用；保留原样以免改动逻辑）
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """一维标量的正弦位置编码 gamma(x)（NeRV/E-NeRV 把帧号或坐标先编码再送入 MLP）。

    输出为 [sin(b^0*pi*x), cos(b^0*pi*x), ..., sin(b^(l-1)*pi*x), cos(b^(l-1)*pi*x)]，
    共 2*l 维。论文中时间/空间支路均取 b=1.25, l=80（embed_length=160）。
    若 pe_embed_b=0 则关闭编码，直接返回 [x]（embed_length=1）。
    """
    def __init__(self, pe_embed_b, pe_embed_l):
        super(PositionalEncoding, self).__init__()
        if pe_embed_b == 0:      # 关闭位置编码的模式
            self.embed_length = 1
            self.pe_embed = False
        else:
            self.lbase = float(pe_embed_b)   # 频率底数 b
            self.levels = float(pe_embed_l)  # 频率层数 l
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels  # sin/cos 成对，输出维度 = 2l
            self.pe_embed = True
    
    def __repr__(self):
        return f"Positional Encoder: pos_b={self.lbase}, pos_l={self.levels}, embed_length={self.embed_length}, to_embed={self.pe_embed}"

    def forward(self, pos):
        """pos: [N] 标量列表 -> [N, 2l] 编码张量"""
        if self.pe_embed is False:
            return pos[:,None]  # 不编码：[N] -> [N, 1]
        else:
            pe_list = []
            for i in range(self.levels):  # 逐频率生成 sin/cos 对
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)  # [N, 2l]


class Sin(nn.Module):
    """SIREN 使用的周期激活函数：sin(30*x)（系数 30 来自 SIREN 论文）"""
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(30 * input)  # see SIREN paper for the factor 30


def sin_init(m):
    """SIREN 隐藏层权重初始化：U(-sqrt(6/fan_in)/30, sqrt(6/fan_in)/30)，保证第一层输出覆盖多个周期"""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    """SIREN 第一层权重初始化：U(-1/fan_in, 1/fan_in)"""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class SIREN(nn.Module):
    """基于正弦激活的多层感知机（SIREN, Sitzmann et al.）。
    本仓库默认配置未启用（act=gelu），作为可选的激活方案保留。"""
    def __init__(self, dim_list, bias=True, first_layer_init=True):
        super(SIREN, self).__init__()

        nl = Sin()
        weight_init = sin_init
        if first_layer_init:
            first_init = first_layer_sine_init  # 第一层用专门初始化
        else:
            first_init = None

        fc_list = []
        for i in range(len(dim_list) - 1):
            dim_in = dim_list[i]
            dim_out = dim_list[i+1]
            fc_list += [nn.Linear(dim_in, dim_out, bias=bias), nl]  # Linear + Sin 交替
        self.fc = nn.Sequential(*fc_list)
        # init
        self.fc.apply(weight_init)      # 对所有层应用 SIREN 初始化
        if first_init is not None:
            self.fc[0].apply(first_init)  # 单独处理第一层
    
    def forward(self, x):
        return self.fc(x)


def ActivationLayer(act_type):
    """激活函数工厂：按配置字符串返回对应的 nn 激活模块。
    默认配置使用 'gelu'；'sin' 对应 SIREN 方案；'non' 为恒等映射。"""
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        # act_layer = torch.sin
        act_layer = Sin()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    elif act_type == 'non':
        act_layer = nn.Identity()
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    """归一化层工厂：'none' -> 恒等（默认配置），'bn' -> BatchNorm2d，'in' -> InstanceNorm2d。"""
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


def gradient(y, t):
    """计算 y 对 t 的梯度（保留计算图以便高阶导数）。
    为梯度差辅助损失预留的工具函数，当前默认配置未使用。"""
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad([y], [t], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
