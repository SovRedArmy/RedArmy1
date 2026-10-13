"""
熵模型专用层：掩码卷积（MaskedConv）——自回归上下文网络的基本单元。
掩码卷积只存储“半个”卷积核（按扫描顺序的先行邻居），前向时补零还原成完整卷积核，
天然保证因果性（不偷看未来位置）。is_head=True 的最后一层输出 2C 通道（mean 与 log_scale），
其参数名前缀为 em_conv，便于与普通网络参数区分。
"""
from .utils import *


class FuncMaskedConv2d(FuncConv2D):
    """（2D 掩码卷积：用于自回归熵参数预测；权重形状只含前一半核位置）"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: str = 'same', dilation: int = 1, groups: int = 1, bias: bool = True,
                 mask_cur: bool = True, is_head: bool = False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_shape = (1, 1, *self.kernel_size)
        self.mask_cur = mask_cur   # 是否允许看到当前位置（False = 严格因果）
        self.is_head = is_head     # 是否为输出层（决定参数前缀 em_conv）

    def _get_params(self):
        """（参数：只创建前 half + mask_cur 个核位置的权重）"""
        param_prefix = 'em_conv' if self.is_head else 'conv'
        weight_shape = (self.out_channels, self.in_channels // self.groups,
                        math.prod(self.kernel_size) // 2 + (not self.mask_cur))
        weight = create_param(torch.empty(weight_shape), f'{param_prefix}_weight')
        bias = create_param(torch.zeros(self.out_channels), f'{param_prefix}_bias') if self.bias else None
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        """（前向：把半核权重补零成完整卷积核后调用普通卷积）"""
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, T, H, W, C]'
        weight, bias = self._extract_params(params)
        weight = F.pad(weight, (0, math.prod(self.kernel_size) - weight.shape[-1])) \
                  .view(*weight.shape[:-1], *self.kernel_size)
        prefix = self.func_key if self.func_key == '' else self.func_key + '__'
        return super().forward(x, {f'{prefix}weight': weight, f'{prefix}bias': bias})


class FuncMaskedConv3d(FuncConv3D):
    """（3D 掩码卷积：NVRC 特征网格上下文网络（ARGridEntropyModel）使用的版本）"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: str = 'same', dilation: int = 1, groups: int = 1, bias: bool = True,
                 mask_cur: bool = True, is_head: bool = False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_shape = (1, 1, *self.kernel_size)
        self.mask_cur = mask_cur
        self.is_head = is_head

    def _get_params(self):
        """（参数：只创建前 half + mask_cur 个核位置的权重）"""
        param_prefix = 'em_conv' if self.is_head else 'conv'
        weight_shape = (self.out_channels, self.in_channels // self.groups, 
                        math.prod(self.kernel_size) // 2 + (not self.mask_cur))
        weight = create_param(torch.empty(weight_shape), f'{param_prefix}_weight')
        bias = create_param(torch.zeros(self.out_channels), f'{param_prefix}_bias') if self.bias else None
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        """（前向：补零成完整 3D 卷积核后调用普通卷积）"""
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, T, H, W, C]'
        weight, bias = self._extract_params(params)
        weight = F.pad(weight, (0, math.prod(self.kernel_size) - weight.shape[-1])) \
                  .view(*weight.shape[:-1], *self.kernel_size)
        prefix = self.func_key if self.func_key == '' else self.func_key + '__'
        return super().forward(x, {f'{prefix}weight': weight, f'{prefix}bias': bias})