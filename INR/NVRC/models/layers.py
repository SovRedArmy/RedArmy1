"""
Functional Layers - all assume 5D inputs in [N, T, H, W, C].
（函数式层库：所有层假设输入为 5D [N, T, H, W, C]。

关键设计——FuncModule 函数式模块：
    权重不直接挂在层对象上，而是通过 _get_params() 声明，由根模块
    get_params_recursive() 汇总成扁平的 params 字典（键 = 模块路径__参数名），
    前向时用 _extract_params(params) 取回。这样 NVRC 编解码器就能整体地
    量化/替换/注入所有权重，这是“压缩模型参数”得以实现的基础。）
"""
from .utils import *


"""
Addtional properties for parameters
（参数额外属性：param_type 标记参数类别，NVRC 据此把参数分流到不同的熵模型——
  'grid' -> 网格熵模型；'net'/'conv_weight'/'linear_weight' 等 -> 权重熵模型；其余 -> 全精度）
"""
nn.Parameter.param_type = 'default'


def create_param(data, param_type='default', requires_grad=True):
    """（创建带 param_type 标注的 nn.Parameter）"""
    p = nn.Parameter(data, requires_grad=requires_grad)
    p.param_type = param_type
    return p


def is_network_param(p):
    """（判断是否为网络权重参数）"""
    assert isinstance(p, nn.Parameter), 'Parameter should be an instance of nn.Parameter.'
    assert hasattr(p, 'param_type'), 'Parameter should have param_type attribute.'
    return p.param_type == 'net'


def is_grid_param(p):
    """（判断是否为特征网格参数——NVRC 用它把网格分流到上下文熵模型）"""
    assert isinstance(p, nn.Parameter), 'Parameter should be an instance of nn.Parameter.'
    assert hasattr(p, 'param_type'), 'Parameter should have param_type attribute.'
    return p.param_type == 'grid'  


"""
FuncModule
"""
class FuncModule(nn.Module):
    """（函数式模块基类：见文件头说明；子类通过 _get_params() 声明参数）"""
    def __init__(self):
        super().__init__()
        
        # Key of the module relative to the root module.
        # Will be set by the root module.
        self.func_key = ''

        # List of parameters.
        # Should be set by the subclass, otherwise the parameters can't be extracted during forward pass.
        self.func_params_keys = []

    def init_func_modules(self):
        """
        Initialize the func_key of the all the child modules, relative to the module which call the function.
        （为所有子 FuncModule 设置相对键名，用于参数字典的扁平化命名）
        """
        for k, v in self.named_modules():
            if isinstance(v, FuncModule):
                assert v.func_key == '', 'func_key should not be set before initializing, and should be set only once.'
                v.func_key = k.replace('.', '_')

    def _get_params(self):
        """
        Return the parameters of the module for a single sample. Should be implemented by the subclass.
        （子类实现：返回本模块声明的参数字典）
        """
        return {}

    def get_params_prefix(self):
        """
        Return the prefix of the parameters of the module.
        （本模块参数在扁平字典中的前缀）
        """
        return self.func_key if self.func_key == '' else self.func_key + '__'

    def get_params_recursive(self):
        """
        Return the parameters of the module and all its child modules.
        （递归收集本模块与所有子模块的参数，键为 prefix__param_name）
        """
        params = {}
        for module in self.modules():
            if isinstance(module, FuncModule):
                for k, v in module._get_params().items():
                    params[f'{module.get_params_prefix()}{k}'] = v
        return params

    def _extract_params(self, params):
        """
        Extract the parameters of the module from the parameters dictionary.
        （从扁平参数字典中取回本模块的参数列表（按 func_params_keys 顺序））
        """
        return [params.get(f'{self.get_params_prefix()}{k}', None) for k in self.func_params_keys]


"""
Basic Layers（基础层：恒等/常量/线性/卷积/归一化/层缩放，均为函数式实现）
"""
class FuncIdentity(FuncModule):
    """（恒等层）"""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def forward(self, x, params):
        return x


class FuncConstant(FuncModule):
    """（常量层：返回可学习的常向量（用作帧级偏置））"""
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.func_params_keys = ['constant']

    def extra_repr(self):
        s = 'features={features}'
        return s.format(**self.__dict__)

    def _get_params(self):
        constant = create_param(torch.zeros(self.features), 'constant')
        return {'constant': constant}

    def forward(self, params):
        constant, = self._extract_params(params)
        return constant


class FuncLinear(FuncModule):
    """（函数式线性层：F.linear + 外部传入的 weight/bias）"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.func_params_keys = ['weight', 'bias']

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}, bias={bias}'
        return s.format(**self.__dict__)

    def _get_params(self):
        weight_shape = (self.out_features, self.in_features)
        weight = create_param(torch.empty(weight_shape), 'linear_weight')
        bias = create_param(torch.zeros(self.out_features), 'linear_bias') if self.bias else None
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        assert x.ndim >= 2, 'x should have at least 2 dimensions with shape [N, .., C]'
        weight, bias = self._extract_params(params)
        assert weight.ndim == 2, 'weight should have 2 dimensions with shape [C_out, C_in]'
        assert bias is None or bias.ndim == 1, 'bias should have 1 dimension with shape [C_out]'
        return F.linear(x, weight, bias)


class FuncConv2D(FuncModule):
    """（函数式 2D 卷积：把 [N,T,H,W,C] 折叠到 batch 维后做 F.conv2d（逐帧卷积））"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: str = 'same', dilation: int = 1, groups: int = 1, bias: bool= True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.func_params_keys = ['weight', 'bias']

    def extra_repr(self):
        s = 'in_channels={in_channels}, out_channels={out_channels}, ' \
            'kernel_size={kernel_size}, stride={stride}, ' \
            'padding={padding}, dilation={dilation}, groups={groups}, bias={bias}'
        return s.format(**self.__dict__)

    def _get_params(self):
        weight_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        weight = create_param(torch.empty(weight_shape), 'conv_weight')
        bias = create_param(torch.zeros(self.out_channels), 'conv_bias') if self.bias else None
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, T, H, W, C]'
        weight, bias = self._extract_params(params)
        assert weight.ndim == 4, 'weight should have 4 dimensions with shape [C_out, C_in, Kh, Kw]'
        assert bias is None or bias.ndim == 1, 'bias should have 1 dimension with shape [C_out]'
        N, T, H1, W1, _ = x.shape
        x = x.view(N * T, H1, W1, -1).permute(0, 3, 1, 2)
        x = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        _, _, H2, W2 = x.shape
        x = x.permute(0, 2, 3, 1).view(N, T, H2, W2, -1)
        return x


class FuncConv3D(FuncModule):
    """（函数式 3D 卷积：F.conv3d（时空联合卷积））"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: str = 'same', dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 3
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.func_params_keys = ['weight', 'bias']

    def extra_repr(self):
        s = 'in_channels={in_channels}, out_channels={out_channels}, ' \
            'kernel_size={kernel_size}, stride={stride}, '\
            'padding={padding}, dilation={dilation}, groups={groups}, bias={bias}'
        return s.format(**self.__dict__)

    def _get_params(self):
        weight_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        weight = create_param(torch.empty(weight_shape), 'conv_weight')
        bias = create_param(torch.zeros(self.out_channels), 'conv_bias') if self.bias else None
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, T, H, W, C]'
        weight, bias = self._extract_params(params)
        assert weight.ndim == 5, 'weight should have 4 dimensions with shape [C_out, C_in, Kt, Kh, Kw]'
        assert bias is None or bias.ndim == 1, 'bias should have 1 dimension with shape [C_out]'
        x = x.permute(0, 4, 1, 2, 3)
        x = F.conv3d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        x = x.permute(0, 2, 3, 4, 1)
        return x


class FuncLayerNorm(FuncModule):
    """（函数式 LayerNorm（可无仿射））"""
    def __init__(self, normalized_shape, eps: float = 1e-6, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if not isinstance(normalized_shape, tuple) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.bias = bias
        self.init_scale = 1.0
        self.func_params_keys = ['weight', 'bias']

    def extra_repr(self):
        s = 'normalized_shape={normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}, bias={bias}'
        return s.format(**self.__dict__)

    def _get_params(self):
        weight = create_param(self.init_scale * torch.ones(self.normalized_shape), 'norm_weight') \
                 if self.elementwise_affine else None
        bias = create_param(torch.zeros(self.normalized_shape), 'norm_bias') \
               if self.elementwise_affine and self.bias else None
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        assert x.ndim >= 2, 'x should have at least 2 dimensions with shape [N, .., C]'
        weight, bias = self._extract_params(params)
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class FuncLayerScale(FuncModule):
    """（LayerScale：逐通道可学习缩放，用于残差分支）"""
    def __init__(self, out_features: int, init_scale: float = 1e-3):
        super().__init__()
        self.out_features = out_features
        self.init_scale = init_scale
        self.func_params_keys = ['scale']

    def extra_repr(self):
        s = 'init_scale={init_scale}'
        return s.format(**self.__dict__)

    def _get_params(self):
        scale = create_param(self.init_scale * torch.ones([self.out_features]), 'scale')
        return {'scale': scale}

    def forward(self, x, params):
        assert x.ndim >= 2, 'x should have at least 2 dimensions with shape [N, .., C]'
        scale, = self._extract_params(params)
        return scale.view((1,) * (x.ndim - 1) + (scale.shape[-1],)) * x


"""
Advanced Layers（高级块：带残差/掩码的块结构）
"""
class FuncBlockBase(FuncModule):
    """（块基类：block_forward + 可选 LayerScale + 可选残差 shortcut，并用 mask 屏蔽 padding 区域）"""
    def __init__(self, in_features: int, out_features: int, layerscale_init: int, use_shortcut: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layerscale = FuncLayerScale(self.out_features, layerscale_init) if layerscale_init > 0. else None
        self.use_shortcut = use_shortcut
        self.func_params_keys = []

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}, use_shortcut={use_shortcut}'
        return s.format(**self.__dict__)

    def block_forward(self, x, params, mask):
        raise NotImplementedError

    def forward(self, x, params, mask):
        x, shortcut = self.block_forward(x, params, mask), x
        x = self.layerscale(x, params) if self.layerscale is not None else x
        x = x + shortcut if self.use_shortcut and self.in_features == self.out_features else x
        return x


class FuncLinearBlock(FuncBlockBase):
    """（线性块：norm/linear/act，用作 stem 或输出头）"""
    def __init__(self, in_features: int, out_features: int,
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False):
        super().__init__(in_features, out_features, 0., False)
        self.norm_first = norm_first
        self.linear = FuncLinear(in_features, out_features, bias=bias)
        self.norm = get_func_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_func_activation(act)()

    def block_forward(self, x, params, mask):
        x = self.linear(self.norm(x, params), params) if self.norm_first else self.norm(self.linear(x, params), params)
        x = self.act(x)
        x = mask * x if mask is not None else x
        return x


class FuncConv2DBlock(FuncBlockBase):
    """（卷积块：norm/conv2d/act，用作卷积 stem）"""
    def __init__(self, in_features: int, out_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False):
        super().__init__(in_features, out_features, 0., False)
        self.norm_first = norm_first
        self.conv = FuncConv2D(in_features, out_features, kernel_size, padding='same', bias=bias)
        self.norm = get_func_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_func_activation(act)()

    def block_forward(self, x, params, mask):
        x = self.conv(self.norm(x, params), params) if self.norm_first else self.norm(self.conv(x, params), params)
        x = self.act(x)
        x = mask * x if mask is not None else x
        return x


class FuncMLPBlock(FuncBlockBase):
    """（MLP 块：norm/fc1/act/fc2 + 残差）"""
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 act: str = 'gelu', norm: str = 'layernorm', bias: bool = True,
                 layerscale_init: float = 0.):
        super().__init__(in_features, out_features, layerscale_init, True)
        self.norm = get_func_norm(norm)(in_features)
        self.fc1 = FuncLinear(in_features, hidden_features, bias=bias)
        self.act = get_func_activation(act)()
        self.fc2 = FuncLinear(hidden_features, out_features, bias=bias)

    def block_forward(self, x, params, mask):
        x = self.norm(x, params)
        x = self.fc1(x, params)
        x = self.act(x)
        x = self.fc2(x, params)
        x = mask * x if mask is not None else x
        return x


class FuncConvNeXtBlock(FuncBlockBase):
    """（ConvNeXt 块（HiNeRV 主干块）：深度可分离 2D 卷积 -> norm -> fc1 -> act -> fc2 + 残差）"""
    def __init__(self, in_features: int, out_features: int, hidden_features: int, kernel_size: int = 3, 
                 act: str = 'gelu', norm: str = 'layernorm', bias: bool = True,
                 layerscale_init: float = 0.):
        super().__init__(in_features, out_features, layerscale_init, True)
        self.dconv = FuncConv2D(in_features, in_features, kernel_size, groups=in_features, padding='same', bias=bias)
        self.norm = get_func_norm(norm)(in_features)
        self.fc1 = FuncLinear(in_features, hidden_features, bias=bias)
        self.act = get_func_activation(act)()
        self.fc2 = FuncLinear(hidden_features, out_features, bias=bias)

    def block_forward(self, x, params, mask):
        x = self.dconv(x, params)
        x = self.norm(x, params)
        x = self.fc1(x, params)
        x = self.act(x)
        x = self.fc2(x, params)
        x = mask * x if mask is not None else x
        return x


class BatchGridTrilinear3D(nn.Module):
    """
    The module for mapping feature maps to a fixed size with trilinear interpolation.
    （把网格时间维三线性插值到固定输出帧数；因 F.interpolate 的限制仅支持时间维缩放）"""
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            is_batch = False
            N = 1
            T, H, W, C = x.shape
        elif x.ndim == 5:
            is_batch = True
            N, T, H, W, C = x.shape
        else:
            raise NotImplementedError
        assert H == self.output_size[1] and W == self.output_size[2], \
            'F.interpolate has incorrect results in some cases, so use only temporal scale'
        x = x.view(N, 1, T, H * W * C)
        x = F.interpolate(x, size=(self.output_size[0], H * W * C), mode='bilinear', align_corners=self.align_corners)
        x = x.view(-1, H, W, C) if not is_batch else x.view(N, self.output_size[0], H, W, C)
        return x


class Sin(FuncModule):
    """
    A simple sine function module that can be used as an activation function.
    It is useful for testing purposes or as a placeholder in a neural network.
    （正弦激活模块（占位/测试用））
    """
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x, params=None):
        return torch.sin(x)


"""
Utils（层工厂与模块列表执行器）
"""
def get_func_norm(norm, **kwargs):
    """（归一化层工厂：none/layernorm/layernorm-no-affine）"""
    if norm == 'none':
        return FuncIdentity
    elif norm == 'layernorm':
        return partial(FuncLayerNorm, eps=1e-6, **kwargs)
    elif norm == 'layernorm-no-affine':
        return partial(FuncLayerNorm, elementwise_affine=False, eps=1e-6, **kwargs)
    else:
        raise NotImplementedError


def get_func_activation(activation, **kwargs):
    """（激活函数工厂）"""
    if activation == 'none':
        return nn.Identity
    elif activation == 'relu':
        return nn.ReLU
    elif activation == 'relu6':
        return nn.ReLU6
    elif activation == 'leaky_relu':
        return nn.LeakyReLU
    elif activation == 'gelu':
        return nn.GELU
    elif activation == 'tanh':
        return nn.Tanh
    elif activation == 'sigmoid':
        return nn.Sigmoid
    elif activation == 'softplus':
        return nn.Softplus
    elif activation == 'sin':
        return Sin
    else:
        raise NotImplementedError


def get_func_block(type, **kwargs):
    """（块工厂：按类型名创建 stem/head/mlp/convnext 等块）"""
    if type == 'identity':
        return FuncIdentity()
    elif type == 'linear_stem':
        return FuncLinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                               act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'conv_stem':
        return FuncConv2DBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], kernel_size=kwargs['kernel_size'],
                               act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'linear_head':
        return FuncLinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                               act=kwargs['act'], norm=kwargs['norm'], norm_first=True, bias=kwargs['bias'])
    elif type == 'mlp':
        return FuncMLPBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['C_hidden'],
                            act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                            layerscale_init=kwargs['layerscale'])
    elif type == 'convnext':
        return FuncConvNeXtBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                                 hidden_features=kwargs['C_hidden'], kernel_size=kwargs['kernel_size'],
                                 act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                                 layerscale_init=kwargs['layerscale'])
    elif type == 'none':
        return
    else:
        raise NotImplementedError   


def _run_modules(module_list, x=None, params={}, mask=None):
    """（顺序执行模块列表：块类带 mask，普通函数式层带 params，纯 PyTorch 层直接调用）"""
    for m in module_list:
        if isinstance(m, FuncBlockBase):
            # For the normal blocks
            x = m(x, params, mask)
        elif isinstance(m, FuncConstant):
            # For the constant layers
            x = m(params)
        elif isinstance(m, FuncModule):
            # For the normal layers
            if mask is not None:
                x = m(x, params) * mask
            else:
                x = m(x, params)
        else:
            # General PyTorch layer
            x = m(x)
    return x


def run_modules(module_list, x=None, params={}, mask=None):
    """（_run_modules 的公开包装）"""
    return _run_modules(module_list, x, params=params, mask=mask)