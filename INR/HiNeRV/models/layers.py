"""
共享网络层 —— 所有层都假定输入为 5D 张量 [N, T, H, W, C]（channels-last 风格）。
包含：基本层(FeatureGrid/Conv2d 包装)、高级块(MLP/ConvNeXt)、网格展开层(GridTrilinear3D)，
以及 norm/activation/block 的工厂函数 get_norm / get_activation / get_block。
"""
from .utils import *
from timm.models.layers import DropPath


"""
Basic Layers
"""
class Sin(nn.Module):
    """正弦激活函数（SIREN 风格）。"""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class FeatureGrid(nn.Module):
    """
    The module for storing learnable embedding.
    可学习嵌入网格：本质就是一个可训练的权重参数，forward 时直接返回自身。
    以 2D 形状存储，避免被 channels_last 内存格式转换。
    """
    def __init__(self, shape, init_scale):
        super().__init__()
        # 均匀初始化到 [-init_scale, init_scale]
        self.register_parameter(name='weight', param=nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=True))
        torch.nn.init.uniform_(self.weight, -init_scale, init_scale)

    def forward(self):
        return self.weight #.clone()


class FeatureBuffer(nn.Module):
    """
    The module for storing non-learnable embedding, e.g. for autoencoder.
    不可学习嵌入缓冲（requires_grad=False）：传入 x 时写入缓存，不传 x 时读出缓存。
    """
    def __init__(self, shape):
        super().__init__()
        self.register_parameter(name='weight', param=nn.Parameter(torch.zeros(shape), requires_grad=False))

    def forward(self, idx, x):
        if x is not None:
            output = self.weight[idx] = x.detach().to(self.weight.dtype)
        else:
            output = self.weight[idx]
        return output


class Conv2d(nn.Conv2d):
    """
    2D 卷积的 5D 包装：把 [N,T,H,W,C] 折叠成 [N*T,C,H,W] 做逐帧卷积，
    再还原回 [N,T,H,W,C]。网络内部统一使用该层而非 nn.Conv2d。
    """
    def forward(self, input):
        N, T, H, W, _ = input.shape
        x = input.view(N * T, H, W, -1).permute(0, 3, 1, 2)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = x.permute(0, 2, 3, 1).view(N, T, H, W, -1)
        return x


class Conv3d(nn.Conv3d):
    """3D 卷积的 5D 包装：在 [N,T,H,W,C] ↔ [N,C,T,H,W] 之间转置。"""
    def forward(self, input):
        x = input.permute(0, 4, 1, 2, 3)
        x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = x.permute(0, 2, 3, 4, 1)
        return x


"""
Advanced Layers
"""
class LinearBlock(nn.Module):
    """线性层块：Linear → Norm → Act（norm_first 时 Norm → Linear → Act）。用于 stem/head。"""
    def __init__(self, in_features: int, out_features: int,
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = get_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_activation(act)()

    def forward(self, input):
        x = self.linear(self.norm(input)) if self.norm_first else self.norm(self.linear(input))
        x = self.act(x)
        return x


class Conv2dBlock(nn.Module):
    """卷积层块：Conv2d(same padding) → Norm → Act。用于 stem。"""
    def __init__(self, in_features: int, out_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding='same', bias=bias)
        self.norm = get_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_activation(act)()

    def forward(self, input):
        x = self.conv(self.norm(input)) if self.norm_first else self.norm(self.conv(input))
        x = self.act(x)
        return x


class BlockBase(nn.Module):
    """
    残差块基类：统一处理 LayerScale、DropPath（随机深度）、掩码与残差连接。
    子类只需实现 block_forward。
    """
    def __init__(self, in_features, out_features, layerscale_init, droppath):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layerscale_init = layerscale_init

        # layer scale
        # 仅当输入输出同维且 layerscale_init>0 时启用逐通道缩放
        if self.in_features == self.out_features and layerscale_init > 0.:
            self.layerscale = torch.nn.Parameter(self.layerscale_init * torch.ones([self.out_features]), requires_grad=True)
        else:
            self.layerscale = None

        # Stochastic Depth
        # 随机深度（训练时随机跳过整个块）
        if self.in_features == self.out_features and droppath > 0.:
            self.droppath = DropPath(droppath)
        else:
            self.droppath = None

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}, layerscale_init={layerscale_init}'
        return s.format(**self.__dict__)

    def block_forward(self, input):
        raise NotImplementedError

    def forward(self, input, mask=None):
        """
        y = input + mask · DropPath(LayerScale · block_forward(input))
        （仅当输入输出同维时才有残差连接）
        """
        x = self.block_forward(input)

        if self.layerscale is not None:
            x = self.layerscale * x

        if self.droppath is not None:
            x = self.droppath(x)

        if mask is not None:
            x = mask * x  # 补丁模式下屏蔽 padding 区域，防止残差把无效特征带回主干

        if self.in_features == self.out_features:
            x = x + input

        return x


class MLPBlock(BlockBase):
    """纯 MLP 残差块（Transformer FFN 风格）：Norm → FC → Act → FC，带残差。"""
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.norm = get_norm(norm)(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)  # 升维（扩展率由 exps 控制）
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)  # 降维
        self.dropout2 = nn.Dropout(dropout)

    def block_forward(self, input):
        x = self.norm(input)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConvNeXtBlock(BlockBase):
    """
    ConvNeXt 风格残差块（论文默认 block-type）：DepthwiseConv → Norm → FC → GELU → FC，带残差。
    深度可分离卷积提供空间交互，MLP 提供通道交互，参数量远小于普通卷积块。
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.dconv = Conv2d(in_features, in_features, kernel_size, groups=in_features, padding='same', bias=bias)  # 深度可分离卷积
        self.norm = get_norm(norm)(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def block_forward(self, input):
        x = self.dconv(input)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConvNeXtBlockLessNorm(BlockBase):
    """
    ConvNeXtBlock 变体：仅当输入输出同维时才使用 Norm（通道变换的首个块省去归一化）。
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.dconv = Conv2d(in_features, in_features, kernel_size, groups=in_features, padding='same', bias=bias)
        self.norm = get_norm(norm if in_features == out_features else 'none')(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)
    
    def block_forward(self, input):
        x = self.dconv(input)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class GridTrilinear3D(nn.Module):
    """
    The module for mapping feature maps to a fixed size with trilinear interpolation.
    网格展开层：用三线性插值把（基础编码的）网格映射到固定输出尺寸。
    由于 F.interpolate 的 trilinear 在部分情况下结果不正确，这里只对时间维插值
    （要求空间维尺寸已经一致）。
    """
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        #x = F.interpolate(x, size=self.output_size, mode='trilinear', align_corners=self.align_corners)
        T, H, W, C = x.shape
        assert H == self.output_size[1] and W == self.output_size[2], 'F.interpolate has incorrect results in some cases, so use only temporal scale'
        # 把 (H,W,C) 折叠成通道维，仅对时间维做双线性插值（等价于仅时间维的三线性插值）
        x = x.view(1, 1, T, H * W * C)
        x = F.interpolate(x, size=(self.output_size[0], H * W * C), mode='bilinear', align_corners=self.align_corners)
        x = x.view(self.output_size + (C,))
        return x


"""
Utils
"""
def get_norm(norm, **kwargs):
    """归一化层工厂：none / layernorm / layernorm-no-affine（无仿射参数，网络默认）。"""
    if norm == "none":
        return nn.Identity
    elif norm == "layernorm":
        return partial(nn.LayerNorm, eps=1e-6, **kwargs)
    elif norm == "layernorm-no-affine":
        return partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6, **kwargs)
    else:
        raise NotImplementedError


def get_activation(activation):
    """激活函数工厂：none/relu/relu6/leaky_relu/gelu/gelu_fast/tanh/sigmoid/softplus/sin。"""
    if activation == "none":
        return nn.Identity
    elif activation == "relu":
        return nn.ReLU
    elif activation == "relu6":
        return nn.ReLU6
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "gelu_fast":
        return partial(nn.GELU, approximate='tanh')
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "sin":
        return Sin
    else:
        raise NotImplementedError


def get_block(type, **kwargs):
    """
    网络块工厂：根据 type 创建 stem/head/处理块。
    - linear_stem/conv_stem: 起始层（C1→C2）
    - linear_head: 输出头（norm_first=True，通常配 sigmoid 输出 RGB）
    - mlp: MLPBlock；convnext/convnext-lessnorm: ConvNeXt 风格块（论文默认 'convnext'）
    """
    if type == 'identity':
        return torch.nn.Identity()
    elif type == 'linear_stem':
        return LinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'conv_stem':
        return Conv2dBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], kernel_size=kwargs['kernel_size'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'linear_head':
        return LinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=True, bias=kwargs['bias'])
    elif type == 'mlp':
        return MLPBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                        act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                        layerscale_init=kwargs['layerscale'],
                        dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == 'convnext':
        return ConvNeXtBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                             kernel_size=kwargs['kernel_size'], act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                             layerscale_init=kwargs['layerscale'],
                             dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == 'convnext-lessnorm':
        return ConvNeXtBlockLessNorm(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                                     kernel_size=kwargs['kernel_size'], act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                                     layerscale_init=kwargs['layerscale'],
                                     dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    else:
        raise NotImplementedError