"""
HiNeRV 模型定义（论文核心网络）。

HiNeRV = 基础编码(HiNeRVEncoding, 多分辨率可学习特征网格) + 解码器(HiNeRVDecoder)。
解码器由 N 个 stage 组成，每个 stage 依次执行：
    1) HiNeRVUpsampler   —— 无参数三线性插值上采样（替代 NeRV 系列的子像素卷积，节省参数预算）
    2) PositionalEncoder —— 层级位置编码（论文核心创新）：向上采样后的特征图注入局部/时间局部网格编码
    3) 若干 ConvNeXt/MLP 块（get_block 构建）—— 处理特征
整体数据流：x = F_n(U_n(x) + F_enc(γ_n(i,j,t)))，与论文式一致。
"""
from .utils import *
from .layers import *
from .encoding import PositionalEncoder
from .upsample import FastTrilinearInterpolation, FastNearestInterpolation, crop_tensor_nthwc
from .patch_utils import *


def get_encoding_cfg(enc_type, depth, size, scale, **kwargs):
    """
    生成某一 stage 的(层级)位置编码配置 enc_cfg。

    参数:
        enc_type: 'base'(基础编码) 或 'upsample'(上采样层级编码)。
        depth:    当前 stage 编号，用于按 grid_depth_scale 逐级缩放网格尺寸。
        size:     当前 stage 上采样后的特征图尺寸 (T, H, W, C)。
        scale:    该 stage 的上采样倍率。
        kwargs:   命令行/配置文件中的 enc-* 参数（type/pe/grid_size 等）。
    返回:
        dict, 传给 PositionalEncoder 的 cfg；若编码类型为 'none' 则返回空 dict。
    """
    assert enc_type in ['base', 'upsample']
    assert (enc_type == 'base' and depth == 0) or (enc_type in ['upsample'] and depth >= 0)

    if kwargs['type'] == 'none':
        return {}

    # Positional
    Bt, Lt, Bs, Ls = kwargs['pe']
    Lt, Ls = int(Lt), int(Ls)

    # Grid
    if len(kwargs['grid_size']) == 4:
        # Learned/Local learned (full)
        T_grid, H_grid, W_grid, C_grid = kwargs['grid_size']
        if 'grid_depth_scale' in kwargs:
            T_grid_scale, H_grid_scale, W_grid_scale, C_grid_scale = kwargs['grid_depth_scale']
        else:
            T_grid_scale = H_grid_scale = W_grid_scale = C_grid_scale = 1.
    elif len(kwargs['grid_size']) == 2:
        # Local learned (temporal only)
        T_grid, C_grid = kwargs['grid_size']
        H_grid, W_grid = 1, 1
        if 'grid_depth_scale' in kwargs:
            T_grid_scale, C_grid_scale = kwargs['grid_depth_scale']
            H_grid_scale, W_grid_scale = 1, 1
        else:
            T_grid_scale = H_grid_scale = W_grid_scale = C_grid_scale = 1.
    else:
        raise NotImplementedError

    T_grid = max(int(T_grid * T_grid_scale ** depth), 1) if T_grid != -1 else size[0]
    H_grid = max(int(H_grid * H_grid_scale ** depth), 1) if H_grid != -1 else size[1]
    W_grid = max(int(W_grid * W_grid_scale ** depth), 1) if W_grid != -1 else size[2]
    C_grid = max(int(C_grid * C_grid_scale ** depth), 1) if C_grid != -1 else size[3]

    if len(kwargs['grid_size']) == 4:
        kwargs['grid_size'] = [T_grid, H_grid, W_grid, C_grid]
    elif len(kwargs['grid_size']) == 2:
        kwargs['grid_size'] = [T_grid, C_grid]
    else:
        raise NotImplementedError

    # All configs
    enc_cfg = {
        'type': kwargs['type'],
        # frequency encoding
        'Bt': Bt, 'Lt': Lt, 'Bs': Bs, 'Ls': Ls, 'no_t': kwargs['pe_no_t'],
        # grid encoding
        'grid_size': kwargs['grid_size'], 'grid_level': kwargs['grid_level'], 'grid_level_scale': kwargs['grid_level_scale'],
        'grid_init_scale': kwargs['grid_init_scale'], 'align_corners': kwargs['align_corners']
    }

    return enc_cfg


class HiNeRVEncoding(nn.Module):
    """
    HiNeRV 基础编码层（论文中的 γ_base）。
    使用多级可学习特征网格(FeatureGrid)：第 l 级网格尺寸逐级缩小、通道数逐级变化
    （由 grid_level_scale 控制，例如时间维减半时通道加倍），
    每级网格先经 GridTrilinear3D 在时间维上插值到完整帧数，再在通道维拼接，
    作为解码器 stem 之前的初始特征图。该设计继承自 FFNeRV 的多分辨率时间网格。
    """
    def __init__(self, size, channels, **kwargs):
        """构建多级特征网格。size=(T,H,W) 为第一层特征图尺寸, channels 为输出通道数。"""
        super().__init__()
        self.size = size
        self.channels = channels

        T, H, W = self.size
        C = self.channels

        # Grids
        self.grids = nn.ParameterList()
        self.grid_expands = nn.ModuleList()

        self.grid_level = kwargs['grid_level']
        self.grid_sizes = []

        T_grid, H_grid, W_grid, C_grid = kwargs['grid_size']
        T_scale, H_scale, W_scale, C_scale = kwargs['grid_level_scale']

        for i in range(self.grid_level):
            # 第 i 级网格：各维度按 grid_level_scale 的 i 次方缩小（-1 表示取满分辨率）
            T_i, H_i, W_i, C_i = int(T_grid / T_scale ** i), int(H_grid / H_scale ** i), int(W_grid / W_scale ** i), int(C_grid / C_scale ** i)
            self.grid_sizes.append((T_i, H_i, W_i, C_i))
            self.grids.append(FeatureGrid((T_i * H_i * W_i, C_i), init_scale=kwargs['grid_init_scale']))  # 可学习嵌入网格
            self.grid_expands.append(GridTrilinear3D((T, H, W)))  # 将网格插值展开到完整 (T,H,W) 尺寸

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """将线性/卷积层 bias 初始化为 0。"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def extra_repr(self):
        s = 'size={size}, channels={channels}, grid_level={grid_level}, grid_sizes={grid_sizes}'
        return s.format(**self.__dict__)

    def forward(self, idx: torch.IntTensor, idx_max: tuple[int, int, int], padding: tuple[int, int, int]):
        """
        根据补丁(patch)索引取出对应的编码特征图。

        Inputs:
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes.
            patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.

        Output:
            a tensor with shape [N, T, H, W, C]
        """
        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        # Compute the global voxels coordinates
        # 计算每个补丁覆盖的（含 padding 的）全局体素坐标及越界掩码
        patch_size = tuple(self.size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.size, padding=padding, clipped=True)
        px_idx_flat = (px_idx[0][:, :, None, None] * self.size[1] * self.size[2]
                       + px_idx[1][:, None, :, None] * self.size[2]
                       + px_idx[2][:, None, None, :]).view(-1)  # 展平成一维体素索引
        px_mask_flat = (px_mask[0][:, :, None, None, None]
                        * px_mask[1][:, None, :, None, None] 
                        * px_mask[2][:, None, None, :, None]).view(-1, 1)

        # Encode
        # 每级网格先插值展开到 (T,H,W)，再沿通道维拼接成完整编码图
        enc_splits = [self.grid_expands[i](self.grids[i]().view(self.grid_sizes[i])) for i in range(self.grid_level)]
        enc = torch.concat(enc_splits, dim=-1)

        # 用一维体素索引一次性取出所有补丁的特征，并用掩码将越界 padding 区域置零
        output = (px_mask_flat * torch.index_select(enc.view(self.size[0] * self.size[1] * self.size[2], self.channels), 0, px_idx_flat)) \
                    .view((idx.shape[0],) + patch_padded_size + (self.channels,))
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.channels,))

        return output


class HiNeRVUpsampler(nn.Module):
    """
    HiNeRV 上采样器：把"上采样 + 归一化 + 激活"封装在一起。
    默认使用无参数的三线性插值(trilinear, FastTrilinearInterpolation)，
    也支持 nearest / 子像素卷积(conv1x1/conv3x3/深度可分离版本)等上采样方式。
    论文用插值替代 NeRV 的子像素卷积，把参数预算留给更深的网络与层级编码。
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.channels = kwargs['channels']
        self.scale = kwargs['scale']
        self.upsample_type = kwargs['type']
        self.norm = get_norm(kwargs['norm'])(self.channels)
        self.act = get_activation(kwargs['act'])()

        # Layer
        # 根据 upsample_type 选择具体的上采样实现
        if self.upsample_type == 'trilinear':
            self.layer = FastTrilinearInterpolation(kwargs['config'])  # 无参数三线性插值（默认）
        elif self.upsample_type == 'nearest':
            self.layer = FastNearestInterpolation(kwargs['config'])  # 无参数最近邻插值
        elif self.upsample_type == 'conv1x1':
            assert self.scale[0] == 1
            self.layer = nn.Sequential(
                            nn.Conv2d(self.channels, self.channels * self.scale[1] * self.scale[2], 1, 1, padding='same'),
                            nn.PixelShuffle((self.scale[1]))
                        )
        elif self.upsample_type == 'conv1x1':
            assert self.scale[0] == 1 and self.scale[1] == self.scale[2]
            self.layer = nn.Sequential(
                            nn.Conv2d(self.channels, self.channels * self.scale[1] * self.scale[2], 1, 1, padding='same'),
                            nn.PixelShuffle((self.scale[1]))
                        )
        elif self.upsample_type == 'conv3x3':
            assert self.scale[0] == 1 and self.scale[1] == self.scale[2]
            self.layer = nn.Sequential(
                            nn.Conv2d(self.channels, self.channels * self.scale[1] * self.scale[2], 3, 1, padding='same'),
                            nn.PixelShuffle((self.scale[1]))
                        )
        elif self.upsample_type == 'dconv1x1':
            assert self.scale[0] == 1 and self.scale[1] == self.scale[2]
            self.layer = nn.Sequential(
                            nn.Conv2d(self.channels, self.channels * self.scale[1] * self.scale[2], 1, 1, padding='same', groups=self.channels),
                            nn.PixelShuffle((self.scale[1]))
                        )
        elif self.upsample_type == 'dconv3x3':
            assert self.scale[0] == 1 and self.scale[1] == self.scale[2]
            self.layer = nn.Sequential(
                            nn.Conv2d(self.channels, self.channels * self.scale[1] * self.scale[2], 3, 1, padding='same', groups=self.channels),
                            nn.PixelShuffle((self.scale[1]))
                        )
        else:
            raise ValueError

    def extra_repr(self):
        s = 'scale={scale}, upsample_type={upsample_type}'
        return s.format(**self.__dict__)

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int], patch_mode: bool=True,
                mask: Optional[torch.Tensor]=None):
        """
        前向传播：将输入特征图按 scale 上采样，再裁剪到目标补丁大小(padding 内)。
        During the forward pass, the input tensor will be upscale by the 'scale' factor, then a cropping will be applied to reduce the size to 'output_size'.

        Inputs:
            x: input tensor with shape [N, T1, H1, W1, C]
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes.
            size: list of 3 ints. Represents the size of the fulle video. It does not have to be the same as the input size, as the input can be a patch from the full video.
            scale: list of 3 ints. Represents the scale factor. This will be used to compute the output size.
            padding: list of 3 ints. Represents the padding size. This will be used to compute the output size.
            patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.
            mask: mask tensor with shape [N, T1, H1, W1, C]. If not None, it will be multiplied to the output.

        Output:
            a tensor with shape [N, T2, H2, W2, C]
        """
        assert x.ndim == 5, x.shape
        assert idx.ndim == 2 and idx.shape[1] == 3, idx.shape
        assert len(idx_max) == 3
        assert len(scale) == 3
        assert len(size) == 3
        assert len(padding) == 3

        N, T_in, H_in, W_in, C = x.shape
        T_scale, H_scale, W_scale = int(T_in * scale[0]), int(H_in * scale[1]), int(W_in * scale[2])
        T_out, H_out, W_out = tuple(size[d] // idx_max[d] + 2 * padding[d] for d in range(3))
        assert (T_out - T_in * scale[0]) % 2 == (H_out - H_in * scale[1]) % 2 == (W_out - W_in * scale[2]) % 2 == 0, 'Under this configuration, padding is not symmetric and can cause problems!'

        x = self.norm(x)  # 上采样前先做 LayerNorm（无仿射参数）

        if self.upsample_type in ['trilinear', 'nearest']:
            # 插值上采样：内部已包含裁剪到目标补丁尺寸的逻辑
            x = self.layer(x, idx, idx_max, size, scale, padding, patch_mode)
            assert_shape(x, (N, T_out, H_out, W_out, C))
        elif 'conv' in self.upsample_type:
            # 卷积+子像素卷积(PixelShuffle)上采样：逐帧做 2D 卷积后重排像素，再中心裁剪
            x = x.view(N * T_in, H_in, W_in, C).permute(0, 3, 1, 2)
            x = self.layer(x)
            x = x.permute(0, 2, 3, 1).view(N, T_scale, H_scale, W_scale, C)
            x = crop_tensor_nthwc(x, size=(T_out, H_out, W_out))
            assert_shape(x, (N, T_out, H_out, W_out, C))
            if mask is not None:
                x = mask * x # For conv layers, mask should be applied  # 卷积会把 padding 区域的值扩散进有效区，故需用掩码置零
        else:
            raise NotImplementedError

        x = self.act(x)  # 上采样后的激活（默认无激活）

        return x


class HiNeRVDecoder(nn.Module):
    """
    HiNeRV 解码器（论文主体网络）：stem → N 个 stage(上采样器+层级编码+ConvNeXt块) → head。
    每个 stage i 计算 X_n = F_n(U_n(X_{n-1}) + F_enc(γ_n(i,j,t)))，
    空间分辨率按 scales 逐级放大（如 5,3,2,2），通道数按 channels_reduce 逐级减半，
    最终经 head(线性层+sigmoid) 输出 RGB 帧。补丁(patch)训练时通过 padding 保证
    帧模式与补丁模式可以无缝切换（统一表示）。
    """
    def __init__(self, logger, input_size: tuple[int, int, int]=(600, 9, 16), input_channels: int=128, 
                 output_size: tuple[int, int, int]=(600, 1080, 1920), output_channels: int=3,
                 channels: int=256, channels_reduce: float=2.0, channels_reduce_base: int=1, channels_min: int=1,
                 depths: list=[3, 3, 3, 3], exps: list=[4, 4, 4, 4], kernels: list=[1, 1, 1, 1],
                 scales: list=[[1, 5, 5], [1, 4, 4], [1, 3, 3], [1, 2, 2]],
                 paddings: list=[[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
                 stem_kernels: int=1, stem_paddings: tuple=(0, 0, 0),
                 block_cfg: dict={}, upsample_cfg: dict={}, enc_cfg: dict={}, stem_cfg: dict={}, head_cfg: dict={}):
        super().__init__()
        assert isinstance(input_size, (tuple, list))
        assert isinstance(output_size, (tuple, list))
        assert isinstance(depths, (tuple, list))
        assert isinstance(exps, (tuple, list))
        assert isinstance(scales, (tuple, list)) and all([isinstance(s, (tuple, list)) for s in scales])
        assert isinstance(paddings, (tuple, list)) and all([isinstance(p, (tuple, list)) for p in paddings])
        assert len(depths) == len(exps) == len(kernels) == len(scales) == len(paddings)
        assert all(input_size[d] * math.prod([scale[d] for scale in scales]) == output_size[d] for d in range(3)), \
                f'input_size {input_size} does no match output_size {output_size} and scales {scales}'

        self.input_size = input_size
        self.input_channels = input_channels
        self.ouput_size = output_size
        self.output_channels = output_channels

        self.channels = channels
        self.channels_reduce = channels_reduce
        self.channels_reduce_base = channels_reduce_base
        self.channels_min = channels_min

        self.depths = depths
        self.exps = exps
        self.kernels = kernels
        self.scales = [tuple(scales[i]) for i in range(len(scales))]
        self.paddings = [tuple(paddings[i]) for i in range(len(paddings))]

        self.stem_kernels = stem_kernels
        self.stem_paddings = stem_paddings

        self.min_patch_size = tuple(np.prod(np.array(self.scales), axis=0).tolist())  # 所有上采样倍率之积 = 最小补丁尺寸

        self.T, self.H, self.W = self.ouput_size

        logger.info(f'HiNeRV:')

        # Stem
        # 起始卷积层：把基础编码的通道数映射到网络主通道数
        _stem_cfg = cfg_override(stem_cfg, C1=self.input_channels, C2=self.channels,
                                 Ch=max(self.input_channels, self.channels), 
                                 kernel_size=self.stem_kernels)
        self.stem = get_block(**_stem_cfg)

        # HiNeRV blocks
        # N 个 stage，每个 stage 由 [上采样器, 层级编码器, depths[i] 个卷积/MLP 块] 组成
        self.blocks = nn.ModuleList()

        for i, scale in enumerate(self.scales):
            # 计算该 stage 的输入/输出尺寸与通道数：分辨率放大 scale 倍，通道数按 channels_reduce 缩减
            if i == 0:
                T1, H1, W1 = (self.input_size[d] for d in range(3))
                T2, H2, W2 = T1 * scale[0], H1 * scale[1], W1 * scale[2]
                C1 = self.channels
                C2 = max(self.channels, self.channels_min)
            else:
                T1, H1, W1 = T2, H2, W2
                T2, H2, W2 = T1 * scale[0], H1 * scale[1], W1 * scale[2]
                C1 = C2
                C2 = max(divide_to_multiple(C2, self.channels_reduce, self.channels_reduce_base), self.channels_min)

            logger.info(f'     Stage {i + 1}:  T1 - {T1}  H1 - {H1}  W1 - {W1}  C1 - {C1}')
            logger.info(f'                     T2 - {T2}  H2 - {H2}  W2 - {W2}  C2 - {C2}')
            logger.info(f'                     Depth - {self.depths[i]}  Exp - {self.exps[i]}  Kernel - {self.kernels[i]}')
            logger.info(f'                     Scale - {self.scales[i]}  Padding - {self.paddings[i]}')

            # Block
            # 新建该 stage 的层列表
            self.blocks.append(nn.ModuleList())

            # Upsampler
            # 该 stage 的上采样器（三线性插值）
            _upsample_cfg = cfg_override(upsample_cfg, channels=C1, scale=scale)
            self.blocks[-1].append(HiNeRVUpsampler(**_upsample_cfg))

            # Encoder
            # 该 stage 的层级位置编码器（temp_local_grid 等），编码尺寸按 stage 深度逐级缩放
            _enc_cfg = get_encoding_cfg('upsample', i, size=(T2, H2, W2, C1), scale=scale, **enc_cfg)
            self.blocks[-1].append(PositionalEncoder(scale=scale, channels=C1, cfg=_enc_cfg))

            # Conv/MLP Block
            # depths[i] 个 ConvNeXt/MLP 块，第一块负责 C1→C2 通道变换，其后保持 C2
            assert self.depths[i] > 0
            for j in range(self.depths[i]):
                _block_cfg = cfg_override(block_cfg, C1=C1 if j == 0 else C2, C2=C2,
                                          Ch=int(C2 * self.exps[i]),
                                          kernel_size=self.kernels[i])
                self.blocks[-1].append(get_block(**_block_cfg))

        # Heads
        # 输出头：线性层 + sigmoid，输出 3 通道 RGB
        logger.info(f'     Output channels: {self.output_channels}')
        _head_cfg = cfg_override(head_cfg, C1=C2, C2=self.output_channels,
                                    Ch=max(C2, self.output_channels),
                                    kernel_size=1)
        self.head = get_block(**_head_cfg)

        # Initialization
        # 初始化：stem/blocks/head 的 bias 置零
        self.stem.apply(self._init_stem_blocks)
        self.blocks.apply(self._init_stem_blocks)
        self.head.apply(self._init_heads)

    def _init_stem_blocks(self, m):
        """stem 与 blocks 中线性/卷积层 bias 置零。"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_heads(self, m):
        """head 中线性/卷积层 bias 置零。"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def extra_repr(self):
        s = 'input_size={input_size}, input_channels={input_channels}, ouput_size={ouput_size}, output_channels={output_channels}'
        return s.format(**self.__dict__)

    def get_input_padding(self, patch_mode: bool=True):
        """返回网络输入(stem)所需的补丁 padding：帧模式只需时间维 padding。"""
        return self.stem_paddings if patch_mode else (self.stem_paddings[0], 0, 0)

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int], patch_mode: bool=True):
        """ 
        解码器前向传播：stem → 各 stage(上采样+层级编码+卷积块) → head，输出重建的视频补丁。
        Inputs:
            x: input tensor with shape [N, T1, H1, W1, C]
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes.
            patch_mode: if True, the input is a patch from the full video, and the faster implementation will be used.

        Output:
            a tensor with shape [N, T2, H2, W2, C]
        """
        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        # initial configs
        # 初始化各尺寸记录：v_size_* 为整段视频在当前 stage 的尺寸，p_size_* 为单个补丁的尺寸
        padding = self.stem_paddings if patch_mode else (0, 0, 0)
        v_size_in = v_size_out = self.input_size
        p_size_in = p_size_out = tuple(v_size_out[d] // idx_max[d] for d in range(3))
        p_size_out_padded = tuple(p_size_out[d] + 2 * padding[d] for d in range(3))

        if patch_mode:
            # 计算补丁（含 padding）对应的全局像素范围掩码，用于屏蔽越界区域
            _, px_mask = compute_pixel_idx_3d(idx, idx_max, v_size_out, padding, clipped=False, return_mask=True)
            px_mask_3d = px_mask[0][:, :, None, None, None] \
                            * px_mask[1][:, None, :, None, None] \
                            * px_mask[2][:, None, None, :, None]
        else:
            px_mask_3d = None

        # Check input
        assert_shape(x, (idx.shape[0],) + p_size_out_padded + (self.input_channels,))

        x = self.stem(x)
        if px_mask_3d is not None:
            x = x * px_mask_3d  # stem 卷积后重新屏蔽 padding 区域

        for i, block in enumerate(self.blocks):
            # Update configs
            # 更新该 stage 的尺寸/补丁/padding 配置
            padding = self.paddings[i] if patch_mode else (self.paddings[0][0], 0, 0)
            scale = self.scales[i]
            v_size_in = v_size_out
            v_size_out = tuple(int(v_size_in[d] * scale[d]) for d in range(3))
            p_size_in = p_size_out
            p_size_in_padded = p_size_out_padded
            p_size_out = tuple(int(p_size_in[d] * scale[d]) for d in range(3))
            p_size_out_padded = tuple(p_size_out[d] + 2 * padding[d] for d in range(3))

            assert all(p_size_in_padded[d] * scale[d] >= p_size_out_padded[d] for d in range(3)), 'the input padding is too small'

            # Compute mask
            # masking is only needed if kernel size > 1
            # 只有卷积核 >1 时才需要掩码（防止 padding 区域污染有效特征）
            if patch_mode:
                _, px_mask = compute_pixel_idx_3d(idx, idx_max, v_size_out, padding, clipped=False, return_mask=True)
                px_mask_3d = px_mask[0][:, :, None, None, None] \
                                * px_mask[1][:, None, :, None, None] \
                                * px_mask[2][:, None, None, :, None]
            else:
                px_mask_3d = None

            # Run block
            # 依次执行该 stage 的三层结构：上采样器 → 层级编码器 → 卷积/MLP 块
            for _, layer in enumerate(block):
                if isinstance(layer, HiNeRVUpsampler):
                    x = layer(x, idx=idx, idx_max=idx_max,
                              size=v_size_out,
                              scale=scale,
                              padding=padding,
                              patch_mode=patch_mode,
                              mask=px_mask_3d)
                elif isinstance(layer, PositionalEncoder):
                    x = layer(x, idx=idx, idx_max=idx_max,
                              size=v_size_out, 
                              scale=scale,
                              padding=padding)
                else:
                    x = layer(x, px_mask_3d)

            assert_shape(x, (idx.shape[0],) + p_size_out_padded + (x.shape[-1],))


        x = self.head(x)  # 输出头：映射到 RGB
        y = crop_tensor_nthwc(x, p_size_out)  # 裁掉 padding，只保留有效补丁区域
        assert_shape(y, (idx.shape[0],) + p_size_out + (self.output_channels,))

        return y


class HiNeRV(nn.Module):
    """
    HiNeRV 顶层模型：基础编码 + 解码器。
    训练时按补丁(patch)模式前向；评估时若设置 eval_patch_size，
    会把帧索引切分成多个补丁索引强制走补丁模式（统一表示，无需重新训练）。
    同时定义了剪枝/量化的模块前缀：
        no_prune_prefix: 不参与剪枝的模块（编码相关，参数量很小）
        bitstream_prefix: 需要写入码流(存储/传输)的模块
    """
    def __init__(self, encoding, decoder, eval_patch_size):
        super().__init__()

        # Model
        self.encoding = encoding
        self.decoder = decoder
        self.eval_patch_size = eval_patch_size

        # Encoding only account for small amount of parameters, so simply skip them.
        # 编码部分参数量很小，直接跳过剪枝
        self.no_prune_prefix = ('encoding',) + tuple(k for k, v in self.named_modules() if isinstance(v, PositionalEncoder))
        self.no_quant_prefix = ()

        # The bitstream is the part of data that will be transmitted/stored.
        self.bitstream_prefix = ('encoding', 'decoder')

    def forward(self, input):
        """
        模型前向：编码 → 解码 → 整理输出形状。
        Inputs:
            a dictionaty with {
                idx: patch index tensor with shape [N, 3]
                idx_max: list of 3 ints. Represents the range of patch indexes.
                video_size: list of 3 ints. Represents the size of the full video.
                patch_size: list of 3 ints. Represents the size of the patch.
            }
        Output:
            a tensor with shape [N, C, T, H, W]
        }
        """
        assert all((input['patch_size'][d] % self.decoder.min_patch_size[d] == 0) for d in range(3))
        assert self.eval_patch_size is None or all(input['video_size'][d] % self.eval_patch_size[d] == 0 for d in range(3))

        # Config
        # 评估时强制切分成 eval_patch_size 大小的补丁（帧→补丁索引转换），训练时自动选择帧/补丁模式
        if not self.training and self.eval_patch_size is not None :
            # Force using patch mode for evaluation
            idx_max = tuple(input['video_size'][d] // self.eval_patch_size[d] for d in range(3))
            idx = vidx_to_pidx(input['idx'], input['idx_max'], idx_max)
            patch_mode = True
        else:
            # Auto choose frame/patch mode
            idx_max = input['idx_max']
            idx = input['idx']
            patch_mode = self.training

        # Compute output
        # 基础编码取特征 → 解码器逐级上采样重建
        input_padding = self.decoder.get_input_padding(patch_mode=patch_mode)
        output = self.encoding(idx, idx_max, padding=input_padding)
        output = self.decoder(output, idx, idx_max, patch_mode=patch_mode)

        # Reshape output
        # 评估模式下把各补丁输出重新拼回完整帧
        if not self.training and self.eval_patch_size is not None :
            output = patch_to_video(output, input['patch_size'])
        output = output.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)

        return output

    def get_num_parameters(self):
        """统计各部分参数量（All/Decoder/Model/Encoding/Upsampling Encoding）。"""
        base_encoding_param = sum([v.numel() for _, v in self.encoding.state_dict().items()])
        decoder_param = sum([v.numel() for _, v in self.decoder.state_dict().items()])
        upsample_encoding_param = 0
        for block in self.decoder.blocks:
            for layer in block:
                # 注: 原代码此处判断的是 block 而非 layer，因此该统计项实际恒为 0（保持原逻辑，未修改）
                if isinstance(block, PositionalEncoder):
                    upsample_encoding_param += sum([v.numel() for _, v in layer.state_dict().items()])
        model_param = decoder_param - upsample_encoding_param
        return {
            'All': decoder_param + base_encoding_param,
            'Decoder': decoder_param,
            'Model': model_param,
            'Encoding': base_encoding_param,
            'Upsampling Encoding': upsample_encoding_param
        }


def build_encoding(args, logger, size, channels):
    """根据命令行参数构建基础编码层 HiNeRVEncoding。"""
    # Encoding config
    cfg = {}
    cfg['size'] = size
    cfg['channels'] = channels
    cfg['grid_size'] = args.base_grid_size
    cfg['grid_level'] = args.base_grid_level
    cfg['grid_level_scale'] = args.base_grid_level_scale
    cfg['grid_init_scale'] = args.base_grid_init_scale

    # Bulid encoding
    logger.info(f'Building HiNeRV Encoding with cfg: {cfg}')
    encoding = HiNeRVEncoding(**cfg)

    return encoding


def build_decoder(args, logger, input_size, input_channels, output_size, output_channels):
    """根据命令行参数组装解码器配置并构建 HiNeRVDecoder。"""
    # Network config
    assert len(args.depths) == len(args.exps) == len(args.kernels) == len(args.scales_t) == len(args.scales_hw) 
    cfg = {}
    cfg['input_size'] = input_size
    cfg['input_channels'] = input_channels
    cfg['output_size'] = output_size
    cfg['output_channels'] = output_channels

    cfg['channels'] = args.channels
    cfg['channels_reduce'] = args.channels_reduce
    cfg['channels_reduce_base'] = args.channels_reduce_base
    cfg['channels_min'] = args.channels_min

    cfg['depths'] = args.depths
    cfg['exps'] = args.exps    
    cfg['scales'] = [[scale_t, scale_hw, scale_hw] for scale_t, scale_hw in zip(args.scales_t, args.scales_hw)]
    cfg['stem_kernels'] = args.stem_kernels
    cfg['kernels'] = args.kernels

    if tuple(args.paddings) == (-1, -1, -1):
        # 未显式指定 padding 时，自动根据卷积核/上采样倍率反推各 stage 所需的补丁 padding
        assert tuple(args.stem_paddings) == (-1, -1, -1), 'both padding must be set/not set at the same time'
        paddings = compute_paddings(output_patchsize=(math.prod(args.scales_t), math.prod(args.scales_hw), math.prod(args.scales_hw)),
                                      scales=cfg['scales'], kernel_sizes=tuple((0, k, k) for k in cfg['kernels']),
                                      depths=cfg['depths'], resize_methods=args.upsample_type)
        cfg['stem_paddings'] = tuple(paddings[0][d] + (0 if d == 0 else (cfg['stem_kernels'] - 1) // 2) for d in range(3))
        cfg['paddings'] = paddings[1:]
    else:
        assert tuple(args.stem_paddings) != (-1, -1, -1), 'both padding must be set/not set at the same time'
        assert all(p >= 0 for p in args.stem_paddings) and all(p >= 0 for p in args.paddings)
        cfg['stem_paddings'] = args.stem_paddings
        cfg['paddings'] = [args.paddings for _ in range(len(args.depths))]

    # Stem/Blocks/Head/Upsampling/Encoding config
    # 收集所有 stem_*/head_*/block_*/enc_*/upsample_* 前缀的参数，分组为各子模块的配置
    for prefix in ['block', 'stem', 'head', 'enc', 'upsample']:
        cfg[f'{prefix}_cfg'] = {}
        for k, v in vars(args).items():
            if k.startswith(f'{prefix}_'):
                cfg[f'{prefix}_cfg'][k.replace(f'{prefix}_', f'')] = v

    # Bulid decoder
    logger.info(f'Building HiNeRV Decoder with cfg: {cfg}')
    decoder = HiNeRVDecoder(logger, **cfg)

    return decoder


def build_model(args, logger, input):
    """
    模型总构建入口：由 hinerv_main.py 在创建任务后调用。
    根据视频尺寸推导第一层特征图尺寸(base_size)与基础编码通道数(base_channels)，
    再依次构建 encoding、decoder，最后包装成 HiNeRV。
    """
    # Set default configurations
    # The first level feature map size
    # 第一层特征图尺寸 = 视频尺寸 / 所有上采样倍率之积
    args.base_size = (input['video_size'][0] // int(math.prod(args.scales_t)) if args.base_size[0] == -1 else args.base_size[0],
                     input['video_size'][1] // int(math.prod(args.scales_hw)) if args.base_size[1] == -1 else args.base_size[1],
                     input['video_size'][2] // int(math.prod(args.scales_hw)) if args.base_size[2] == -1 else args.base_size[2])
    # 基础编码通道数 = 各级网格通道数之和（多级网格拼接）
    args.base_channels = sum([int(args.base_grid_size[3] // args.base_grid_level_scale[3] ** i) for i in range(args.base_grid_level)])

    # The first level grid size
    # 网格尺寸缺省值：T/H/W 默认取满分辨率，C 默认 8
    args.base_grid_size[0] = args.base_grid_size[0] if args.base_grid_size[0] != -1 else args.base_size[0]
    args.base_grid_size[1] = args.base_grid_size[1] if args.base_grid_size[1] != -1 else args.base_size[1]
    args.base_grid_size[2] = args.base_grid_size[2] if args.base_grid_size[2] != -1 else args.base_size[2]
    args.base_grid_size[3] = args.base_grid_size[3] if args.base_grid_size[3] != -1 else 8

    assert args.base_size[0] == input['video_size'][0] // int(math.prod(args.scales_t))
    assert args.base_size[1] == input['video_size'][1] // int(math.prod(args.scales_hw))
    assert args.base_size[2] == input['video_size'][2] // int(math.prod(args.scales_hw))
    
    # Building encoding and decoder
    encoding = build_encoding(args, logger, args.base_size, args.base_channels)
    decoder = build_decoder(args, logger, args.base_size, args.base_channels, input['video_size'], 3)

    # Building model wrapper
    wrapper_cfg = {}
    wrapper_cfg['eval_patch_size'] = args.eval_patch_size

    logger.info(f'Building HiNeRV with cfg: {wrapper_cfg}')

    model = HiNeRV(encoding, decoder, **wrapper_cfg)

    return model


def set_args(parser):
    """注册 HiNeRV 特有的命令行参数（网络结构/网格/编码/上采样等）。"""
    # HiNeRV architecture parameters
    group = parser.add_argument_group('Model specific parameters')

    # Model complexity
    group.add_argument(f'--channels', type=int, default=256, help='channels of HiNeRV')
    group.add_argument(f'--channels-reduce', type=float, default=2.0, help='channels reduction factor of HiNeRV')
    group.add_argument(f'--channels-reduce-base', type=int, default=1, help='the base number which force to be a factor of channels of HiNeRV')
    group.add_argument(f'--channels-min', type=int, default=1, help='min channels of HiNeRV')
    group.add_argument(f'--depths', type=int, nargs='+', default=[3, 3, 3, 1], help='depths of HiNeRV')
    group.add_argument(f'--exps', type=float, nargs='+', default=[4., 4., 4., 1.], help='expansion ratio of HiNeRV')
    group.add_argument(f'--stem-kernels', type=int, default=1, help='stem kernel sizes of HiNeRV')
    group.add_argument(f'--kernels', type=int, nargs='+', default=[1, 1, 1, 1], help='kernel sizes of HiNeRV')
    group.add_argument(f'--scales-t', type=int, nargs='+', default=[1, 1, 1, 1], help='scaling factors in temporal dimension of HiNeRV')
    group.add_argument(f'--scales-hw', type=int, nargs='+', default=[5, 4, 3, 2], help='scaling factors in spatial dimensions of HiNeRV')
    group.add_argument(f'--stem-paddings', type=int, nargs='+', default=[0, 1, 1], help='stem paddings of HiNeRV')
    group.add_argument(f'--paddings', type=int, nargs='+', default=[0, 1, 1], help='paddings of HiNeRV')

    # Base
    for prefix in ['base']:
        group.add_argument(f'--{prefix}-size', type=int, nargs='+', default=[600, 9, 16], help='encoding size of HiNeRV')
        group.add_argument(f'--{prefix}-grid-size', type=int, nargs='+', default=[1, 1, 1, 1], help=f'{prefix.title()} grid dimensions (T/H/W/C). The dimension of highest resolution for the multilevel case.')
        group.add_argument(f'--{prefix}-grid-level', type=int, default=3, help=f'{prefix.title()} number of grid levels')
        group.add_argument(f'--{prefix}-grid-level-scale', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0], help=f'{prefix.title()} grid scaling ratio (T/H/W/C) with the grid level.')
        group.add_argument(f'--{prefix}-grid-init-scale', type=float, default=1e-3, help=f'{prefix.title()} grid initialization scaling factor.')

    # Blocks
    default_block = {'block': 'mlp', 'stem': 'conv_stem', 'head': 'linear_head'}
    default_norm = {'block': 'layernorm-no-affine', 'stem': 'none', 'head': 'none'}
    default_act = {'block': 'gelu', 'stem': 'none', 'head': 'sigmoid'}
    default_bias = {'block': False, 'stem': True, 'head': True}
    for prefix in ['block', 'stem', 'head']:
        group.add_argument(f'--{prefix}-type', type=str, default=default_block[prefix], help=f'type of {prefix}')
        group.add_argument(f'--{prefix}-norm', type=str, default=default_norm[prefix], help=f'type of normalization for {prefix}s')
        group.add_argument(f'--{prefix}-act', type=str, default=default_act[prefix], help=f'type of activation for {prefix}s')
        group.add_argument(f'--{prefix}-layerscale', type=float, default=0.0, help=f'initial layerscale for {prefix}s (0.: disable)')
        group.add_argument(f'--{prefix}-dropout', type=float, default=0.0, help=f'dropout rate for {prefix}s (0.: disable)')
        group.add_argument(f'--{prefix}-droppath', type=float, default=0.0, help=f'droppath rate for {prefix}s (0.: disable)')
        group.add_argument(f'--{prefix}-bias', type=str_to_bool, default=default_bias[prefix], help=f'Use bias in blocks')

    # Upsampling Encoders
    group.add_argument('--enc-type', type=str, default='normalized+temp_local_grid', help='type of upsampling encoding')
    group.add_argument('--enc-align-corners', type=str_to_bool, default=False, help='compute upsampling coordinate with align corners.')
    group.add_argument('--enc-pe', type=float, nargs='+', default=[1.2, 60, 1.2, 60], help='Frequency Encoding parameters (Bt/Lt/Bs/Ls).')
    group.add_argument('--enc-pe-no-t', type=str_to_bool, default=False, help='Dont use temporal dimension for enc enocoding')
    group.add_argument('--enc-grid-size', type=int, nargs='+', default=[1, 1, 1, 1], help='grid dimensions (T/H/W/C). The dimension of highest resolution for the multilevel case.')
    group.add_argument('--enc-grid-level', type=int, default=3, help='number of grid levels')
    group.add_argument('--enc-grid-level-scale', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0], help='grid scaling ratio (T/H/W/C) with the grid level.')
    group.add_argument('--enc-grid-init-scale', type=float, default=1e-3, help='grid initialization scaling factor.')
    group.add_argument('--enc-grid-depth-scale', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0], help='grid scaling ratio (T/H/W/C) with the stage.')

    # Upsample layers
    group.add_argument('--upsample-type', type=str, default='trilinear', help='Upsampling method of HiNeRV')
    group.add_argument('--upsample-config', type=str, default='matmul-th-w', help='Upsampling method config of HiNeRV')
    group.add_argument('--upsample-norm', type=str, default='layernorm-no-affine', help=f'type of normalization for decoder')
    group.add_argument('--upsample-act', type=str, default='none', help=f'type of activation for decoder')

    # Wrapper
    group.add_argument('--eval-patch-size', type=int, nargs='+', default=None, help='patch size during evalution for HiNeRV')