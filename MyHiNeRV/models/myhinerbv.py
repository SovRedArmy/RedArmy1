"""
MyHiNeRV 模型定义：在 HiNeRV 基础上的改进版，用于视频压缩实验（MyHiNeRV）。

相对 HiNeRV 的改动：
1) CRSPE 每帧内容嵌入（借鉴 Scene Matters, ICCV 2023 的 context-related spatial embedding
   思想）：为每一帧学一个全局内容向量 [T, C_f]，与基础编码（多级时间网格）在通道维拼接后
   注入 stem。HiNeRV 的 base grid 只提供低分辨率的空间-时间内容，CRSPE 补充帧级全局内容
   （亮度风格/场景状态等）。支持用 GT 下采样特征做内容初始化（init_from_content），
   该嵌入量化后写入码流（参数量 T*C_f，码率代价可忽略）。
2) 多尺度投影头（借鉴 FaNeRV 的多分辨率监督 / MSNeRV 的多尺度特征利用）：
   解码器每个 stage（除最后一级，最后一级由主输出头负责）的输出特征接一个轻量
   Linear+sigmoid 头，投影成该分辨率的低分辨率小视频，与 max-pooling 下采样的 GT 金字塔
   分别算损失——浅层先学全局结构、深层专攻高频细节。投影头仅在训练期使用，不进码流。
3) 光流伪标签约束头（借鉴 Scene Matters 的 SFCM，但模型本身不含显式光流估计）：
   在最后一级特征上接 flow head（前向/后向光流各 2 通道）与置信度 head（2 通道 softmax，
   可学习置信权重用于过滤 RAFT 伪标签噪声）。二者仅在训练期使用，不进码流、零码率。
4) 前向提供 forward_with_aux，返回 (output, aux)；aux 含多尺度预测、光流/置信度预测，
   供任务层计算新增损失。aux 模块统一挂在 decoder.aux / wrapper.aux 下，
   不参与剪枝/量化、不写入码流。
"""
from .utils import *
from .layers import *
from .encoding import PositionalEncoder
from .upsample import FastTrilinearInterpolation, FastNearestInterpolation, crop_tensor_nthwc
from .patch_utils import *
from .hinerv import HiNeRVDecoder, HiNeRVUpsampler, build_encoding


class CRSPE(nn.Module):
    """
    Content-Related per-frame SPatial Embedding（每帧内容嵌入）。
    以 [T, C_f] 的可学习表存储每帧全局内容向量；forward 按补丁的时间坐标取回，
    越界（时间 padding）位置取第 0 帧嵌入并乘 0 掩码。
    权重以 2D 形状存储（避免 channels_last 转换），量化/熵编码与普通权重一致。
    """
    def __init__(self, num_frames, channels, init_scale=1e-2):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels
        self.register_parameter(name='weight', param=nn.Parameter(torch.zeros((num_frames, channels), dtype=torch.float32), requires_grad=True))
        torch.nn.init.uniform_(self.weight, -init_scale, init_scale)

    def extra_repr(self):
        s = 'num_frames={num_frames}, channels={channels}'
        return s.format(**self.__dict__)

    @torch.no_grad()
    def init_from_content(self, feats: torch.Tensor, seed: int = 0):
        """
        用 GT 内容特征初始化每帧嵌入（内容感知初始化，仅影响训练起点，解码端无需知道）。
        feats: [T, D] 任意维度的每帧内容描述（如 GT 下采样+随机投影后的特征）。
        D != C_f 时用固定种子的高斯随机投影降到 C_f；随后 L2 归一化并乘 init_scale。
        """
        assert feats.ndim == 2 and feats.shape[0] == self.num_frames
        feats = feats.float().cpu()
        D = feats.shape[1]
        if D != self.channels:
            g = torch.Generator().manual_seed(seed)
            proj = torch.randn(D, self.channels, generator=g) / math.sqrt(D)
            feats = feats @ proj
        feats = F.normalize(feats, dim=1, eps=1e-8)
        self.weight.copy_(feats * (1. / math.sqrt(self.channels)))

    def forward(self, t_idx: torch.Tensor, t_mask: Optional[torch.Tensor] = None):
        """
        t_idx: [N, T]（clamp 过的）帧坐标；t_mask: [N, T] 有效位掩码（padding 越界为 0）。
        返回 [N, T, C_f]。
        """
        emb = self.weight[t_idx]  # [N, T, C_f]
        if t_mask is not None:
            emb = emb * t_mask[:, :, None].to(emb.dtype)
        return emb


class AuxScaleHead(nn.Module):
    """多尺度投影头：Linear(C→3) + sigmoid，把 stage 特征投影成该分辨率的 RGB 小视频。"""
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 3, bias=True)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor):
        # x: [N, T, H, W, C] -> [N, 3, T, H, W]
        y = torch.sigmoid(self.linear(x))
        return y.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.contiguous_format)


class MyHiNeRVDecoder(HiNeRVDecoder):
    """
    HiNeRV 解码器的改进版：每个 stage（除最后一个）尾接 AuxScaleHead；
    forward 可选返回 aux（多尺度预测列表 + 最后一级 head 前特征）。
    aux 头挂在 self.aux 下（不进码流、不剪枝、不量化）。
    """
    def __init__(self, logger, multiscale: bool = True, **kwargs):
        super().__init__(logger, **kwargs)
        self.multiscale = multiscale
        self.aux = nn.ModuleList()
        # 各 stage 输出通道数（与父类构建顺序一致地重算；flow head 也依赖它）
        C2 = max(self.channels, self.channels_min)
        self.aux_channels = []
        for i in range(len(self.scales)):
            if i > 0:
                C2 = max(divide_to_multiple(C2, self.channels_reduce, self.channels_reduce_base), self.channels_min)
            self.aux_channels.append(C2)
        if self.multiscale:
            # 最后一级由主输出头监督，不再加 aux 头
            for i in range(len(self.scales) - 1):
                self.aux.append(AuxScaleHead(self.aux_channels[i]))
            logger.info(f'     Aux scale heads: {len(self.aux)} (multiscale supervision)')
        else:
            logger.info(f'     Aux scale heads: disabled')

    def forward(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int], patch_mode: bool = True,
                return_aux: bool = False):
        """
        与 HiNeRVDecoder.forward 相同的数据流；return_aux=True 时额外返回
        aux = {'scales': [每级小视频 [N,3,t,h,w]], 'feat': 最后一级 head 前特征（有效区）}。
        """
        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        aux_scales = []
        aux_feat = None

        # initial configs
        padding = self.stem_paddings if patch_mode else (0, 0, 0)
        v_size_in = v_size_out = self.input_size
        p_size_in = p_size_out = tuple(v_size_out[d] // idx_max[d] for d in range(3))
        p_size_out_padded = tuple(p_size_out[d] + 2 * padding[d] for d in range(3))

        if patch_mode:
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
            x = x * px_mask_3d

        for i, block in enumerate(self.blocks):
            # Update configs
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
            if patch_mode:
                _, px_mask = compute_pixel_idx_3d(idx, idx_max, v_size_out, padding, clipped=False, return_mask=True)
                px_mask_3d = px_mask[0][:, :, None, None, None] \
                                * px_mask[1][:, None, :, None, None] \
                                * px_mask[2][:, None, None, :, None]
            else:
                px_mask_3d = None

            # Run block
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

            # 多尺度投影：裁掉 padding 只保留有效区（时间维也要裁）
            if return_aux and self.multiscale and i < len(self.aux):
                aux_scales.append(self.aux[i](crop_tensor_nthwc(x, p_size_out)))

        if return_aux:
            aux_feat = crop_tensor_nthwc(x, p_size_out)  # head 前最后一级特征（有效区）

        x = self.head(x)
        y = crop_tensor_nthwc(x, p_size_out)
        assert_shape(y, (idx.shape[0],) + p_size_out + (self.output_channels,))

        if return_aux:
            aux = {'scales': aux_scales, 'feat': aux_feat, 'scale_ratios': self.get_scale_ratios()}
            return y, aux
        return y

    def get_scale_ratios(self):
        """
        各 aux 级相对最终输出的空间下采样倍率：
        第 i 级小视频的分辦率 = 最终分辨率 / prod(scales[i+1:])。
        """
        ratios = []
        for i in range(len(self.scales) - 1):
            ratios.append(int(math.prod([s[1] for s in self.scales[i + 1:]])))
        return ratios


class MyHiNeRV(nn.Module):
    """
    MyHiNeRV 顶层模型 = HiNeRV(encoding + decoder) + CRSPE 每帧内容嵌入 + 辅助头。
    forward 返回 [N, C, T, H, W]；forward_with_aux 额外返回 aux dict（训练期新增损失用）。
    码流/剪枝/量化的前缀约定：
        - bitstream_prefix: 需要写入码流的部分（encoding、crspe、decoder 主体）
        - no_bitstream_prefix: decoder.aux / aux（训练期辅助头，不进码流）
        - no_prune_prefix / no_quant_prefix: 同 HiNeRV，另加辅助头
    """
    def __init__(self, encoding, decoder, crspe, eval_patch_size, flow_head=True, gop_mode='full'):
        super().__init__()

        # Model
        self.encoding = encoding
        self.decoder = decoder
        self.crspe = crspe
        self.eval_patch_size = eval_patch_size
        # GOP 残差模式（ResNet zero-init residual 思路）：输出头零初始化 → 初始修正量恒 0，
        # 重建 = 冻结 base 参考输出 + 修正量（恒等起步），训练只学对 base 的修正。
        self.gop_mode = gop_mode
        self.base_ref = None
        self.base_ref_offset = 0    # GOP 首帧在全局视频中的位置（局部→全局坐标平移）
        self.base_ref_full_size = None  # base 参考对应的完整视频尺寸 (T,H,W)

        # Train-only flow / confidence heads (Scene Matters SFCM; zero bitrate)
        self.flow_head_enabled = flow_head
        self.aux = None
        if flow_head:
            last_channels = self.decoder.aux_channels[-1]  # 最后一级 stage 的输出通道
            self.aux = nn.Module()
            self.aux.flow_head = Conv2d(last_channels, 4, kernel_size=1, padding='same', bias=True)
            self.aux.conf_head = Conv2d(last_channels, 2, kernel_size=1, padding='same', bias=True)
            nn.init.zeros_(self.aux.flow_head.bias)
            nn.init.zeros_(self.aux.conf_head.bias)

        # Prefixes for compression
        self.no_prune_prefix = ('encoding', 'crspe', 'decoder.aux', 'aux') + \
            tuple(k for k, v in self.named_modules() if isinstance(v, PositionalEncoder))
        self.no_quant_prefix = ('decoder.aux', 'aux')
        self.bitstream_prefix = ('encoding', 'crspe', 'decoder')
        self.no_bitstream_prefix = ('decoder.aux', 'aux')

    def attach_base_ref(self, base_ref, frame_offset, full_video_size):
        """
        挂载冻结的 base 参考模型（残差 GOP 模式）。
        base_ref 是按完整视频（full_video_size=(T,H,W)）构建/训练的模型；
        GOP 前向时把 clip 局部补丁坐标平移到全局坐标后求 base 输出。

        存储于普通 list 容器以绕过 nn.Module 的子模块注册——否则 base_ref 会进入
        state_dict/parameters：set_zero 解码校验时被清零（码流不含它，解码后不恢复，
        前向 base 分量归零 → 输出废掉，A1 10.8dB 的根因），且被优化器/检查点/
        参数化误包含。设备转移由调用方负责（attach 前 base_ref.to(device)）。
        """
        self._base_ref_box = [base_ref]
        self.base_ref_offset = int(frame_offset)
        self.base_ref_full_size = tuple(full_video_size)

    @property
    def base_ref(self):
        box = getattr(self, '_base_ref_box', None)
        return box[0] if box else None

    @base_ref.setter
    def base_ref(self, m):
        self._base_ref_box = [m] if m is not None else []

    def _base_ref_forward(self, input):
        """以全局坐标求 base 参考输出（GOP 局部 idx 的 t 维加上 clip 偏移）。

        关键：镜像调用方的 train/eval 模式——解码器的 patch_mode 由 self.training
        决定。base 若保持 eval 而调用方在训练（120×120 patch 输入），base 会按
        全帧语义解码 patch 坐标 → 输出错位（A1v2 训练 14.5dB 根因；旧实现靠
        子模块注册让 model.train() 连带切换 base 模式，注销注册后必须显式镜像）。
        """
        base_in = dict(input)
        idx = input['idx'].clone()
        idx[:, 0] = idx[:, 0] + self.base_ref_offset
        base_in['idx'] = idx
        base_in['video_size'] = self.base_ref_full_size
        base_in['idx_max'] = (self.base_ref_full_size[0] // input['patch_size'][0],
                              input['idx_max'][1], input['idx_max'][2])
        was_training = self.training
        self.base_ref.train(was_training)
        out = self.base_ref(base_in)
        if not was_training:
            self.base_ref.eval()
        return out

    def compute_frame_idx(self, idx, idx_max, video_size, padding_t, patch_mode):
        """
        计算补丁（含时间 padding）覆盖的帧坐标与有效掩码，供 CRSPE 取嵌入。
        返回 (t_idx [N, T_patch], t_mask [N, T_patch])。
        """
        t_idx, t_mask = compute_pixel_idx_1d(idx[:, 0:1], idx_max[0:1], (video_size[0],), (padding_t,), clipped=True, return_mask=True)
        return t_idx[0], t_mask[0]

    def forward(self, input):
        output, _ = self.forward_with_aux(input)
        return output

    def forward_with_aux(self, input):
        """
        模型前向（含辅助输出）。
        Input 字典与 HiNeRV 相同；输出 (output [N,C,T,H,W], aux)。
        aux = {
            'scales':  [多级小视频 [N,3,t,h,w], ...]（multiscale 关闭时为空表）
            'feat':    最后一级 head 前特征（flow head 关闭时为 None）
            'flow':    (fwd [N,2,T,H,W], bwd [N,2,T,H,W], conf [N,2,T,H,W]) 或 None
            'frame_idx': [N, T] 该补丁的（clamp 后）帧坐标（对比损失用）
            'frame_mask': [N, T] 有效帧掩码
            'scale_ratios': 各级小视频相对输出的下采样倍率
        }
        """
        assert all((input['patch_size'][d] % self.decoder.min_patch_size[d] == 0) for d in range(3))
        assert self.eval_patch_size is None or all(input['video_size'][d] % self.eval_patch_size[d] == 0 for d in range(3))

        need_aux = self.training

        # Config（评估时强制补丁模式，与 HiNeRV 一致）
        if not self.training and self.eval_patch_size is not None:
            idx_max = tuple(input['video_size'][d] // self.eval_patch_size[d] for d in range(3))
            idx = vidx_to_pidx(input['idx'], input['idx_max'], idx_max)
            patch_mode = True
        else:
            idx_max = input['idx_max']
            idx = input['idx']
            patch_mode = self.training

        # Compute output
        input_padding = self.decoder.get_input_padding(patch_mode=patch_mode)
        output = self.encoding(idx, idx_max, padding=input_padding)

        # CRSPE：每帧内容嵌入，广播到空间维并与基础编码拼接（Scene Matters 的拼接注入）
        if self.crspe is not None:
            t_idx, t_mask = self.compute_frame_idx(idx, idx_max, input['video_size'], input_padding[0], patch_mode)
            emb = self.crspe(t_idx, t_mask)  # [N, T, C_f]
            N, T = emb.shape[0], emb.shape[1]
            C_f = self.crspe.channels
            emb = emb[:, :, None, None, :].expand(N, T, output.shape[2], output.shape[3], C_f)
            output = torch.concat([output, emb], dim=-1)
        else:
            t_idx, t_mask = None, None

        if need_aux:
            output, aux = self.decoder(output, idx, idx_max, patch_mode=patch_mode, return_aux=True)
            aux['frame_idx'] = t_idx
            aux['frame_mask'] = t_mask
            aux['flow'] = None
            if self.flow_head_enabled and aux['feat'] is not None:
                feat = aux['feat']  # [N, T, H, W, C]
                flow = self.aux.flow_head(feat)  # [N, T, H, W, 4]
                conf = torch.softmax(self.aux.conf_head(feat), dim=-1)  # [N, T, H, W, 2]
                flow = flow.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)
                conf = conf.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)
                aux['flow'] = (flow[:, 0:2], flow[:, 2:4], conf)
            aux['feat'] = None  # 特征仅内部使用，不外传（省显存）
        else:
            output = self.decoder(output, idx, idx_max, patch_mode=patch_mode)
            aux = None

        # Reshape output
        if not self.training and self.eval_patch_size is not None:
            output = patch_to_video(output, input['patch_size'])
        output = output.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)

        # GOP 残差合成（在 [N,C,T,H,W] 上）：重建 = clamp(base 输出 + 零初始化修正量, 0, 1)
        if self.gop_mode == 'residual' and self.base_ref is not None:
            with torch.no_grad():
                base_out = self._base_ref_forward(input)
            output = (base_out.detach() + output).clamp(0., 1.)

        return output, aux

    def init_from_content(self, feats: torch.Tensor, seed: int = 0):
        """CRSPE 内容初始化：feats [T, D] 为 GT 下采样的每帧内容描述。"""
        if self.crspe is not None:
            self.crspe.init_from_content(feats, seed=seed)

    def get_num_parameters(self):
        """统计各部分参数量。"""
        base_encoding_param = sum([v.numel() for _, v in self.encoding.state_dict().items()])
        crspe_param = sum([v.numel() for _, v in self.crspe.state_dict().items()]) if self.crspe is not None else 0
        decoder_param = sum([v.numel() for k, v in self.decoder.state_dict().items() if not k.startswith('aux.')])
        aux_param = sum([v.numel() for k, v in self.named_parameters() if k.startswith('decoder.aux.') or k.startswith('aux.')])
        return {
            'All': decoder_param + base_encoding_param + crspe_param,
            'Decoder': decoder_param,
            'Model': decoder_param,
            'Encoding': base_encoding_param,
            'CRSPE': crspe_param,
            'Train-only aux': aux_param
        }


def build_decoder(args, logger, input_size, input_channels, output_size, output_channels):
    """根据命令行参数组装解码器配置并构建 MyHiNeRVDecoder（含多尺度辅助头）。"""
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
    for prefix in ['block', 'stem', 'head', 'enc', 'upsample']:
        cfg[f'{prefix}_cfg'] = {}
        for k, v in vars(args).items():
            if k.startswith(f'{prefix}_'):
                cfg[f'{prefix}_cfg'][k.replace(f'{prefix}_', f'')] = v

    cfg['multiscale'] = args.multiscale

    # Bulid decoder
    logger.info(f'Building MyHiNeRV Decoder with cfg: {cfg}')
    decoder = MyHiNeRVDecoder(logger, **cfg)

    return decoder


def build_model(args, logger, input):
    """
    模型总构建入口。在 HiNeRV 的基础上：
    - stem 输入通道 = base_channels + crspe_channels（CRSPE 拼接注入）；
    - 构建 CRSPE（帧数 = clip 帧数）与流/置信度辅助头。
    """
    # Set default configurations
    args.base_size = (input['video_size'][0] // int(math.prod(args.scales_t)) if args.base_size[0] == -1 else args.base_size[0],
                     input['video_size'][1] // int(math.prod(args.scales_hw)) if args.base_size[1] == -1 else args.base_size[1],
                     input['video_size'][2] // int(math.prod(args.scales_hw)) if args.base_size[2] == -1 else args.base_size[2])
    args.base_channels = sum([int(args.base_grid_size[3] // args.base_grid_level_scale[3] ** i) for i in range(args.base_grid_level)])

    args.base_grid_size[0] = args.base_grid_size[0] if args.base_grid_size[0] != -1 else args.base_size[0]
    args.base_grid_size[1] = args.base_grid_size[1] if args.base_grid_size[1] != -1 else args.base_size[1]
    args.base_grid_size[2] = args.base_grid_size[2] if args.base_grid_size[2] != -1 else args.base_size[2]
    args.base_grid_size[3] = args.base_grid_size[3] if args.base_grid_size[3] != -1 else 8

    assert args.base_size[0] == input['video_size'][0] // int(math.prod(args.scales_t))
    assert args.base_size[1] == input['video_size'][1] // int(math.prod(args.scales_hw))
    assert args.base_size[2] == input['video_size'][2] // int(math.prod(args.scales_hw))

    # Building encoding and decoder
    encoding = build_encoding(args, logger, args.base_size, args.base_channels)

    # CRSPE per-frame content embedding
    crspe_channels = args.crspe_channels
    if crspe_channels > 0:
        crspe = CRSPE(input['video_size'][0], crspe_channels, init_scale=args.crspe_init_scale)
        logger.info(f'Building CRSPE: {input["video_size"][0]} frames x {crspe_channels} channels')
    else:
        crspe = None
        logger.info(f'CRSPE disabled')

    # 残差修正头：无 sigmoid（需带符号修正量）；多尺度辅助头关闭（其监督对象是绝对图像而非修正量）
    # 必须在 build_decoder 之前覆盖，否则头仍用 sigmoid（sigmoid(0)=0.5 会破坏恒等起步）
    gop_mode = getattr(args, 'gop_mode', 'full')
    if gop_mode == 'residual':
        args.head_act = 'none'
        args.multiscale = False

    decoder = build_decoder(args, logger, args.base_size, args.base_channels + max(crspe_channels, 0), input['video_size'], 3)

    # Building model wrapper
    wrapper_cfg = {}
    wrapper_cfg['eval_patch_size'] = args.eval_patch_size
    wrapper_cfg['flow_head'] = args.flow_head
    wrapper_cfg['gop_mode'] = gop_mode

    logger.info(f'Building MyHiNeRV with cfg: {wrapper_cfg}')

    model = MyHiNeRV(encoding, decoder, crspe, **wrapper_cfg)

    # 零初始化输出头（ResNet zero-init residual）：初始修正量恒 0 → GOP 从恒等于 base 起步
    if gop_mode == 'residual':
        with torch.no_grad():
            for m in model.decoder.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    return model


def set_args(parser):
    """注册 MyHiNeRV 参数：先继承 HiNeRV 全部参数，再追加改进模块参数。"""
    from . import hinerv
    hinerv.set_args(parser)

    group = parser.add_argument_group('MyHiNeRV specific parameters')
    group.add_argument('--crspe-channels', type=int, default=32, help='channels of the per-frame content embedding (CRSPE). 0: disabled')
    group.add_argument('--crspe-init-scale', type=float, default=1e-2, help='initialization scale of CRSPE weights.')
    group.add_argument('--crspe-content-init', type=str_to_bool, default=True, help='initialize CRSPE from downsampled GT content.')
    group.add_argument('--crspe-content-size', type=int, default=32, help='resolution of the downsampled GT used for CRSPE content initialization.')
    group.add_argument('--multiscale', type=str_to_bool, default=True, help='per-stage aux scale heads for multi-scale (frequency) supervision.')
    group.add_argument('--flow-head', type=str_to_bool, default=True, help='train-only flow & confidence heads for RAFT pseudo-label supervision.')
    group.add_argument('--gop-mode', type=str, default='full', choices=['full', 'residual'],
                       help='GOP model mode: full (reconstruct segment itself) or residual (zero-init correction head added to the frozen base output).')
