"""
FFNeRV 模型定义文件。

论文: FFNeRV: Flow-Guided Frame-Wise Neural Representations for Videos
      (Joo Chan Lee, Daniel Rho, Jong Hwan Ko, Eunbyung Park; ICML 2023; arXiv:2212.12294)

本文件包含 FFNeRV 的核心网络组件：
- CustomDataSet:       视频帧数据集，把目录里的帧图片读成 (图像张量, 归一化时间坐标) 样本
- Sin:                 sin 激活函数模块
- ActivationLayer:     激活函数工厂（按字符串名返回对应激活层）
- NormLayer:           归一化层工厂（none / bn / in）
- qfn:                 伪量化函数（量化感知训练 QAT 的核心：前向量化、反向直通 STE）
- weight_quantize_fn:  权重量化模块（tanh 压缩到 (-1,1) 后做 k-bit 伪量化）
- Conv2d_Q:            量化卷积基类（仅继承 nn.Conv2d，本身无改动）
- conv2d_quantize_fn:  卷积层工厂，返回"前向时对权重/偏置做伪量化"的 Conv2d 变体
- CustomConv:          上采样卷积块（conv / compact / deconv / bilinear 四种上采样方式）
- ConvBlock:           卷积 + 归一化 + 激活的组合块（生成器的基本单元）
- Generator:           FFNeRV 生成器：多分辨率时间网格 + 全卷积上采样 + 双输出头
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    """视频帧数据集。

    把存放视频帧图片的目录包装成 PyTorch Dataset。
    每个样本返回 (tensor_image, frame_idx)：
      - tensor_image: 该帧图像张量 (3, H, W)，取值范围 [0,1]
      - frame_idx:    归一化到 [0,1] 的帧索引，作为 Generator 的连续时间坐标输入
    """
    def __init__(self, main_dir, transform, vid_list=[None], resol=[1280, 720], frame_gap=1,  visualize=False, offset=0):
        """初始化数据集。

        参数:
            main_dir:   视频帧图片所在目录
            transform:  图像变换（main.py 中传入 transforms.ToTensor()）
            vid_list:   帧索引筛选列表；默认 [None] 表示不筛选、使用全部帧
            resol:      目标分辨率 [宽, 高]（注意顺序：__getitem__ 中 resize((w, h))，
                        与命令行 --resol 参数一致，如 1920 1080 表示宽 1920 高 1080）
            frame_gap:  取帧间隔（每隔 frame_gap 帧取一帧，用于帧插值等实验）
            offset:     取帧起始偏移
        """
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()

        # 逐个登记帧文件名，并记录从 0 开始的帧序号
        num_frame = 0
        for img_id in all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)
            num_frame += 1

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        #accum_img_num.append(num_frame)
        # 帧序号归一化到 [0,1]，作为喂给网络的连续时间坐标
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        if None not in vid_list:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap
        self.offset = offset
        self.h, self.w = resol

    def __len__(self):
        """数据集长度：按 frame_gap 抽取后的帧数。"""
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        """按索引取一帧：读图片 -> resize 到目标分辨率 -> 转成 (3,H,W) 张量。

        返回 (tensor_image, frame_idx)：图像张量与该帧的归一化时间坐标。
        """
        valid_idx = idx * self.frame_gap + self.offset  # 实际帧序号 = 采样序号 * 间隔 + 偏移
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        image = image.resize((self.h, self.w))          # PIL resize 参数为 (宽, 高)
        tensor_image = self.transform(image)
        # 统一为 (C, 高, 宽) 布局：若第 1 维 > 第 2 维（按宽 x 高存储的转置格式），交换后两维
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])

        return tensor_image, frame_idx

class Sin(nn.Module):
    """sin 激活函数模块（SIREN 风格，可选激活之一）。"""
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    """激活函数工厂：按名称字符串返回对应的激活层模块。"""
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
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):
    """归一化层工厂：按类型返回 none(恒等) / bn(批归一化) / in(实例归一化)。"""
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class qfn(torch.autograd.Function):
    """k-bit 伪量化函数（量化感知训练 QAT 的核心）。

    前向: 对称均匀量化。量化级数 n = 2^(k-1) - 1（如 8bit 时 n=127），
          out = floor(|x| * n) / n * sign(x)，把数值吸附到最近的量化电平上。
    反向: 直通估计器（Straight-Through Estimator, STE），梯度原样传回，
          从而让"不可导的量化"可以放进网络里端到端训练。
    """
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)          # 量化级数（正半轴电平数）
        out = torch.floor(torch.abs(input) * n) / n  # 绝对值量化
        out = out*torch.sign(input)      # 恢复符号
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()  # STE：梯度直通
        return grad_input, None


class weight_quantize_fn(nn.Module):
    """权重量化模块：前向时对输入张量做 k-bit 伪量化。

    先用 tanh 把权重压缩到 (-1,1)，再经过 qfn 量化到 k-bit 电平；
    wbit=32 视为全精度、直接返回原值。
    """
    def __init__(self, bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            weight_q = x                  # 全精度，不量化
        else:
            weight = torch.tanh(x)        # 压缩到 (-1,1)
            weight_q = qfn.apply(weight, self.wbit)  # 伪量化到 wbit 位
        return weight_q

class Conv2d_Q(nn.Conv2d):
    """量化卷积的基类（直接继承 nn.Conv2d，未做任何修改）。"""
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)


def conv2d_quantize_fn(bit):
    """卷积层工厂：按指定位宽 bit 返回一个"带 QAT 的 Conv2d"类。

    返回的 Conv2d_Q_ 类在 forward 中先用 weight_quantize_fn 对自身的
    weight 和 bias 做伪量化，再调用 F.conv2d 完成卷积——
    训练时即模拟量化误差，使训练出的权重对量化更鲁棒（QAT）。
    """
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.w_bit = bit
            self.quantize_fn = weight_quantize_fn(self.w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)  # 权重伪量化
            bias_q = self.quantize_fn(self.bias)      # 偏置伪量化
            return F.conv2d(input, weight_q, bias_q, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_

class CustomConv(nn.Module):
    """上采样卷积块：一次完成"通道变换 + 空间上采样 stride 倍"。

    conv_type 四种方式:
      - 'conv':     3x3 卷积输出 new_ngf*stride^2 通道，再用 PixelShuffle(stride)
                    将通道折叠成空间分辨率（NeRV/HNeRV 的标准上采样做法）
      - 'compact':  论文的紧凑设计（用于压缩版模型）：
                    3x3 分组卷积（groups=8）+ 1x1 逐点卷积，再 PixelShuffle，
                    大幅减少参数量（分组卷积降参 + 逐点卷积混合通道信息）。
                    特例: stride==5 时与 'conv' 相同（单卷积, groups=1）
      - 'deconv':   转置卷积上采样（注意: forward 只调用 conv1/conv2，而该分支
                    未定义 conv1，因此实际不可用——历史遗留代码）
      - 'bilinear': 双线性上采样 + 带步幅卷积（同样存在上述遗留问题，实际不可用）
    实际可用的是 'conv' 与 'compact'（README 中的两条命令分别使用这两种）。
    """
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()
        Conv2d = conv2d_quantize_fn(kargs['wbit'])  # 根据 wbit 得到带 QAT 的卷积类
        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            # 3x3 卷积 -> PixelShuffle 上采样
            self.conv1 = Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.conv2 = nn.Identity()
            self.up_scale = nn.PixelShuffle(stride)
        # compact convolution
        elif self.conv_type == 'compact':
            if stride == 5:
                # stride=5 的第一级：直接单卷积（groups=1）
                self.conv1 = Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'], groups=1)
                self.conv2 = nn.Identity()
                self.up_scale = nn.PixelShuffle(stride)
            else:
                # 紧凑版：3x3 分组卷积(groups=8) + 1x1 逐点卷积，再 PixelShuffle
                self.conv1 = Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'], groups=8)
                self.conv2 = Conv2d(new_ngf * stride * stride, new_ngf * stride * stride, 1, bias=kargs['bias'])
                self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        """前向：conv1 -> conv2 -> up_scale(PixelShuffle 或恒等)。"""
        out = self.conv1(x)
        out = self.conv2(out)
        return self.up_scale(out)


class ConvBlock(nn.Module):
    """生成器的基本卷积单元：CustomConv(上采样+通道变换) -> Norm -> Activation。"""
    def __init__(self, **kargs):
        super().__init__()

        self.conv = CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'],
            conv_type=kargs['conv_type'], wbit=kargs['wbit'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        """前向：卷积 -> 归一化 -> 激活。"""
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    """FFNeRV 生成器（即视频的隐式神经表示本身；"编码视频" = 训练这个网络）。

    结构（论文图 2）：
      1. 多分辨率时间网格 self.video_grid:
         K = len(t_dim) 个形状 (t_i, fc_dim/K, fc_h, fc_w) 的可学习参数（t_i 各不相同）。
         给定归一化时间坐标 t，在每个网格内沿时间轴做线性插值取出隐特征，
         再在通道维拼接成 (B, fc_dim, fc_h, fc_w)。
         这种一维时间网格取代了 NeRV 的 MLP 头，使网络可以做成全卷积，
         空间特征连续性更好（论文 Sec. 4.1）。
      2. 全卷积上采样 self.layers:
         按 stride_list 逐级上采样（默认 5,3,2,2,2 -> 共 120 倍），
         每级含 num_blocks 个 ConvBlock，其中只有第一个块做上采样。
      3. 双输出头 self.head_layers（sin_res=False，论文默认）:
         - 倒数第 3 级（默认配置下为 1/4 分辨率）: 1x1 卷积输出 aggs*3 通道
           = 每个参考帧 2 通道光流 + 1 通道第一级聚合权重（论文的流预测分支）
         - 最后一级（全分辨率）: 1x1 卷积输出 5 通道
           = 3 通道"独立帧"RGB + 2 通道第二级聚合权重（独立帧 vs 聚合帧的融合权重）
      两个头的输出由 main.py 的训练/评估循环完成流引导的两级帧聚合。
    """
    def __init__(self, **kargs):
        """按关键字参数构建网络。主要参数：
            fc_hw_dim:   '高_宽_通道' 字符串，初始特征图的尺寸与通道数（如 '9_16_48'）
            expansion:   第一级通道扩张倍数（紧凑压缩版常用 8）
            reduction:   各级通道缩减倍数
            stride_list: 各级上采样倍率（乘积 = 初始特征图到整帧的总放大倍数）
            lower_width: 通道数下限
            t_dim:       各时间网格的时间分辨率列表（多分辨率）
            aggs:        参考帧个数 = len(agg_ind)，决定第一级头输出通道数 aggs*3
            wbit:        QAT 权重位宽（32 表示全精度不量化）
            sin_res:     单头消融模式（NeRV 式直接输出 RGB，不走流引导聚合）
            sigmoid:     独立帧 RGB 用 sigmoid 还是 (tanh+1)/2 归一化到 [0,1]
        """
        super().__init__()
        Conv2d = conv2d_quantize_fn(kargs['wbit'])            # 带 QAT 的卷积类
        self.quantize_fn = weight_quantize_fn(kargs['wbit'])  # 用于量化时间网格参数
        self.strides = kargs['stride_list']
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        # multi-resolution temporal grids —— 多分辨率时间网格（论文 Sec. 4.1）
        # 第 i 个网格形状 (t_dim[i], fc_dim/K, fc_h, fc_w)，通道数在 K 个网格间均分
        self.video_grid = nn.ParameterList()
        for t in kargs['t_dim']:
            self.video_grid.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(t,self.fc_dim//len(kargs['t_dim']),self.fc_h,self.fc_w))))

        # BUILD CONV LAYERS —— 构建逐级上采样卷积层与各级输出头
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                # —— 第一级：通道从 fc_dim 扩张 expansion 倍
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                # —— 后续各级：通道按 reduction 缩减（stride=1 的块不缩减），但不低于 lower_width
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                # 每级 num_blocks 个卷积块，只有第一个块（j=0）做上采样，其余 stride=1
                self.layers.append(ConvBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type'], wbit=kargs['wbit']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer
            # 构建该级输出头（双头含义见类 docstring；head_layer=None 表示该级无头）
            head_layer = [None]
            if kargs['sin_res']:
                # 单头消融：只有最后一级输出 3 通道 RGB
                if i == len(kargs['stride_list']) - 1:
                    head_layer = Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                else:
                    head_layer = None
            else:
                if i == len(kargs['stride_list']) - 3:
                    # 第一级头（默认配置下 1/4 分辨率）：每个参考帧 2 通道光流 + 1 通道聚合权重
                    head_layer = Conv2d(ngf, kargs['aggs']*3, 1, 1, 0, bias=kargs['bias'])
                elif i == len(kargs['stride_list']) - 1:
                    # 第二级头（全分辨率）：3 通道独立帧 RGB + 2 通道二级聚合权重
                    head_layer = Conv2d(ngf, 5, 1, 1, 0, bias=kargs['bias'])
                else:
                    head_layer = None
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']  # 独立帧 RGB 的归一化方式开关


    def forward(self, input):
        """前向传播：时间坐标 -> 插值网格特征 -> 逐级卷积上采样 -> 双头输出。

        输入 input: (B,) 归一化时间坐标 t ∈ [0,1]。
        返回 out_list（sin_res=False 时两个元素）:
          [0] 第一级头 (B, 3K, H/4, W/4)：前 2K 通道为 K 个参考帧的光流（每帧 2 通道），
              后 K 通道为第一级聚合权重（每帧 1 通道）
          [1] 第二级头 (B, 5, H, W)：前 3 通道为"独立帧"RGB（已归一化到 [0,1]），
              后 2 通道为第二级聚合权重
        """
        out_list = []
        for param in self.video_grid: # multi-resolution grids
            # 对网格参数做伪量化（QAT：训练时即模拟 wbit 位宽的量化误差）
            vg = self.quantize_fn(param)
            # interpolate grid features
            # 用时间坐标在网格时间轴上做线性插值，取出该时刻的隐特征：
            inp = input*(param.size(0))             # [0,1] -> [0,t) 的浮点时间坐标
            left = torch.floor(inp+1e-6).long()     # 左侧网格切片索引
            right = torch.min(left+1, torch.tensor(param.size(0)-1))  # 右侧切片索引（截断在网格末端）
            d_left = (inp - left).view(-1, 1, 1, 1) # 距左侧切片的归一化距离
            d_right = (right - inp).view(-1, 1, 1, 1)
            # 线性插值: d_right*vg[left] + d_left*vg[right]；
            # 第三项是边界修正：当时间坐标落在最后一个切片（right==left）时 right-left-1 = -1，
            # 恰好抵消前两项并补回 vg[left]，使输出退化为边界切片本身
            out_list.append(d_right*vg[left] + d_left*vg[right] - ((right-left-1).view(-1,1,1,1))*vg[left])
        output = out_list[0]
        # concat latent features from multi-resolution grids
        # 在通道维拼接各分辨率网格插值出的特征 -> (B, fc_dim, fc_h, fc_w)
        for i in range(len(out_list)-1):
            output = torch.cat([output, out_list[i+1]],dim=1)

        out_list = []
        # 逐级通过 ConvBlock 上采样；遇到有输出头的级，额外产生一个头输出
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output)
            if head_layer is not None:
                img_out = head_layer(output)
                out_list.append(img_out)
        # normalize the independent frame to [0,1]
        # 把第二级头输出的"独立帧"3 个 RGB 通道归一化到 [0,1]
        # （sigmoid 或 (tanh+1)*0.5，由 --sigmoid 开关选择）
        out_list[1][:,0:3,:,:] = torch.sigmoid(out_list[1][:,0:3,:,:]) if self.sigmoid else (torch.tanh(out_list[1][:,0:3,:,:]) + 1) * 0.5
        return  out_list
