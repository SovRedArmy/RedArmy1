# ==========================================================================================
# NeRV: Neural Representations for Videos (NeurIPS 2021) —— 模型与数据集定义
# 官方代码: https://github.com/haochen-rye/NeRV    论文: https://arxiv.org/abs/2110.13903
#
# 本文件包含:
#   1. CustomDataSet  : 视频帧数据集(把视频拆成逐帧 PNG, 每个样本返回 图像张量 + 归一化帧索引)
#   2. Sin / ActivationLayer / NormLayer : 正弦激活模块 / 激活函数与归一化层的"工厂函数"
#   3. CustomConv     : 可切换上采样方式的卷积块(PixelShuffle / 转置卷积 / 双线性插值)
#   4. MLP            : 全连接层堆叠成的 MLP(论文中的 stem / FC Layers)
#   5. NeRVBlock      : NeRV 基本块 = CustomConv 上采样 + Norm + Activation
#   6. Generator      : NeRV 主体网络(帧索引位置编码 → MLP → 多级上采样卷积 → RGB 图像)
# ==========================================================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    """视频帧数据集: 把一段视频的逐帧 PNG 图片当作训练样本.

    每个样本返回 (图像张量, 归一化帧索引), 其中帧索引被归一化到 [0, 1) 区间,
    作为 NeRV 网络的输入 —— 对应论文 Sec. 3.1 的核心设定: 网络输入是"帧索引"而非像素坐标.

    参数:
        main_dir   : 存放逐帧 PNG 的目录, 例如 ./data/bunny
        transform  : PIL 图像 → 张量的变换(训练脚本传入 transforms.ToTensor())
        vid_list   : 指定使用的帧序号列表; 默认 [None] 表示使用全部帧
        frame_gap  : 帧采样间隔(每隔 frame_gap 帧取一张), 用于控制参与训练的帧数
        visualize  : 调试用开关(当前实现中未使用)
    """
    def __init__(self, main_dir, transform, vid_list=[None], frame_gap=1,  visualize=False):
        self.main_dir = main_dir                 # 帧图片所在目录
        self.transform = transform               # PIL 图像 → Tensor 的变换
        frame_idx, self.frame_path = [], []      # 帧编号列表 / 帧文件名列表
        accum_img_num = []                       # 累计帧数(多视频拼接场景使用)
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()                          # 按文件名排序, 保证帧的时间顺序正确

        num_frame = 0 
        for img_id in all_imgs:                  # 遍历所有帧, 依次记录编号 0, 1, 2, ...
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)
            num_frame += 1          

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        # 帧索引归一化到 [0, 1) 区间(例如共 100 帧时: 第 0 帧 → 0.0, 第 1 帧 → 0.01, ...)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        self.accum_img_num = np.asfarray(accum_img_num)
        # 若 vid_list 指定了帧序号, 则只保留这些帧(用于挑选子集训练)
        if None not in vid_list:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap

    def __len__(self):
        # 数据集长度 = 可用帧数 // 采样间隔
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        # 先把逻辑下标映射为实际帧号(乘上采样间隔 frame_gap)
        valid_idx = idx * self.frame_gap
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")   # 读取 PNG 帧并转为 RGB 三通道
        tensor_image = self.transform(image)          # PIL → Tensor, 值域归一化到 [0,1]
        # 若 H > W(竖屏视频)则转置为横屏; bunny/UVG 等横屏数据不会触发
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        # 返回该帧对应的归一化帧索引, 即 NeRV 网络的输入
        frame_idx = torch.tensor(self.frame_idx[valid_idx])

        return tensor_image, frame_idx

class Sin(nn.Module):
    """正弦激活函数模块(对输入直接取 sin), 供 ActivationLayer('sin') 选用.

    NeRV 是"图像级"的神经表示, 论文默认使用 GELU/Swish 等激活;
    Sin 在此作为可选项保留(SIREN 一类网络使用正弦激活).
    """
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    """激活函数工厂: 按字符串名字返回对应的激活层实例.

    论文 Sec. 4.3 消融实验对比了 relu / leaky / gelu / swish 等,
    结论是 GELU 与 Swish 优于 ReLU(代码默认 act='gelu').
    """
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
    """归一化层工厂: 按名字返回 Identity(不加归一化)/BatchNorm2d/InstanceNorm2d.

    论文 Sec. 4.3 消融: 不使用归一化(norm='none')效果最好, BN 次之, IN 最差
    (在"单个视频过拟合"的任务里, 归一化反而会损害表达能力).
    参数 ch_width 为该层对应的通道数.
    """
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class CustomConv(nn.Module):
    """带可切换上采样方式的卷积块 —— NeRV 上采样层的三种实现(论文 Sec. 3.2, 消融见 Sec. 4.3):

    conv     : 3x3 卷积(输出通道先放大 stride^2 倍) + PixelShuffle 上采样
               —— 论文主推方案, 消融中 PSNR 最高(PixelShuffle 37.26 vs 转置卷积 36.63 vs 双线性 29.56)
    deconv   : 转置卷积(kernel/stride 相同)直接上采样
    bilinear : 双线性插值先放大分辨率, 再用卷积调整通道数

    参数(通过 kargs 传入):
        ngf / new_ngf : 输入 / 输出通道数
        stride        : 空间上采样倍率
        conv_type     : 'conv' | 'deconv' | 'bilinear' 三选一
        bias          : 卷积是否带偏置
    """
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            # 卷积把通道数扩到 new_ngf*stride*stride, 再由 PixelShuffle "用通道换分辨率"
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            # 转置卷积: 以 stride 为步长做反卷积, 一步完成上采样
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            # 双线性插值把空间分辨率放大 stride 倍, 再用卷积调整通道数
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)          # 先做卷积/插值
        return self.up_scale(out)   # 再上采样(PixelShuffle 等)


def MLP(dim_list, act='relu', bias=True):
    """按维度列表 dim_list 构造 MLP(全连接层 + 激活函数交替堆叠).

    例如 dim_list=[160, 1024, 18432] 会生成:
        Linear(160,1024) → Act → Linear(1024,18432) → Act
    在 NeRV 中作为 stem/FC 层: 把帧索引的位置编码向量映射为展平的低分辨率特征图
    (对应论文 Sec. 3.2 中的 FC Layers). 注意每个 Linear 后共用同一个激活层实例 act_fn.
    """
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class NeRVBlock(nn.Module):
    """NeRV 基本块(论文 Fig. 2 中的 NeRV Layers): 上采样卷积 + 归一化 + 激活.

    数据流: x → CustomConv(空间分辨率放大 stride 倍) → NormLayer → Activation
    """
    def __init__(self, **kargs):
        super().__init__()

        self.conv = CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'], 
            conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        # conv(上采样) → norm → act 的顺序结构
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    """NeRV 生成网络(它本身即"视频的表示"): 输入帧索引的位置编码, 输出一整帧 RGB 图像.

    整体结构(论文 Fig. 2):
        [位置编码向量, embed_length 维]
            → stem (MLP, 即论文中的 FC Layers)
            → reshape 成 (fc_dim, fc_h, fc_w) 的低分辨率特征图
            → 若干 NeRVBlock 逐级上采样(如 strides=5,3,2,2,2 时: 9x16 → 45x80 → ... → 1440x2560)
            → 每个 stage 的 head(1x1 卷积) 把特征图映射为 3 通道 RGB 图像

    关键 kargs(由 train_nerv.py 的命令行参数传入):
        embed_length : 位置编码维度(PositionalEncoding 的输出维度, 如 '1.25_80' → 160)
        stem_dim_num : MLP 隐层'宽度_层数', 如 '1024_1' 表示 1 个宽 1024 的隐层
        fc_hw_dim    : '高_宽_通道', MLP 输出被 reshape 成该尺寸的特征图, 如 '9_16_128'
        expansion    : 第一个卷积 stage 相对 fc_dim 的通道扩张倍数
        reduction    : 之后每个 stage 的通道数缩减倍数(分辨率越高、通道越少)
        lower_width  : 通道数下限(对应最小模型宽度, NeRV-S/M/L 靠这些宽度参数调节大小)
        stride_list  : 每个 stage 的上采样倍率列表(全部乘积 × fc_hw_dim = 目标分辨率)
        num_blocks   : 每个 stage 内 NeRVBlock 的个数
        sin_res      : True=单分辨率输出(仅最后一个 stage 接 head)/False=多分辨率监督(每个 stage 都接 head)
        sigmoid      : 输出归一化用 sigmoid 还是 (tanh+1)/2
    """
    def __init__(self, **kargs):
        super().__init__()

        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]          # '1024_1' → (1024, 1)
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]  # '9_16_128' → (9, 16, 128)
        # MLP 的维度序列: [位置编码维度] + stem_num 个宽 stem_dim 的隐层 + [fc_h*fc_w*fc_dim]
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act'])   # 论文中的 FC Layers
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]  # 卷积主体 / 各级输出头
        ngf = self.fc_dim                                            # 当前通道数, 从 fc_dim 开始
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                # 第一个 stage: 用 expansion 倍扩张通道(分辨率最低时通道最多)
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                # 后续 stage: 通道按 reduction 缩减(stride=1 时不变), 但不低于 lower_width
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                # 每个 stage 内第一个 block 用 stride 上采样, 其余 block stride=1 只变换通道
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if kargs['sin_res']:
                # 单分辨率模式(论文主实验设定): 只有最后一个 stage 接 head, 仅输出最终分辨率图像
                if i == len(kargs['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias']) 
                    # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias']) 
                else:
                    head_layer = None
            else:
                # 多分辨率模式: 每个 stage 都接 head, 输出多个分辨率的图像(多尺度监督)
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']   # forward 时决定输出用 sigmoid 还是 tanh 归一化

    def forward(self, input):
        """前向传播: 位置编码后的帧索引向量 input (B, embed_length) → 各尺度 RGB 图像列表.

        返回 out_list: 每个 head 输出的 (B,3,H,W) 图像张量;
        多分辨率模式下包含每个 stage 的输出, 最终帧取 out_list[-1](分辨率最高).
        """
        output = self.stem(input)   # MLP: (B, embed_length) → (B, fc_h*fc_w*fc_dim)
        # 把一维特征向量 reshape 成低分辨率特征图 (B, fc_dim, fc_h, fc_w)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output)            # NeRVBlock: 逐级上采样(分辨率↑, 通道↓)
            if head_layer is not None:
                img_out = head_layer(output)  # 1x1 卷积头: 特征图 → 3 通道 RGB
                # normalize the final output iwth sigmoid or tanh function
                # 输出归一化到 [0,1]: sigmoid 或 (tanh+1)/2
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list

