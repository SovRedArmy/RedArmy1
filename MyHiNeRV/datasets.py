"""
数据集与加载器（MyHiNeRV 版）：
- VideoDataset 支持帧区间（frame_range）——每个 clip 一个数据集实例；
- 可选返回 RAFT 光流伪标签 patch（供 SFCM 约束，见 flow_utils）；
- create_loader 支持加权采样（dynamicAL 帧权重 → 补丁级 WeightedRandomSampler）；
- compute_content_features：GT 下采样内容特征（CRSPE 内容初始化与
  场景切分的余弦散度都用它）。
"""
from utils import *

import numpy as np
from flow_utils import FlowProvider


class VideoDataset(torch.utils.data.Dataset):
    """
    视频数据集：一个实例对应一个视频（或其中一段 clip）的 PNG 帧文件夹。
    支持 center-crop → resize 预处理、3D 补丁加载、image/patch 两级缓存、
    clip 帧区间（frame_range）与光流伪标签（flow_provider）。
    """
    def __init__(self, logger, root, name, crop=[-1, -1], resize=[-1, -1], patch_size=[1, -1, -1],
                 cached='none', frame_range=None, flow_provider: FlowProvider = None):
        super().__init__()
        self.logger = logger
        self.root = os.path.expanduser(root)
        self.name = name
        self.img_paths = sorted([f for f in os.listdir(os.path.join(self.root, self.name)) if not f.startswith(".")])

        # Clip 帧区间：数据集只暴露 [t0, t1) 的帧，所有索引/坐标都是 clip 内局部坐标
        self.frame_range = (0, len(self.img_paths)) if frame_range is None else tuple(frame_range)
        assert 0 <= self.frame_range[0] < self.frame_range[1] <= len(self.img_paths)
        self.img_paths = self.img_paths[self.frame_range[0]:self.frame_range[1]]

        self.raw_size = self.get_raw_size()
        self.crop = tuple(crop[d] if crop[d] != -1 else self.raw_size[d] for d in range(2))
        self.resize = tuple(resize[d] if resize[d] != -1 else self.crop[d] for d in range(2))

        self.video_size = (len(self.img_paths), self.resize[0], self.resize[1])
        self.patch_size = tuple(patch_size[d] if patch_size[d] != -1 else self.video_size[d] for d in range(3))

        assert all(self.video_size[d] % self.patch_size[d] == 0 for d in range(3))
        self.num_patches = tuple(self.video_size[d] // self.patch_size[d] for d in range(3))

        # RAFT 伪标签提供器（仅训练集使用；可为 None）
        self.flow_provider = flow_provider

        assert cached in ['none', 'image', 'patch']
        self.cached = cached
        self.load_cache()

        self.logger.info(f'VideoDataset:')
        self.logger.info(f'     root: {self.root}    name: {self.name}    number of images: {len(self.img_paths)}'
                         + (f'    frame range: {self.frame_range}' if self.frame_range != (0, len(self.img_paths)) else ''))
        self.logger.info(f'     video_size: {self.video_size}    patch_size: {self.patch_size}    num_patches: {self.num_patches}')
        self.logger.info(f'     cached: {self.cached}    flow: {self.flow_provider is not None}')

    def load_cache(self):
        """缓存全部帧 / 全部补丁（训练加速）。"""
        if self.cached == 'image' or self.cached == 'patch':
            self.image_cached = self.load_all_images()
        else:
            self.image_cached = None

        if self.cached == 'patch':
            self.patch_cached = self.load_all_patches()
        else:
            self.patch_cached = None

        if self.cached == 'patch':
            self.image_cached = None

    def get_raw_size(self):
        """读取第一帧得到原始分辨率（假定所有帧同尺寸）。"""
        img = torchvision.io.read_image(os.path.join(self.root, self.name, self.img_paths[0]))
        return img.shape[1:3]

    def load_image(self, idx):
        """加载单帧（无缓存路径）：读 PNG → center crop → bicubic resize。"""
        assert isinstance(idx, int)
        img = torchvision.io.read_image(os.path.join(self.root, self.name, self.img_paths[idx]))
        img = torchvision.transforms.functional.center_crop(img, self.crop)
        img = torchvision.transforms.functional.resize(img, self.resize, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
        return img

    def load_patch(self, idx):
        """加载单个 3D 补丁 (t,h,w)：逐帧取对应空间区域后沿时间维拼接 → [C, Pt, Ph, Pw]。"""
        assert isinstance(idx, tuple) and len(idx) == 3
        patches = []
        h = idx[1] * self.patch_size[1]
        w = idx[2] * self.patch_size[2]
        for dt in range(self.patch_size[0]):
            t = idx[0] * self.patch_size[0] + dt
            image = self.image_cached[t] if self.image_cached is not None else self.load_image(t)
            patch = image[:, None, h: h + self.patch_size[1], w: w + self.patch_size[2]]
            patches.append(patch)
        return torch.concatenate(patches, dim=1)

    def load_all_images(self):
        """加载并缓存全部帧。"""
        images = {}
        for t in range(self.video_size[0]):
            images[t] = self.load_image(t)
        return images

    def load_all_patches(self):
        """加载并缓存全部 3D 补丁。"""
        patches = {}
        for t in range(self.num_patches[0]):
            for h in range(self.num_patches[1]):
                for w in range(self.num_patches[2]):
                    patches[(t, h, w)] = self.load_patch((t, h, w))
        return patches

    def get_image(self, idx):
        """取单帧（优先缓存）。"""
        assert isinstance(idx, int)
        return self.image_cached[idx] if self.image_cached is not None else self.load_image(idx)

    def get_patch(self, idx):
        """取单个 3D 补丁（优先缓存）。"""
        assert isinstance(idx, tuple) and len(idx) == 3
        return self.patch_cached[idx] if self.patch_cached is not None else self.load_patch(idx)

    def __len__(self):
        """数据集大小 = 补丁总数。"""
        return math.prod(self.num_patches)

    def __getitem__(self, idx):
        """
        线性索引 → 3D 补丁坐标 (t,h,w)。
        无 flow_provider：返回 (坐标张量, [0,1] 补丁)；
        有 flow_provider：额外返回 (flow_fwd, flow_bwd, valid) 伪标签 patch。
        """
        assert isinstance(idx, int)
        idx_thw = (idx // (self.num_patches[1] * self.num_patches[2]),
                    (idx % (self.num_patches[1] * self.num_patches[2])) // self.num_patches[2],
                    (idx % (self.num_patches[1] * self.num_patches[2])) % self.num_patches[2])
        patch = self.get_patch(idx_thw)
        x = torch.clone(patch).float() / 255.
        idx_t = torch.tensor(idx_thw, dtype=int)
        if self.flow_provider is not None:
            f, b, valid = self.flow_provider.get_patch_flow(idx_thw, self.patch_size)
            return idx_t, x, f, b, torch.tensor(valid, dtype=torch.float32)
        return idx_t, x


@torch.no_grad()
def compute_content_features(dataset, size=32, batch_frames=32):
    """
    每帧内容特征：GT resize 到 size×size 后展平 [T, 3*size*size]（float32）。
    用于 CRSPE 内容初始化与场景切分的余弦散度。
    """
    T = dataset.video_size[0]
    feats = []
    for s in range(0, T, batch_frames):
        imgs = torch.stack([dataset.get_image(t).float() / 255. for t in range(s, min(s + batch_frames, T))])
        f = torch.nn.functional.interpolate(imgs, size=(size, size), mode='bilinear', align_corners=False)
        feats.append(f.flatten(1))
    return torch.cat(feats, dim=0)


def create_dataset(args, logger, training, frame_range=None, with_flow=False):
    """
    创建数据集：训练集按 patch_size 切补丁；评估集只按帧切。
    frame_range: clip 帧区间（None = 全视频）。
    with_flow: 训练集是否附带 RAFT 伪标签（需预先运行 scripts/precompute_flow.py）。
    """
    from flow_utils import flow_file_path
    flow_provider = None
    if training and with_flow:
        fp = flow_file_path(args.dataset, args.dataset_name)
        if os.path.isfile(fp):
            flow_provider = FlowProvider(fp, video_size=None, frame_offset=frame_range[0] if frame_range else 0)
        else:
            logger.info(f'Flow pseudo-label not found at {fp}; SFCM constraint disabled. Run scripts/precompute_flow.py first.')

    # 评估集不做缓存（省内存；eval 频率低，逐帧读 PNG 的开销可接受）
    eval_cached = 'none' if not training else args.cached
    dataset = VideoDataset(logger, root=args.dataset, name=args.dataset_name,
                           crop=args.crop_size, resize=args.input_size,
                           patch_size=args.patch_size if training else [args.patch_size[0], -1, -1],
                           cached=eval_cached,
                           frame_range=frame_range,
                           flow_provider=flow_provider)

    # FlowProvider 需要数据集推导出的真实 video_size（处理 -1 默认值）
    if flow_provider is not None:
        flow_provider.video_size = dataset.video_size
        flow_provider.frame_offset = dataset.frame_range[0]

    return dataset


def create_loader(args, logger, training, dataset, sample_weights=None, num_samples=None):
    """
    创建 DataLoader：
    - sample_weights 为 None：训练 shuffle / 评估顺序（与 HiNeRV 一致）；
    - sample_weights 为 [len(dataset)] 张量：训练用 WeightedRandomSampler
      （dynamicAL 帧权重 + patch 丢弃的联合采样权重，损失仍逐帧等权）；
      num_samples 指定每 epoch 采样数（patch 丢弃时 < len(dataset)）。
    """
    if training and sample_weights is not None:
        n = len(dataset) if num_samples is None else num_samples
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=n,
                                                         replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = training

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.eval_batch_size if not training else args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        persistent_workers=args.workers > 0,
        drop_last=True,
        prefetch_factor=2 if args.workers > 0 else None
    )
    return loader


def set_dataset_args(parser):
    """注册数据集相关命令行参数。"""
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--dataset', default='~/Data', type=str, help='root path of datasets')
    group.add_argument('--dataset-name', default='Bunny', type=str, help='dataset name. dataset/dataset_name should be the path for image folder.')
    group.add_argument('--crop-size', default=[-1, -1], type=int, nargs='+', help='crop size (before resizing to the input)')
    group.add_argument('--input-size', default=[-1, -1], type=int, nargs='+', help='input size (scaling that apply after cropping)')
    group.add_argument('--patch-size', default=[1, -1, -1], type=int, nargs='+', help='patch size (apply after cropping and scaling)')
    group.add_argument('--cached', type=str, default='none', help='cache setting for the datasets')
    group.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    group.add_argument('--eval-batch-size', type=int, default=1, help='Evaluation batch size')
    group.add_argument('--workers', type=int, default=2, help='Number of workers for dataloader (default: 2)')
    group.add_argument('--pin-mem', type=str_to_bool, default=True, help='Pin memory for dataloader')
