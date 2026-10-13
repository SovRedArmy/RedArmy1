"""
Datasets（数据集：把一段视频包装成逐 3D patch 读取的 PyTorch Dataset）
"""
from utils import *
from io_utils import *
import bisect


class VideoDatasetBase(torch.utils.data.Dataset):
    """
    The base dataset class for loading videos.
    It loads video frames in 3D patches.

    （视频数据集基类：将整段视频按 patch_size 切成若干 3D patch，
      __getitem__(idx) 返回 (patch 坐标 idx_thw, 归一化后的 patch 张量 [C, T, H, W])。）
    """
    def __init__(self, logger, video, channel_scale=None, channel_shift=None, is_subset=False):
        self.logger = logger
        self.video = video

        assert channel_scale is None or channel_shift is None or \
               (len(channel_scale) == self.video.get_num_channels() and \
                len(channel_shift) == self.video.get_num_channels())
        self.channel_scale, self.channel_shift = compute_scale_shift(self.video.get_num_channels(),
                                                                     self.video.get_bit_depth(),
                                                                     channel_scale, channel_shift)

        if not is_subset:
            self.logger.info(f'{type(self).__name__}:')
            self.logger.info(f'     Path: {self.video.path}')
            self.logger.info(f'     Video size: {self.video.get_video_size()}    Patch size: {self.video.get_patch_size()}    Padding: {self.video.get_padding()}')
            self.logger.info(f'     Channels: {self.video.get_num_channels()}    Bit depth: {self.video.get_bit_depth()}')
            self.logger.info(f'     Scale: {self.channel_scale}    Shift: {self.channel_shift}')

    def get_video_size(self):
        return self.video.get_video_size()

    def get_patch_size(self):
        return self.video.get_patch_size()

    def get_num_channels(self):
        return self.video.get_num_channels()

    def get_start_frame(self):
        return self.video.get_start_frame()

    def get_num_frames(self):
        return self.video.get_num_frames()

    def set_frames(self, start_frame, num_frames):
        self.video.set_frames(start_frame, num_frames)

    def create_cache(self):
        self.video.create_cache()

    def _get_patch(self, idx_thw):
        raise NotImplementedError

    def __len__(self):
        # 数据集长度 = 视频（含 padding）能切出的 patch 总数
        return math.prod(self.video.get_idx_max())

    def __getitem__(self, idx):
        # 把一维下标展成 3D patch 坐标 (t, h, w)，并读取对应 patch
        assert isinstance(idx, int)
        idx_max = self.video.get_idx_max()
        idx_thw = (idx // (idx_max[1] * idx_max[2]),
                   (idx % (idx_max[1] * idx_max[2])) // idx_max[2],
                   (idx % (idx_max[1] * idx_max[2])) % idx_max[2])
        return torch.tensor(idx_thw, dtype=int), self._get_patch(idx_thw)


class PNGVideoDataset(VideoDatasetBase):
    """
    The dataset class for loading videos (in PNG files). Each dataset instance loads all video frames in a folder.

    （PNG 视频数据集：读取一个文件夹内的 PNG 序列，
      _get_patch 把整数值像素归一化到 [0, 1] 区间。）
    """
    def _get_patch(self, idx_thw):
        scale = self.channel_scale.view(self.video.get_num_channels(), 1, 1, 1)
        shift = self.channel_shift.view(self.video.get_num_channels(), 1, 1, 1)
        # [T, H, W, C] -> [C, T, H, W]，并做 x = x * scale + shift 归一化
        rgb_patch = torch.tensor(self.video.read_patch(idx_thw)).permute(3, 0, 1, 2).float() * scale + shift
        return torch.clip_(rgb_patch, 0., 1.)


class YUVVideoDataset(VideoDatasetBase):
    """
    The dataset class for loading videos (in YUV format). Each dataset instance loads all video frames from a yuv file.

    （YUV 视频数据集：从原始 YUV 文件读取 patch，
      将 YUV420 上采样为 YUV444 送入网络（训练 4:4:4，评估 4:2:0）。）
    """
    def _get_patch(self, idx_thw):
        scale = self.channel_scale
        shift = self.channel_shift

        # List of [T, 1, H, W] tensors（Y/U/V 三个平面分别归一化）
        yuv420_patch = [torch.tensor(patch_i)[:, None].float() * scale[i] + shift[i] \
                        for i, patch_i in enumerate(self.video.read_patch(idx_thw))]
        # List of [T, 1, H, W] tensors -> [T, 3, H, W] tensor -> [3, T, H, W] tensor（420 -> 444 上采样）
        yuv444_patch = yuv420_to_yuv444(yuv420_patch, mode='nearest').permute(1, 0, 2, 3)

        return torch.clip_(yuv444_patch, 0., 1.)


class VideoSubset(torch.utils.data.Subset):
    """（视频子集：只保留 [start_frame, start_frame + num_frames) 内的 patch，用于按 GoP 训练）"""
    def __init__(self, dataset, start_frame, num_frames):
        super().__init__(dataset, [])
        self.dataset = dataset
        self.indices = []
        dataset = self.dataset
        patches_per_frame = math.prod(dataset.video.get_idx_max()[1:])
        assert start_frame % dataset.get_patch_size()[0] == 0 and num_frames % dataset.get_patch_size()[0] == 0, \
            f'start_frame and num_frames should be multiples of patch_size[0]'
        start_patch = (start_frame // dataset.get_patch_size()[0]) * patches_per_frame
        num_patches = (num_frames // dataset.get_patch_size()[0]) * patches_per_frame
        self.indices += list(range(start_patch, start_patch + num_patches))

    def get_video_size(self):
        return self.dataset.get_video_size()

    def get_patch_size(self):
        return self.dataset.get_patch_size()


def create_overfit_dataset(args, logger, video, channel_scale=None, channel_shift=None, is_subset=False):
    # Create dataset（按输入视频格式创建对应的数据集：PNG 序列或原始 YUV）
    if isinstance(video, PNGVideo):
        dataset = PNGVideoDataset(logger, video, 
                                  channel_scale=channel_scale, channel_shift=channel_shift,
                                  is_subset=is_subset)
    elif isinstance(video, YUVVideo):
        dataset = YUVVideoDataset(logger, video, 
                                  channel_scale=channel_scale, channel_shift=channel_shift,
                                  is_subset=is_subset)
    else:
        raise ValueError
    return dataset