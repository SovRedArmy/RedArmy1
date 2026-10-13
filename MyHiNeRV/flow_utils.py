"""
RAFT 伪标签光流工具（Scene Matters SFCM 的伪标签生成与加载）。

- precompute_flow（scripts/precompute_flow.py 调用）：用 torchvision 内置的
  raft_large 在半分辨率上离线估计相邻帧的双向光流，保存为 {flow_fwd, flow_bwd}.pt
  （[T, 2, H/2, W/2]，fp16，单位：半分辨率像素）。帧 t 的 fwd = RAFT(v_{t-1}, v_t)，
  bwd = RAFT(v_t, v_{t-1})；t = 0 无定义（首帧）。
- FlowProvider：数据集侧按 patch 坐标裁剪对应光流 patch 并放大回 patch 分辨率，
  flow 值按比例缩放（坐标尺度）。
注意：RAFT 仅作为训练期教师（伪标签），模型本身不含显式光流估计，头不进码流。
"""
import os
import math
import torch
import torch.nn.functional as F


def flow_file_path(dataset_root, dataset_name):
    """伪标签文件路径：{root}/{name}_flow.pt（root 与 dataset 同级目录）。"""
    return os.path.join(os.path.expanduser(dataset_root), f'{dataset_name}_flow.pt')


@torch.no_grad()
def precompute_flow(args, logger):
    """
    用 torchvision 的 raft_large 为整个视频生成相邻帧双向光流伪标签。
    输入帧读取复用 VideoDataset（含 crop/resize 预处理），RAFT 在半分辨率运行。
    """
    import torchvision

    from datasets import VideoDataset

    dataset = VideoDataset(logger, root=args.dataset, name=args.dataset_name,
                           crop=args.crop_size, resize=args.input_size,
                           patch_size=[-1, -1, -1], cached='image')

    T, H, W = dataset.video_size
    fh, fw = H // 2, W // 2  # RAFT 在半分辨率运行（省显存；流值单位=半分辨率像素）
    logger.info(f'Precomputing RAFT pseudo-label flow for {T} frames @ {fh}x{fw} (half of {H}x{W})')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raft = torchvision.models.optical_flow.raft_large(weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2,
                                                      progress=False).to(device).eval()

    def preprocess(img):
        # img: [3, H, W] in [0,1] -> [-1,1] half-res，尺寸对齐 8 的倍数且 ≥128（RAFT 相关性金字塔要求）
        x = F.interpolate(img[None], size=(fh, fw), mode='bilinear', align_corners=False)[0]
        pad_h = max((8 - x.shape[1] % 8) % 8, 128 - x.shape[1])
        pad_w = max((8 - x.shape[2] % 8) % 8, 128 - x.shape[2])
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return (x * 2. - 1.)[None].to(device), pad_h, pad_w

    flow_fwd = torch.zeros(T, 2, fh, fw, dtype=torch.float16)  # RAFT(v_{t-1}, v_t)
    flow_bwd = torch.zeros(T, 2, fh, fw, dtype=torch.float16)  # RAFT(v_t, v_{t-1})

    frames = [dataset.get_image(t).float() / 255. for t in range(T)] if T <= args.flow_cache_frames else None

    for t in range(1, T):
        i0 = frames[t - 1] if frames is not None else dataset.get_image(t - 1).float() / 255.
        i1 = frames[t] if frames is not None else dataset.get_image(t).float() / 255.
        p0, pad_h, pad_w = preprocess(i0)
        p1, _, _ = preprocess(i1)
        with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            fl = raft(p0, p1)[-1]  # [1, 2, H', W']  正向
            bl = raft(p1, p0)[-1]  # 反向
        fl = fl[0, :, :fl.shape[2] - pad_h, :fl.shape[3] - pad_w]
        bl = bl[0, :, :bl.shape[2] - pad_h, :bl.shape[3] - pad_w]
        if fl.shape[-2:] != (fh, fw):
            fl = F.interpolate(fl[None], size=(fh, fw), mode='bilinear', align_corners=False)[0]
            bl = F.interpolate(bl[None], size=(fh, fw), mode='bilinear', align_corners=False)[0]
        flow_fwd[t] = fl.half().cpu()
        flow_bwd[t] = bl.half().cpu()
        if t % 50 == 0:
            logger.info(f'    flow {t}/{T}')

    out_path = flow_file_path(args.dataset, args.dataset_name)
    torch.save({'flow_fwd': flow_fwd, 'flow_bwd': flow_bwd, 'size': (fh, fw)}, out_path)
    logger.info(f'Saved pseudo-label flow to {out_path}')
    return out_path


class FlowProvider:
    """
    按补丁坐标提供光流伪标签（供训练时的 SFCM 约束）。
    半分辨率流图被双线性放大到 patch 分辨率，flow 值同步按比例缩放。
    帧级有效性：clip 内首帧（t=0）无伪标签（valid=0）。
    """
    def __init__(self, flow_path, video_size=None, frame_offset=0):
        data = torch.load(flow_path, map_location='cpu')
        self.fwd = data['flow_fwd']  # [T_global, 2, fh, fw]（保持 fp16 省内存）
        self.bwd = data['flow_bwd']
        self.video_size = video_size  # (T_clip, H, W)；可为 None，稍后由数据集回填
        self.frame_offset = frame_offset  # clip 首帧在全局视频中的位置
        if video_size is not None:
            self._check()

    def _check(self):
        T_g = self.fwd.shape[0]
        assert T_g >= self.frame_offset + self.video_size[0], 'flow pseudo-labels do not cover the clip'

    def get_patch_flow(self, idx_thw, patch_size):
        """
        idx_thw: (t, h, w) 补丁坐标（clip 内局部坐标）；patch_size: (Pt, Ph, Pw)。
        返回 (flow_fwd [2, Pt, Ph, Pw], flow_bwd, valid标量bool)。
        """
        t0, h0, w0 = idx_thw[0] + self.frame_offset, idx_thw[1], idx_thw[2]
        Pt, Ph, Pw = patch_size
        T, H, W = self.video_size

        if t0 == 0:
            # clip 首帧无伪标签（其 fwd/bwd 参考上一 clip 的帧，不监督）
            z = torch.zeros(2, Pt, Ph, Pw)
            return z, z.clone(), False

        # 半分辨率流图上的对应区域（向下取整，再用 interpolate 对齐）
        fh, fw = self.fwd.shape[-2:]
        sy, sx = fh / H, fw / W
        y0, x0 = int(h0 * Ph * sy), int(w0 * Pw * sx)
        y1, x1 = int((h0 + 1) * Ph * sy), int((w0 + 1) * Pw * sx)
        y1, x1 = max(y1, y0 + 1), max(x1, x0 + 1)
        f = self.fwd[t0:t0 + Pt, :, y0:y1, x0:x1].float()  # [Pt, 2, fh', fw']
        b = self.bwd[t0:t0 + Pt, :, y0:y1, x0:x1].float()
        if Pt == 1:
            f = F.interpolate(f[0][None], size=(Ph, Pw), mode='bilinear', align_corners=False)[0].unsqueeze(1)  # [2,1,Ph,Pw]
            b = F.interpolate(b[0][None], size=(Ph, Pw), mode='bilinear', align_corners=False)[0].unsqueeze(1)
        else:
            # [Pt,2,fh,fw] -> [1,2,Pt,fh,fw]（trilinear 需要 5D）-> [2,Pt,Ph,Pw]
            f = F.interpolate(f.permute(1, 0, 2, 3)[None], size=(Pt, Ph, Pw), mode='trilinear', align_corners=False)[0]
            b = F.interpolate(b.permute(1, 0, 2, 3)[None], size=(Pt, Ph, Pw), mode='trilinear', align_corners=False)[0]
        # 流值单位是半分辨率像素；放大到 patch 分辨率时按比例缩放
        # （通道 0 = Δx 对应宽度，通道 1 = Δy 对应高度）
        f[0] *= (W / fw)
        f[1] *= (H / fh)
        b[0] *= (W / fw)
        b[1] *= (H / fh)
        return f, b, True
