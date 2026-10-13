'''
Metrics & Losses
like: ['mse', 'psnr', 'ssim', 'ms-ssim']
'''
import torch
import math

import torch.nn             as nn
import torch.nn.functional  as F
from pytorch_msssim     import ms_ssim, ssim
from tensorboardX       import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class metricsBase(AverageMeter):
    def __init__(self, name: str='mse'):
        super().__init__()
        self.metricName = name
        
    def update(self, val, n=1):
        return super().update(self._item_format(val), n)
        
    def stat_dict(self):
        return {self.metricName: self.avg}
    
    def tensorboard_renew(self, 
                          writer: SummaryWriter, 
                          iters: int=0):
        writer.add_scalar(self.metricName, self.avg, iters)
        
    def metrics_round(self, n: int=3):
        self.avg = round(self.avg, n)
        
    @staticmethod
    def _item_format(input0):
        return input0.item() if torch.is_tensor(input0) else input0



class metrics:
    def __init__(self, metrics_str_set):
        self.metrics_str_set = ['loss']  # for essential metrics
        self.metrics_str_set += metrics_str_set
        
        self.metrics_set = []
        for metric in self.metrics_str_set:
            self.metrics_set.append(metricsBase(name=metric))
            support_metrics_list    = ['loss', 'bpp', 'mse', 'psnr', 'ms_ssim',
                                       'perplexity', 'vqloss', 'positive_nums',
                                       'weight_bpp', 'linear_bpp', 'bias_bpp',
                                       'side_loss']
            assert metric in support_metrics_list, 'not support metrics: {}'.format(metric)

    def reset(self):
        for metric in self.metrics_set:
            metric.reset()

    def update(self, out_criterion):
        for i, metric in enumerate(self.metrics_set):
            metric.update(val=out_criterion[self.metrics_str_set[i]])

    def stat_dict(self, stat='train'):
        dictionary = {'stat': stat}
        for metric in self.metrics_set:
            dictionary.update(metric.stat_dict())
        return dictionary

    def tensorboard_renew(self, 
                          writer: SummaryWriter, 
                          iters: int=0):
        for metric in self.metrics_set:
            metric.tensorboard_renew(writer, iters)

    def metrics_round(self, n: int=3):
        for metric in self.metrics_set:
            metric.metrics_round()
            
    def get_loss(self):
        return self.metrics_set[0].avg


def PSNR(input1, input2):
    delta = input1 - input2
    mse = torch.mean(delta.reshape(input1.size(0), -1) ** 2)
    psnr = -10 * torch.log10(mse)
    return torch.mean(psnr).item()


class RateDistortionLoss(nn.Module):
    def __init__(self, num_pixels=1920*1080*10, Lambda=0.1):
        super().__init__()
        self.Lambda      = Lambda
        self.num_pixels = num_pixels
        
    def mse_loss(self, x, y):
        assert x.shape == y.shape, 'shape of tensors must be the same!'
        if x.ndim == y.ndim == 4:
            return F.mse_loss(x, y)
        elif x.ndim == y.ndim == 5:
            N, C, T, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
            y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
            return F.mse_loss(x, y, reduction='none').mean(dim=2)
        else:
            raise ValueError('inputs are expected to have 4D or 5D ([N, C, H, W]) or ([N, C, T, H, W])')
    
    def l1_loss(self, x, y):
        assert x.shape == y.shape, 'shape of tensors must be the same!'
        if x.ndim == y.ndim == 4:
            return F.l1_loss(x, y)
        elif x.ndim == y.ndim == 5:
            N, C, T, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
            y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
            return F.l1_loss(x, y, reduction='none').mean(dim=2)
        else:
            raise ValueError('inputs are expected to have 4D or 5D ([N, C, H, W]) or ([N, C, T, H, W])')
        
    def ms_ssim_loss(self, x, y, win_size=5):
        assert x.shape == y.shape, 'shape of tensors must be the same!'
        if x.ndim == y.ndim == 4:
            return 1. - ms_ssim(x, y, data_range=1., size_average=False)
        elif x.ndim == y.ndim == 5:
            N, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
            y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
            return 1. - ms_ssim(x, y, data_range=1., win_size=win_size, size_average=False).view(N, T)
        else:
            raise ValueError('inputs are expected to have 4D or 5D ([N, C, H, W]) or ([N, C, T, H, W])')

    def fft_loss(self, x, y):
        assert x.shape == y.shape, 'shape of tensors must be the same!'
        if x.ndim == y.ndim == 5:
            N, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
            y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
        pred, target = x, y
        
        pred_freq   = torch.fft.fft2(pred, dim=(-2, -1))
        pred_freq   = torch.stack([pred_freq.real, pred_freq.imag], -1)
        
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)
        
        freq_loss   = F.l1_loss(pred_freq, target_freq, reduction='none').flatten(1).mean(1)
        return freq_loss
    
    def forward(self, output, target):
        if len(target.size()) == 4:
            B, C, H, W  = target.size()
        else:
            B, C, T, H, W  = target.size()
        out         = {}
        out["bpp"]  = 0
        zero_scalar_cuda        = torch.tensor(0, dtype=torch.float32).to(output["x_hat"].device)
        bpp_list                = [(torch.log(likelihoods).sum() / (-math.log(2) * self.num_pixels) if likelihoods != None else zero_scalar_cuda)
                                for likelihoods in output["likelihoods"]]
        out["bpp"]              = sum(bpp_list)  # BPP comes from network parameters
        # if output["likelihoods"] != [] and output["likelihoods"][0] != None:
        #     bit = [(torch.log(likelihoods).view(-1) / (-math.log(2)) if likelihoods != None else zero_scalar_cuda) for likelihoods in output["likelihoods"]]
        #     bit_per_param       = torch.concat(bit, dim=0).mean()
        # else:
        #     bit_per_param = 0.
        
        out["psnr"]         = PSNR(output["x_hat"], target)
        
        output["x_hat"], target = output["x_hat"], target
        
        out["mse"]          = self.mse_loss(output["x_hat"], target).mean() * (255 ** 2)
        out["ssim"]         = ssim(output["x_hat"], target)
        # out["ms_ssim"]      = ms_ssim(output["x_hat"], target, data_range=1.).clamp_max_(1.) if (H >= 160 and W >= 160) else 0
        out["ms_ssim"]      = (1. - self.ms_ssim_loss(output["x_hat"], target)) if (H >= 160 and W >= 160) else 0
        l_d                 = 0.7 * self.l1_loss(output["x_hat"], target) + \
                              0.3 * self.ms_ssim_loss(output["x_hat"], target, win_size=5)
        # [repro patch] HiNeRV 5D 输出时 l1/ms_ssim 返回 (N,T) 矩阵 → loss 非标量, backward 必崩;
        # 与 4D(NeRV) 路径的 mean 语义对齐(此前仅 NeRV 骨干被复现验证过)
        if torch.is_tensor(l_d) and l_d.dim() > 0:
            l_d = l_d.mean()
                              
        # l_d_m = 60 * l_d.mean() + self.fft_loss(output["x_hat"], target).mean()
        
        out["loss"] = out["bpp"] + self.Lambda * l_d
        # if bit_per_param > 6.:
        #     out["loss"] = out["bpp"] + self.Lambda * l_d
        # else:
        #     out["loss"] = l_d
        return out





















