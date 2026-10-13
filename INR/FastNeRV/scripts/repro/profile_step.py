"""独立诊断：分解 NeRV-Enc 训练 step 的各阶段耗时（单卡、随机数据、无 DDP）。"""
import sys, time
import torch
import yaml

sys.path.insert(0, '/remote-home/gdzj21/VideoCompression/INR/FastNeRV')
import models

cfg = yaml.safe_load(open('/remote-home/gdzj21/VideoCompression/INR/FastNeRV/cfgs/nerv_enc_repro.yaml'))
model_args = cfg['model']['args']
model_args['tokenizer']['args']['input_size'] = 256
model_args['tokenizer']['args']['frame_num'] = 8
model_args['hyponet']['args']['frame_num'] = 8
model = models.make({'name': 'nerv_enc', 'args': model_args}).cuda()
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

from pytorch_msssim import ssim
import einops

B = 8
data = torch.rand(B, 3, 8, 256, 256).cuda()

def timed(fn, n=5):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n, out

def full_step():
    out = model(data)
    gt = einops.rearrange(out['gt_frames'], 'b c t h w -> t b c h w')
    for k, pred in out['out_frames'].items():
        flat_pred, flat_gt = pred.flatten(end_dim=1), gt.flatten(end_dim=1)
        psnr = -10*torch.log10(((pred.detach()-gt.detach())**2).flatten(-3).mean(-1)).mean()
        s = ssim(flat_pred, flat_gt, data_range=1)
    loss = sum((p - gt).pow(2).mean() for p in out['out_frames'].values()) + out.get('kl_loss', torch.tensor(0.))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

# 预热
for _ in range(3):
    full_step()
torch.cuda.synchronize()

t_full, _ = timed(full_step)
t_fwd, out = timed(lambda: model(data))
def fwd_loss_bwd():
    out = model(data)
    gt = einops.rearrange(out['gt_frames'], 'b c t h w -> t b c h w')
    (sum((p - gt).pow(2).mean() for p in out['out_frames'].values()) + out.get('kl_loss', torch.tensor(0.))).backward()
t_fb, _ = timed(fwd_loss_bwd)
def fwd_only():
    with torch.no_grad():
        return model(data)
t_inf, _ = timed(fwd_only)
t_ssim, _ = timed(lambda: ssim(data.flatten(end_dim=1), data.flatten(end_dim=1), data_range=1))

print(f'full_step   : {t_full*1000:.0f} ms')
print(f'fwd+bwd     : {t_fb*1000:.0f} ms')
print(f'forward only: {t_fwd*1000:.0f} ms')
print(f'no-grad fwd : {t_inf*1000:.0f} ms')
print(f'ssim(64frm) : {t_ssim*1000:.0f} ms')
print(f'optimizer+misc: {(t_full-t_fb)*1000:.0f} ms')
