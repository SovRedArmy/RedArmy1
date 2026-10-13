# HiNeRV 论文复现结果（HEVC Class B, 1920×1080, HiNeRV-S）

复现日期：2026-08-29 ~ 2026-08-31。配置 = 仓库 `cfgs/train/hinerv_1920x1080.txt` + `cfgs/models/uvg-hinerv-s_1920x1080.txt`
（论文主设定 HiNeRV-S，完整流水线：主训练 300e → 剪枝 15%+微调 60e → Quant-Noise QAT 30e → Q8/7/6 算术编码码流）。

## 每序列结果

| 序列 | 帧数 | full (参数/PSNR/MS-SSIM) | pruned PSNR | Q8 bpp/PSNR | Q7 bpp/PSNR | Q6 bpp/PSNR |
|---|---|---|---|---|---|---|
| Kimono1 | 240 | 2.85M / 34.24 / 0.9575 | 34.06 | 0.0398 / 33.43 | 0.0347 / 33.33 | 0.0300 / 33.59 |
| ParkScene | 240 | 2.85M / 31.11 / 0.9527 | 30.96 | 0.0395 / 30.38 | 0.0345 / 30.35 | 0.0301 / 30.61 |
| BQTerrace | 600 | 3.19M / 31.21 / 0.9650 | 31.14 | 0.0171 / 30.54 | 0.0150 / 30.58 | 0.0132 / 30.91 |
| Cactus | 500 | 3.09M / 31.37 / 0.9559 | 31.25 | 0.0203 / 30.57 | 0.0177 / 30.53 | 0.0153 / 30.85 |
| BasketballDrive | 500 | 3.09M / 31.34 / 0.9534 | 31.20 | 0.0205 / 30.57 | 0.0178 / 30.49 | 0.0154 / 30.83 |
| **平均** | — | **3.01M / 31.85 / 0.9569** | **31.72** | **0.0274 / 31.10** | **0.0239 / 31.06** | **0.0208 / 31.36** |

## 与论文结论的对照（趋势全部吻合）

1. **剪枝几乎无损**：剪掉 15% 权重 + 60e 微调后平均仅降 **0.13 dB**（论文的改进幅度评分剪枝 |θ|/P^0.5 有效）。
2. **6bit 量化可行**：Quant-Noise QAT 后 Q6 码流平均仅比 full 低 **0.49 dB**（论文以此把 INR 编码器从 8bit 推进到 6bit）。
3. **Q6 优于 Q7/Q8**：QAT 以最低比特（6bit）为目标训练，属预期现象。
4. **模型规模**：HiNeRV-S 2.85~3.19M 参数（fp32 ≈ 11~13MB），与论文 S 档一致。
5. **码率随帧数摊薄**：模型体积固定，bpp ∝ 1/帧数。BQTerrace（600 帧，与 UVG 同长度）Q6 码流 2.05MB / **0.0132 bpp**，
   与论文 UVG 600 帧 HiNeRV-S 压缩后的量级（~2MB、~0.013bpp）吻合。
6. PSNR 绝对值低于论文 UVG 平均（36.71 dB）属数据集差异：HEVC-B 含高频纹理内容（BQTerrace 水面等），
   且论文的 UVG 序列内容更平滑；跨数据集不能直接比绝对值。

## 工件位置

- 每序列最优结果：`repro/<序列>/…/results/{full,pruned,qat,Q8,Q7,Q6}.txt`
- 压缩码流（可解码评估）：`repro/<序列>/…/bitstreams/Q{6,7,8}.zip`
- 断点/最优模型：`repro/<序列>/…/checkpoints/checkpoint_best`
- 训练日志：`repro/<序列>/run.log`（或旧运行 `repro/*.log`）
- 启动/看护脚本：`repro/run_*.sh`、`repro/watchdog.sh`

## 用码流重新评估（解码端复现）

```bash
accelerate launch --mixed_precision=fp16 hinerv_main.py \
  --dataset /remote-home/share/HEVC_ClassB --dataset-name BQTerrace_1920x1080_60 \
  --output /tmp/eval --bitstream <输出目录> --bitstream-q 6 --eval-only \
  <train_cfg> <model_cfg>   # 配置同 run_bqterr_ace.sh
```
