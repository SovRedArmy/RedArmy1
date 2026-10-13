# v1 实验数据汇总


## beauty

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| ours<br>场景切分多 clip I/P(完整 v1 方法) | 28.10@41.19e-3 | 29.02@80.35e-3 | 30.62@155.79e-3 | 点数不足 |

## bosphorus

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| ours<br>场景切分多 clip I/P(完整 v1 方法) | 27.74@41.28e-3 | 29.29@80.18e-3 | 31.73@157.23e-3 | 点数不足 |

## bunny

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| base<br>base | 29.37@104.27e-3 | 32.54@204.53e-3 | 36.30@406.84e-3 | anchor |
| A1_ms<br>仅多尺度 max-pool 监督(+pair 采样) | 29.82@104.26e-3 | 32.20@204.75e-3 | 34.81@407.59e-3 | +6.0% (BD-PSNR -0.18dB) |
| A2_crspe<br>+CRSPE 每帧内容嵌入 | 29.75@108.83e-3 | 34.51@413.33e-3 | 点数不足 |
| A3_fft<br>+FFT 频域损失 | 29.62@108.85e-3 | 33.06@209.79e-3 | 34.58@413.41e-3 | +5.2% (BD-PSNR -2.86dB) |
| A4_flow<br>+LK 伪标签光流(带置信度) | 29.80@108.80e-3 | 32.11@209.75e-3 | 34.46@413.47e-3 | +12.0% (BD-PSNR -1.14dB) |
| A5_tcl<br>+TCL 时序对比 | 28.75@108.83e-3 | 30.81@209.55e-3 | 33.05@412.92e-3 | +54.0% (BD-PSNR -2.28dB) |
| A6_fw<br>+帧权重(损失调制) | 28.77@108.77e-3 | 30.80@209.51e-3 | 33.00@412.76e-3 | +54.3% (BD-PSNR -2.30dB) |
| A7_hfb<br>+高频增强目标 = 单模型全模块 | 28.82@108.80e-3 | 30.97@209.53e-3 | 33.27@412.96e-3 | +47.8% (BD-PSNR -2.15dB) |
| ours<br>场景切分多 clip I/P(完整 v1 方法) | 24.92@223.60e-3 | 27.41@435.39e-3 | 30.09@862.14e-3 | +604.1% (BD-PSNR -9.40dB) |
| ourspar<br>多 clip + 步数对齐 | 27.04@221.82e-3 | 点数不足 |

## honeybee

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| base<br>base | 38.43@20.74e-3 | anchor |

## jockey

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| base<br>base | 30.19@20.61e-3 | 32.94@41.12e-3 | anchor |
| ours<br>场景切分多 clip I/P(完整 v1 方法) | 24.40@40.87e-3 | 点数不足 |

## readysetgo

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| base<br>base | 21.97@10.44e-3 | 23.77@20.62e-3 | anchor |
| ours<br>场景切分多 clip I/P(完整 v1 方法) | 20.30@40.55e-3 | 22.42@79.17e-3 | 点数不足 |

## shakendry

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| base<br>base | 32.03@21.11e-3 | 33.50@41.65e-3 | anchor |

## yachtride

| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |
|---|---|---|
| base<br>base | 25.51@10.47e-3 | 27.01@20.70e-3 | 28.64@41.16e-3 | anchor |
