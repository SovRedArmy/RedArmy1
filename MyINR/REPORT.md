# MyINR 实验报告 —— 在 NeRV 基础上的改进与 BD-RATE 评估

(实验进行中, 本文件将在全部实验完成后填充最终数据)

## 1. 目标与总体结论

**目标**: 在 NeRV(NeurIPS 2021) 基础上加入五类改进(Scene Matters 四模块 / dynamicAL 帧权重 /
频率尺度分解 / TeNeRV 场景切分 / 每-clip 多模型 I/P + 剪枝-量化-算术编码), 在 bunny + UVG 上
以与用户 NeRV 复现完全一致的协议(NeRV-S/M/L, 150 epoch, warmup 30, Adam 5e-4 cosine, Fusion6,
8bit 逐通道量化)检验 RD 性能(BD-RATE, PCHIP)。

**结论**: (待填)

## 2. 方法与实现要点

见 README.md。关键工程决策:
- 无法联网下载 RAFT/ResNet 预训练权重 → 金字塔迭代 LK 光流伪标签(合成平移测试验证 + 常数运动
  恢复), ImageNet AlexNet(本机缓存)固定特征用于场景切分与 TCL;
- 帧权重(dynamicAL)必须"温和 + 尺度不变": 大范围权重(如 [0.5,2])使低权重帧欠拟合, 反而
  毁掉 PSNR(bunny 诊断: 16.6dB vs 24.1dB); 最终采用 [0.8,1.2] + 逐批均值中心化;
- TCL 需 λ=0.005 + 训练 20% 后启用 (λ=0.05 立即启用会 −3.6dB);
- 渐进剪枝每步应保留"当前剩余"的 keep^(1/n), 总保留才等于 keep;
- 逐通道量化必须对称(scale=|w|max/2^(b-1)), 使剪枝/权重差的 0 恒对齐同一符号, 熵编码才能
  近乎免费地编码稀疏掩码(修复前 7.2 bits/param → 修复后 4.7)。

## 3. 中期诊断结论(bunny S, 150ep 完整协议, 逐模块分解)

| 配置 | PSNR(dB) | bpp | 说明 |
|---|---|---|---|
| base (NeRV-S 协议) | 29.37 | 0.1043 | 剪枝40% + 8bit 对称量化 + 算术编码 |
| A1 = base + 多尺度监督(pair 训练) | **29.82** | 0.1043 | **同码率 +0.45dB** —— 多尺度 max-pool 金字塔监督稳定有效 |
| A7 = A1 + crspe+fft+flow+tcl+fw+hfb | 28.82 | 0.1088 | 四个 Scene-Matters 辅助损失+HFB+帧权重合计 **−1.0dB**(且各诊断均无害于 40ep 短训, 损失出现在长训收敛段) |
| full = A7 + 场景切分 3-clip I/P | 24.92 | 0.2236 | 多 clip 在 bunny 上失败: −4.9dB 且码率 ×2 |

诊断实验(W/X/Y 系列, 见 logs/diag_*.log)定位的机理:
1. **帧权重(dynamicAL)必须温和**: [0.5,2.0] 范围使低权重帧欠拟合(clip0 40ep: 16.6dB vs 24.1dB);
   均匀放大权重也会因实现不平移不变而 −1.5dB → 最终 [0.8,1.2] + 逐批中心化后近零损失。
2. **TCL**: λ=0.05 立即启用 −3.6dB; λ=0.005 + 20% 后启用 ≈ 无损。TCL 是唯一在 40ep 诊断中
   显著有害的模块。
3. **多 clip 的失败机理**: 固定"150 epoch"下, 每-clip 模型得到的优化步数按比例减少
   (41 帧 clip 只有 6k 步 vs 整段 19.8k 步); 纯 baseline 协议在 clip0 上也只到 26.94dB
   (vs 整段 29.37) —— 是步数饥饿而非容量/损失问题。P-model(50-100ep 微调)进一步欠拟合
   (clip1 仅 23dB)。场景切分本身(结合内容余弦距离+训练动态)在 bunny 上切出 [0,41,68,132],
   边界与 bunny 的镜头切换吻合。
4. **压缩工程**: 渐进剪枝每步保留 keep^(1/n)(否则连乘过度剪枝, −2.7dB); 逐通道量化必须对称
   (否则 0 分散到不同符号, 熵编码失效, 7.2→4.7 bits/param); 算术编码逼近符号熵(≤0.3% 差距)。

(UVG 1080p 结果待全部跑完后填入 §4)



## 4. 消融实验 (bunny, 3 码率点)

| 配置 | 说明 | BD-rate |
|---|---|---|
| A1 ms | 仅多尺度 max-pool 金字塔监督 | |
| A2 crspe | + 每帧内容嵌入 | |
| A3 fft | + FFT 频域损失 | |
| A4 flow | + 光流伪标签约束 | |
| A5 tcl | + 时序对比损失 | |
| A6 fw | + dynamicAL 帧权重 | |
| A7 hfb | + 高频增强目标(单模型全模块) | |
| A8 full | + 场景切分多 clip I/P(完整方法) | |
| A9 fixedclips | A8 但等长切分(无场景检测) | |
| A10 indclips | A8 但各 clip 独立训练/编码(无 P-model 权重差) | |

## 5. 复现命令

见 jobs_wave1.txt / jobs_wave2.txt / jobs_ablation.txt 与 scheduler.py。
