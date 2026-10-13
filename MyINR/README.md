# MyINR —— 在 NeRV 基础上的神经视频表示改进实验

在 NeRV(NeurIPS 2021) 基础上实现五类改进, 目标是改善 RD 性能(BD-RATE)。
原 INR/NeRV 目录保持不动; 全部实验在本目录进行。

## 方法总览(对应任务书五点)

### 一、Scene Matters(MVC, ICCV'23) 的四个模块
| 模块 | 实现 | 文件 |
|---|---|---|
| CRSPE 每帧内容嵌入 | learnable per-frame 空间码 (N, Ce=4, 9, 16), 经 1x1 conv 融合进 MLP 输出特征图; 计入码率(逐帧 8bit 量化+算术编码) | `modules.py: frame_embed/embed_proj` |
| FFT 频域损失 | L1(‖FFT(recon)‖, ‖FFT(GT)‖), ortho 归一化, λ=0.1, 零码率 | `losses.py: fft_loss` |
| RAFT 风格伪标签光流约束 | 金字塔迭代 LK 估计器(本机离线, 替代需下载权重的 RAFT)生成 1/4 分辨率伪标签; 解码器 stage-2 特征→光流头; 1x1 conv 置信度头(softmax)+熵二值化正则, 自动降权噪声标签; 光流/置信头为辅助头, 不计码率 | `pseudo_flow.py`, `losses.py: flow_loss` |
| TCL 时序对比损失 | 固定 ImageNet AlexNet(本机缓存)嵌入, EMA 记忆库 + 高斯先验(σ²=10) InfoNCE, τ=0.1, 窗口 32 | `losses.py: tcl_loss, EmbeddingBank` |

### 二、dynamicAL 帧权重
小探针 NeRV 短训 8 epoch(batch=1 单帧, 每步梯度范数=该帧 per-sample 梯度范数),
得每帧训练动态 g_t; robust z-score 后线性映射到 [0.5, 2.0]、均值 1 的帧权重,
训练中加权逐帧 L1 损失(`dynamics.py: probe_dynamics, frame_weights`)。

### 三、频率尺度分解(MSNeRV/FaNeRV)
- 每个 stage 的中间特征经 head 投影成小视频, 与 **max-pooling** 下采样 GT 金字塔逐尺度算
  尺度自适应损失 α·MSE + β·L1 + γ·(1-MS-SSIM)(低分辨率偏 MSE, 高分辨率加 MS-SSIM);
- 高频增强: 最终尺度目标 V_boost = V_gt + (r − blur(r)), r = GT − recon(detach),
  训练 30% 后启用避免早期残差爆炸(`losses.py: scale_adaptive_loss, high_freq_boosted_target`)。

### 四、场景切分(TeNeRV 风格)
边界得分 = 0.5·z(相邻帧 AlexNet 特征余弦距离) + 0.5·z(|g_t − g_{t−1}| 训练动态突变),
贪心取 top-K 边界(最小段长约束)切成 K 个 clip(`dynamics.py: scene_split`)。

### 五、多模型 + I/P + DCVC 风格压缩
- 每 clip 训练一个较小 INR: clip0=I-model 从零训(150ep, 渐进剪枝到 keep=0.4);
  clipk=P-model 从上一模型的**反量化权重**初始化微调(60ep, lr 2.5e-4,
  最后 30% epoch 权重差 Δw L1 正则诱导稀疏);
- 压缩: I-model dense 量化; P-model 编码 Δw = w_P − w_prev(幅值阈值 keep=0.5),
  逐通道 8bit 量化 + **算术编码**(DCVC 风格, 零值计入直方图→掩码成本被信息论包含);
  解码端 w_P = dequant(P_{k−1}) + dequant(Δw) 级联自洽;
- bpp = 总比特(I + ΣΔ + 帧嵌入) / (N·H·W)。多 clip 小模型解码 FPS 更高。

## 关键文件
- `modules.py` NeRV++ 生成器(CRSPE/多尺度头/辅助光流头), 与原版 NeRV checkpoint 命名兼容
- `trainer.py` ClipTrainer(baseline 模式完全退化为原版 NeRV 协议)
- `encode_video.py` 单视频×单码率点完整流水线
- `dynamics.py` 探针训练动态/帧权重/场景切分; `pseudo_flow.py` 伪标签光流
- `compress.py` 剪枝/量化/权重差编码; `arithmetic_coder.py` CACM87 算术编码
- `bd_rate.py` PCHIP BD-rate(纯 numpy); `collect.py` 汇总; `scheduler.py` GPU 任务调度
- `embedder.py` ImageNet AlexNet 固定特征(本机缓存权重)

## 实验协议(与用户 NeRV 复现一致)
NeRV-S/M/L = stem 512_1, fc 9_16_{26,58,112}, lower 96, expansion 1, reduction 2;
150 epoch, warmup 30, Adam 5e-4, cosine, Fusion6(baseline); 量化 8bit 逐通道;
剪枝 keep 0.4 渐进(0.5/0.625/0.75/0.875 分位); 算术熵编码。
数据: bunny(720p,132帧) + UVG 7 序列(1080p, 3900 帧)。
BD-rate anchor = 同协议 MyINR-baseline 三个码率点。

## 运行
```bash
python encode_video.py --data data/bunny --mode ours --preset S    # ours
python encode_video.py --data data/bunny --mode baseline --preset S # baseline
python encode_video.py --data data/bunny --mode ours --disable flow,tcl --tag abl_no_flow ...
python scheduler.py jobs_main.txt --gpus 0,1,... --jpu 2
python collect.py
```

---

# v2 分支: P0-P6 改进(无显式光流模块)

对照 v1: `git diff master..v2`; v1 结果: `runs/`(tag base_*/A*/ours_*), v2 结果 tag v2_*。

| 改进 | 实现 | 消融开关 |
|---|---|---|
| P0 GOP/GOM 解耦 | `--clip-mode gop`(默认): 固定短 GOP(75/44 帧) + 每 gom-size 个 GOP 一个 I-model; 场景切换点只用于重置 I 锚点(新 GOM), 不再做 clip 边界 | `--clip-mode auto`=v1 场景切分对照 |
| P1 RQEC-lite | P-model 训练损失 + rate_weight×Σ numel·log2(2e·mean\|Δθ\|)/Σnumel(Laplace 熵率) | `--rate-weight 0` 退回 v1 delta-L1 |
| P2 II 初始化 | θ′=ε·θ_fresh+(1−ε)·θ_prev, ε=clip(ii_scale·√adjMSE, 0.15, 0.85) | `--no-ii` 退回 D-init |
| P3 多尺度加权退火 | 中间尺度 ms_lw=0.3, 60% epoch 后退火到 0, 最终尺度恒 1 | `--ms-lw 1`(近似v1) |
| P4 辅助损失退火+门控 | FFT/TCL 最后 30% 退火到 0; HFB 80% 后关闭; TCL 远帧GT相似度>0.75 自动关闭(bunny=0.945→OFF) | `--tcl-gate 0` |
| P5 步数对齐+差分先阈值后恢复 | 每 clip epoch=base·N/(2·n_clip)(≤max-epochs); 差分在 85% epoch 阈值化后带掩码恢复训练 15% | `--steps-parity 0` |
| P6-lite 位深选择 | 压缩时逐张量按相对误差容差(tol=0.05)在 4-8bit 中选最小位深 | compress 参数 |
| 移除光流模块 | 无伪标签/光流头/置信度头(pseudo_flow.py 保留在库中不再调用) | — |

v2 测试(GPU 空闲后自动触发, /tmp/v2_trigger.sh): bunny v2 S/M/L、v2auto(隔离P0贡献)、
v2noii/v2norate(隔离P2/P1贡献)、beauty v2 S。
