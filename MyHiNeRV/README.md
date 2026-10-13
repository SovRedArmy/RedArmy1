# MyHiNeRV

基于 [HiNeRV](https://github.com/hmkx/HiNeRV)（NeurIPS 2023）改进的 INR 视频压缩框架，
融合 Scene Matters / dynamicAL / FaNeRV·MSNeRV / TeNeRV / UAR-NVC / DCVC-RT 的可借鉴技术。

## 总体架构：两级（低动态 base + 高动态 GOP 小模型）

```
Phase A  低动态 base：全视频训练一个"宽而深"的 MyHiNeRV（--base-channels/--base-depths），
         学习整段视频的通用信息；剪枝 + QAT + intra 压缩 → 编码一次、全帧共享（进码流）。
         训练全程记录 逐 patch 训练动态（dynamicAL 得分 μ+λσ）。
Phase B  自适应 GOP 切分：内容余弦散度（TeNeRV D_t）+ 训练动态得分融合，
         贪心选择 K−1 个边界（最小长度约束）。不使用 UAR-NVC 的固定帧数切分。
Phase C  高动态 GOP 小模型（每个 GOP 一个，"小而轻"）：
         - 参数由 base 宽模型确定性推导（derive.py：通道/深度切片 + 时间维插值）；
           所有 GOP 只依赖 base、彼此独立 —— 避免 UAR-NVC 最差变体 D-init 的顺序依赖；
         - 单独完整训练（--p-epochs 默认 = --epochs，补足训练轮数）；
         - 残差编码：量化 θ_gop − derive(base 解码权重)，只传残差（noise 模式
           dead-zone 跳过小残差 / qec 模式可学习 q_step + 高斯熵编码）。
Phase D  汇总 RD 点：各量化级 总字节 = base + Σ GOP 残差 → bpp；GOP 逐帧指标 → 全局 PSNR/MS-SSIM。
```

## 模块与来源对照

| 模块 | 借鉴来源 | 代码位置 |
|------|----------|----------|
| CRSPE 每帧内容嵌入（GT 内容初始化，拼接注入） | Scene Matters (ICCV 2023) | `models/myhinerbv.py: CRSPE` |
| FFT 频域损失（零码率，可选高频加权） | Scene Matters | `losses.py: fft_loss` |
| RAFT 伪标签光流约束 + 可学习置信度（头不进码流，无显式光流估计） | Scene Matters SFCM | `flow_utils.py`, `models/myhinerbv.py: aux` |
| 时序对比损失（时间高斯软标签 InfoNCE，作用于 CRSPE） | Scene Matters TCL | `losses.py: temporal_contrastive_loss` |
| 帧权重采样 + **patch 丢弃 25%**（训练动态 μ+λσ，损失内帧等权） | dynamicAL (NeurIPS 2022) + MSNeRV patch-dropout | `dynamics.py: PatchDynamicsTracker / compute_patch_drop_mask` |
| 多尺度金字塔损失 + 动态高频注入 | FaNeRV / MSNeRV | `models/myhinerbv.py: AuxScaleHead`, `losses.py` |
| 自适应 GOP 切分（余弦散度 + 训练动态，TeNeRV 贪心算法） | TeNeRV + dynamicAL（非 UAR-NVC 固定切分） | `dynamics.py: segment_clips` |
| 两级 base/GOP 架构、GOP 独立推导（非 D-init） | UAR-NVC（架构思想）+ 本文设计 | `myhinerbv_main.py`, `derive.py` |
| 残差预测编码（θ − derive(θ_base_dec)）+ dead-zone 跳过 + scale 档位表 | DCVC-RT | `compression/quant_utils.py`, `compression/codec_utils.py` |
| QEC：每层可学习 q_step + 高斯熵率估计（RD 训练）+ 高斯熵编码 | UAR-NVC QEC | `compression/quant_utils.py: QuantEntropy`, `codec_utils.py: _compress_tensor_qec` |

## 安装

```bash
conda activate NVRC   # torch 2.6 + torchvision 0.21(含RAFT) + accelerate + timm + torchac
# torchac 需要 ninja：export PATH=/remote-home/gdzj21/miniconda3/envs/NVRC/bin:$PATH
pip install -r requirements.txt
```

## 用法

### 1) 预计算 RAFT 光流伪标签（每个视频一次）

```bash
python scripts/precompute_flow.py --dataset <root> --dataset-name <name> --input-size 1080 1920
# 输出 <root>/<name>_flow.pt（[T,2,H/2,W/2] fp16 双向流；缺失时 SFCM 自动关闭）
```

### 2) 两级管线（base → 自适应 GOP → 小模型 → RD 汇总）

```bash
dataset_dir=~/Datasets/UVG/1920x1080; dataset_name=ReadySetGo
train_cfg=$(cat cfgs/train/myhinerbv_1920x1080.txt)
model_cfg=$(cat cfgs/models/uvg-myhinerbv-s_1920x1080.txt)   # 小模型（GOP）结构
python myhinerbv_main.py --dataset ${dataset_dir} --dataset-name ${dataset_name} \
    --output output --batch-size 144 --eval-batch-size 1 --seed 0 ${train_cfg} ${model_cfg}
# base 宽模型结构由 --base-channels/--base-depths/--base-exps 覆盖（默认在 train cfg 中）
```

输出目录结构：
```
output/<exp>/
├── clips.yaml                    # 自适应 GOP 边界 + 散度/动态分数
├── base/                         # 低动态宽模型（intra 码流 + 解码权重 + patch 动态）
│   ├── bitstreams/Q{8,7,6}.zip
│   ├── decoded_state.pt
│   └── patch_dynamics.npy
├── gop00_0000-0007/              # 高动态小模型（残差码流；参考 = derive(base 解码权重)）
│   ├── bitstreams/Q{8,7,6}.zip
│   └── gop_summary.yaml
├── ...
└── summary.yaml / summary.txt    # 全局 RD 点（总 bpp = base + Σ GOP 残差）
```

### 3) 单模型调试入口

```bash
python hinerv_main.py --dataset ... --dataset-name ... --frame-range 0 300 ...
```

### 4) BD-rate 对比

```bash
python scripts/bd_rate.py --ref anchor.txt --test mine.txt   # 每行 "bpp psnr"
```

### 5) 自检（合成数据，秒级）

```bash
python scripts/smoke_test.py
```

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--base-channels / --base-depths` | 128 / 3 3 3 1 | 低动态 base 模型（**≈1.19M ≈ HiNeRV-S(3.19M) 的 1/3**，grid T=150 固定） |
| `--channels / --depths` | 120 / 2 2 1 1 | 高动态 GOP 小模型（**≈0.41M**，UAR-NVC 式小模型，grid T 跟随 GOP 长度） |
| `--p-epochs / --p-quant-epochs` | -1 / 30 | GOP 小模型主训练与 QAT 轮数（-1 = 补足到 --epochs） |
| `--patch-drop-ratio` | 0.25 | dynamicAL 选择性丢弃的 patch 比例（MSNeRV 随机丢 10~15% 不掉点 → 动态版丢 25%） |
| `--num-clips / --clip-len / --min-clip-len / --max-clip-len` | -1 / 16 / 12 / 20 | 自适应切分控制（GOP 长度约束在 12~20，超长段在内部散度峰再切） |
| `--w-fft / --w-scale / --w-flow / --w-tcl` | 0.1 / 0.2 / 0.01 / 0.01 | Scene Matters 四件套损失权重 |
| `--aux-ramp-start / --aux-ramp-len` | 0.3 / 0.3 | 辅助损失 ramp-in（早期主损失独占收敛） |
| `--hf-start / --hf-beta` | 0.5 / 1.0 | 动态高频注入（>1 关闭） |
| `--quant-mode` | noise | noise：HiNeRV Quant-Noise QAT / qec：UAR-NVC QEC（每层标量高斯熵）/ hyper：DCVC-RT 式迷你超先验（**逐元素 μ/σ**，与 INR 一起过拟合训练，无需外部数据） |
| `--w-rate / --base-w-rate` | 1.0 / -1 | 率失真权重 λ：GOP 小模型用 --w-rate（可稍大），base 宽模型用 --base-w-rate（可稍小；-1=继承 --w-rate） |
| `--hyper-embed-dim / --hyper-hidden / --hyper-down` | 16 / 64 / 8 | 超先验网络的层嵌入维度 / MLP 宽度 / 下采样倍率 |
| `--seg-mode` | adaptive | adaptive（内容散度+训练动态）/ fixed（等长，消融对照） |
| `--quant-residual-deadzone` | 1 | GOP 残差死区（\|q\|<dz 置零跳过编码；1=关闭） |
| `--quant-scale-bits` | 0 | DCVC-RT 式对数 scale 档位表（8=256 档；0=关闭） |
| `--dyn-*` | 见 `dynamics.py` | 帧权重采样（温度/更新频率/上限） |

## 实验脚本

```bash
# 消融（变体：A0 基线 / A1 完整-noise / A2 完整-qec / A3 完整-hyper /
#        A4 单模型+全损失 / A5 固定GOP / A6 无SceneMatters / A7 无动态采样）
GPU=4 QUICK=1 SEQ=ReadySetGo bash scripts/run_ablation.sh A0 A1
# QUICK=1 快速档（120 epochs）；QUICK=0 完整档（300 epochs）

# λ 扫描（GOP λ × base λ 多组对照，hyper 或 qec 模式）
GPU=4 QUICK=1 MODE=hyper bash scripts/run_lambda_sweep.sh

# RAFT 流伪标签（一次性）
python scripts/precompute_flow.py --dataset <root> --dataset-name <name> --input-size 1080 1920
```

## DCVC-RT 式超先验（hyper 模式）说明

不能照搬 DCVC-RT 的"数百万视频预训练熵模型"，但 INR 压缩逐视频过拟合的特性让我们可以
把一个极小的超先验网络（`compression/hyperprior.py`，约 5K 参数侧信息）与 INR 一起训练：
`g_a` 把残差权重池化下采样为 z（round 为 z_hat），`g_s`（共享 MLP + 层嵌入 + 位置编码，
在 z 分辨率运行后双线性上采样）逐元素预测 μ 和 σ；熵编码对 `s = q − round(μ)` 用
逐元素 N(0, σ) 的 CDF 行做算术编码（σ 小处 0 附近极便宜——正是 DCVC-RT 高压缩率的来源）。
编码/解码两端用 half 固化后的同一组超网络参数在 CPU 上重建 μ/σ，保证逐位一致。
RD 训练损失中加入 `w_rate · bpp_est`（与 QEC 相同接口）。

## 设计细节备忘

- **两级共享方式**：base 进码流一次；每个 GOP 只传 `θ_gop − derive(θ_base_dec)` 的量化残差。
  解码端：解码 base → derive → 加回 GOP 残差。GOP 之间完全独立（并行训练/解码皆可），
  不存在 UAR-NVC D-init（P 直接继承上一 GOP）的误差链。
- **推导映射**：通道取前 C_s（子网络切片），深度取每 stage 前 D_s 个 block，
  时间维（base 全视频 → GOP 长度）线性插值；aux 训练头与形状不匹配的张量保持自身初始化
  （并在残差编码时自动回退 intra）。
- **patch 丢弃不进损失**：仅修改采样分布（权重置 0 + epoch 采样数缩减），损失对所有保留帧等权。
- **QEC**：`QuantEntropy` 每层一个可学习 q_step，训练损失加入 `w_rate · bpp_est`
  （高斯熵估计），码流用学习到的高斯模型 CDF 做算术编码（torchac 实现，
  meta 只存 q_step/scale/R 三个标量，无需经验符号表）。
- **多卡**：所有 rank 顺序训练同一模型（base 或某 GOP）；加权采样器各 rank 独立同分布。
