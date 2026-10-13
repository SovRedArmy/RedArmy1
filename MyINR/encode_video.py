# ==========================================================================================
# encode_video.py (v2) —— 完整编解码流水线(单视频 × 单码率点)
#
# v2 关键设计(相对 v1, 见 REPORT.md / git log):
#   P0  --clip-mode gop: 固定短 GOP + GOM 结构(每 m 个 GOP 一个 I-model 重新锚定);
#         场景切换点只用于对齐 GOM 边界(在切换处开新 GOP), 不再作为 clip 边界本身 ——
#         修复 v1 "场景切分最大化段间不连续 / 权重差编码需要段间连续"的目标冲突
#   P2  P-model II 插值初始化: ε 由相邻 GOP 帧间 MSE 自适应
#   P5  步数对齐: 每 clip 的 epoch 数按 (基线前向量 / clip 前向量) 缩放(可 --max-epochs 封顶)
#   P4  TCL 静态场景门控(远帧 GT 嵌入相似度过高则关闭)
#   v2 无显式光流模块(移除伪标签/光流头/置信度头)
#
# 用法:
#   python encode_video.py --data data/bunny --preset S --mode ours --clip-mode gop
#   python encode_video.py --data data/beauty --mode baseline --preset S   # v1 协议不变
# ==========================================================================================
import argparse
import sys
import json
import os
import time

import numpy as np
import torch

from compress import compress_state_dict
from dynamics import content_cos_dist, frame_weights, probe_dynamics, scene_split, _strides_for
from modules import NeRVGenerator
from trainer import Cfg, ClipTrainer, build_pe, pe_encode
from video_data import VideoData

PRESETS = {
    # V3 统一架构: 底座 1M/2M/4M + 分支固定 0.4M + GOP 12-18 帧 + 充分训练
    'V3B':  ('256_1', 16, 48),    # 底座 ~0.95M (S 码率点)
    'V3M':  ('384_1', 22, 64),    # 底座 ~2.0M (M 码率点)
    'V3L':  ('512_1', 32, 80),    # 底座 ~4.0M (L 码率点)
    # 对照: 原始 NeRV-S/M/L (v1 协议)
    'S': ('512_1', 26, 96),
    'M': ('512_1', 58, 96),
    'L': ('512_1', 112, 96),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='帧目录')
    p.add_argument('--outf', default='runs')
    p.add_argument('--tag', default=None, help='实验名(默认 mode+preset)')
    p.add_argument('--preset', default='V3B', choices=list(PRESETS.keys()))
    p.add_argument('--mode', default='ours', choices=['baseline', 'ours', 'v3'])
    p.add_argument('--disable', default='', help='逗号分隔: crspe,fft,tcl,fw,ms,hfb,clips')
    p.add_argument('--epochs-i', type=int, default=500, help='底座训练 epoch(充分训练)')
    p.add_argument('--epochs-p', type=int, default=300, help='分支基准 epoch(充分训练)')
    p.add_argument('--max-epochs', type=int, default=800, help='步数对齐的每-clip epoch 上限')
    p.add_argument('--steps-parity', type=int, default=1, help='P5: 1=每clip epoch按前向量比例缩放')
    p.add_argument('--prune-keep', type=float, default=0.4)
    p.add_argument('--delta-keep', type=float, default=0.25)
    p.add_argument('--delta-l1', type=float, default=0.3, help='仅 rate-weight=0 时生效(v1 兼容)')
    p.add_argument('--rate-weight', type=float, default=0, help='V3 分支 R+λD 的 λ(0=纯失真, 方案 2)')
    p.add_argument('--quant-bit', type=int, default=8)
    p.add_argument('--clip-mode', default='gop', choices=['gop', 'auto', 'single', 'fixed'],
                   help='gop=固定短GOP+GOM(v2 默认); auto=场景切分(v1); single/fixed=对照')
    p.add_argument('--gop-len', type=int, default=0, help='GOP 帧数(0=自动: 12-18 目标)')
    p.add_argument('--gom-size', type=int, default=4, help='每 m 个 GOP 一个 I-model')
    p.add_argument('--independent-clips', action='store_true',
                   help='P-model 从零训练并 dense 编码(消融: 关闭 I/P 权重差复用)')
    p.add_argument('--no-ii', action='store_true', help='关闭 P2 II 初始化(退回 D-init)')
    p.add_argument('--ii-scale', type=float, default=8.0,
                   help='ε = clip(ii_scale·sqrt(MSE), 0.15, 0.85); MSE 为相邻 GOP 帧间均方误差')
    p.add_argument('--ms-lw', type=float, default=0.3, help='P3 多尺度中间尺度权重')
    p.add_argument('--tcl-gate', type=int, default=1, help='P4: 静态场景门控(远帧GT嵌入相似度>0.75 关 TCL)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--fft-weight', type=float, default=0.1)
    p.add_argument('--tcl-weight', type=float, default=0.005)
    p.add_argument('--fw-mode', default='resource', choices=['resource', 'loss'],
                   help='resource=资源分配(采样∝残差, 损失不变); loss=v1 损失调制(消融)')
    p.add_argument('--crspe-channels', type=int, default=4)
    p.add_argument('--fps-reps', type=int, default=8)
    # ---- V3: 宽底座 + 自适应 GOP 轻量残差分支(星型, 分支互不依赖) ----
    p.add_argument('--branch-ratio', type=int, default=2, help='底座/分支宽度比(V3B: 2→0.4M(NeRV-S fc=26 只有 1/2 整除)')
    p.add_argument('--branch-init', default='proj', choices=['proj', 'chain'],
                   help='分支初始化: proj=底座模板投影(head清零, 星型) | chain=前一分支解压缩权重(链式, 顺序解码)')
    p.add_argument('--branch-delta', default='sparse', choices=['sparse', 'lora', 'qat'],
                   help='分支差分编码: sparse=阈值化+恢复+稀疏熵编码 | lora=低秩因子(冻结body) | qat=率感知稠密差分(V4.1)')
    p.add_argument('--lora-rank', type=int, default=8, help='LoRA 分支差分的秩')
    p.add_argument('--gap-db', type=float, default=1.0, help='帧 PSNR 低于(均值-该值)视为难帧')
    p.add_argument('--min-gop', type=int, default=12, help='自适应 GOP 最小帧数')
    p.add_argument('--max-branches', type=int, default=10, help='最多分支数(按平均亏欠排序)')
    p.add_argument('--all-gops', action='store_true', help='E3: 分支铺满全部自适应 GOP(不只难段)')
    p.add_argument('--gap-bridge', type=int, default=4, help='难帧游程间允许桥接的空隙')
    p.add_argument('--branch-gpus', default='', help='逗号分隔 GPU 列表: 分支并行训练(如 "0,2,3"); 空=当前GPU串行')
    p.add_argument('--base-rate-weight', type=float, default=0.001,
                   help='V3 底座 R+λD 训练的率项系数(0=纯失真); 底座可更粗糙换码率, 精细度交给分支')
    p.add_argument('--base-prune-keep', type=float, default=0.4, help='V3 底座剪枝保留率(可低于 0.4 换码率)')
    p.add_argument('--rate-warm-frac', type=float, default=0.5, help='率项 warmup 比例(方案 3: 前 X% epoch 纯失真)')
    p.add_argument('--lambda-candidates', default='0.0005,0.001,0.002,0.005',
                   help='逗号分隔的底座 λ 候选列表(自适应选择)')
    p.add_argument('--lambda-probe-epochs', type=int, default=20,
                   help='λ 探针训练 epoch 数(短训评估 RD 后选最优)')
    p.add_argument('--lambda-min-hard-psnr', type=float, default=18.0,
                   help='底座难帧段最低可接受 PSNR(dB); 低于此值的 λ 被淘汰')
    p.add_argument('--delta-coder', default='min', choices=['hist', 'hyper', 'min'],
                   help='分支差分熵编码: hist=直方图算术; hyper=上下文高斯(DCVC-RT 式 μ/σ); min=逐张量择优')
    p.add_argument('--branch-job', action='store_true', help='[内部] 单分支子任务模式(由父进程派发)')
    p.add_argument('--gop-lo', type=int, default=0)
    p.add_argument('--gop-hi', type=int, default=0)
    p.add_argument('--base-ckpt', default='')
    p.add_argument('--out-dir', default='')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def auto_gop_len(n):
    return 75 if n >= 400 else 44


def build_gop_plan(scene_bounds, n, gop_len, gom_size):
    """P0: 固定长度 GOP 切分; 若场景切换点落在 GOP 内部, 则在切换点切开并重置 GOM 计数
    (场景切换处开新 I); 否则每 gom_size 个 GOP 一个 I-model。
    返回 (bounds, is_I_list)。"""
    cuts = sorted(set(b for b in scene_bounds[1:-1]))
    bounds, i_flags = [0], []
    since_I = gom_size          # 让第 0 个 GOP 成为 I
    while bounds[-1] < n:
        start = bounds[-1]
        nominal = start + gop_len
        cut = next((c for c in cuts if start + 8 <= c <= min(nominal, n)), None)
        end = cut if cut is not None else min(nominal, n)
        end = min(end, n)
        if end <= start:
            break
        is_I = (since_I >= gom_size) or (cut is not None)
        i_flags.append(is_I)
        since_I = 0 if is_I else since_I + 1
        bounds.append(end)
    if len(bounds) == 2:        # 单 GOP
        i_flags = [True]
    return bounds, i_flags


def adjacent_clip_mse(video, lo, hi):
    """相邻 GOP 之间的内容差: 前一 GOP 末帧 vs 当前 GOP 首帧 的 MSE(像素域)。"""
    a = video.frame_tensor([lo - 1])
    b = video.frame_tensor([lo])
    return float(torch.mean((a - b) ** 2).item())


def train_one_branch(args, video, cfg, lo, hi, dec_base_sd, base_model, device='cuda',
                     prev_dec_sd=None):
    """训练一个自适应 GOP 的残差分支并做差分压缩。
    返回 (bits, dec_sd, info_dict)。底座(base_model)须已加载反量化权重并冻结。
    --branch-init proj: 分支从底座模板投影初始化(head 清零, 星型);
                 chain: 分支从前一分支的解压缩权重初始化(UAR 式, 同构免投影, 不清 head)。
    --branch-delta sparse: 差分阈值化+恢复+稀疏熵编码(现状);
                   lora:  冻结初始权重, 只训每层 ΔW=U·V 低秩因子, 因子密集量化+熵编码;
                   qat:   V4.1 UAR 式率感知稠密差分 —— R+λD 训练 + STE 可学习 q_step,
                          差分分布在训练中被推向"好编码"形状(大多取整到 0/±1)。"""
    import copy as _copy
    from modules import project_state_to
    N = video.n_frames
    stem_s, fc_b, lw_b = PRESETS[args.preset]
    stem_s = '256_1'   # 分支固定 ~0.4M (用户设计: 高动态用小模型, 不随底座缩放)
    fc_s = 8
    lw_s = 24
    use_lora = getattr(args, 'branch_delta', 'sparse') == 'lora'
    use_qat = getattr(args, 'branch_delta', 'sparse') == 'qat'
    use_chain = getattr(args, 'branch_init', 'proj') == 'chain' and prev_dec_sd is not None
    qat_lam = args.rate_weight if getattr(args, 'rate_weight', 0) > 0 else 0.001
    br_cfg = _copy.copy(cfg)
    br_cfg.mode = 'ours'
    br_cfg.use_crspe = False; br_cfg.crspe_channels = 0
    br_cfg.use_ms = False; br_cfg.use_hfb = False; br_cfg.use_fft = False; br_cfg.use_tcl = False
    br_cfg.use_fw = True                      # 资源采样
    br_cfg.pair_batch = True
    br_cfg.prune = False                      # 分支不做新剪枝, 稀疏化交给差分阈值-恢复
    br_cfg.delta_keep = args.delta_keep       # 差分阈值保留比例(此前漏传, 导致 --delta-keep 无效)
    br_cfg.rate_weight = 0 if use_qat else args.rate_weight   # qat: λ 在初始权重加载后设回(注册顺序)
    br_cfg.use_rate_dense = (not use_qat) and args.rate_weight > 0
    if use_qat:
        br_cfg.delta_thr_frac = 10.0          # V4.1: 稠密差分, 关闭训练内阈值化
    br_cfg.residual_mode = True              # 分支输出原始残差(无 sigmoid, 可为负)
    br_cfg.stem_dim_num = stem_s
    br_cfg.fc_hw_dim = f'9_16_{fc_s}'
    br_cfg.lower_width = lw_s
    tr = ClipTrainer(video, br_cfg, (lo, hi), device=device)
    pb = sum(p.numel() for p in tr.model.parameters()) / 1e6
    if use_chain:
        # 链式: 同构分支直接继承前一分支解压缩权重(相邻 GOP 同场景延续 → 差分天然小), 不清 head
        init_sd = {k: v.clone() for k, v in prev_dec_sd.items()}
        init_src = 'chain(prev_branch)'
    else:
        # 投影初始化(模板版): 底座权重逐层自适应平均到分支形状 —— 无论底座 256/384/512,
        # 分支都能获得"底座模式"的初始化(旧 ratio 版对 V3M/V3L 形状不匹配, 等于随机初始化)
        init_sd = project_state_to(dec_base_sd, {k: tuple(v.shape) for k, v in tr.model.state_dict().items()})
        # ResNet 零初始化残差: 把输出层(head)清零, 使分支初始输出=0 → 合成=底座(正确起点)。
        # 不做这一步, 投影初始化让分支输出≈底座的缩小版, 训练前几百 epoch 花在
        # "压制底座模式"上, 到达零输出后梯度太弱, 卡死在底座水平(诊断: 27.05→27.05)。
        for k in list(init_sd.keys()):
            if k.startswith('head_layers.'):
                init_sd[k] = torch.zeros_like(init_sd[k])
        init_src = 'proj(base)' if prev_dec_sd is None else 'proj(base)(chain首段)'
    n_g = hi - lo
    ep = int(min(args.max_epochs, max(60, args.epochs_i * N / (2 * n_g)))) if args.steps_parity else args.epochs_p
    lbase_, levels_ = build_pe(cfg)
    def res_fn(global_ids):
        idx = torch.tensor([g / N for g in global_ids], dtype=torch.float32, device=device)
        x = pe_encode(idx, lbase_, levels_)
        fid = torch.tensor([int(g) for g in global_ids], device=device)
        return base_model(x, fid)[-1]
    # 段内底座均值(压缩前权重口径, 用于记录)
    with torch.no_grad():
        ps_before = []
        for i in range(lo, hi, 4):
            ids = list(range(i, min(i + 4, hi)))
            out = res_fn(ids).clamp(0, 1)
            gt = video.frame_tensor(ids, device)
            for j in range(len(ids)):
                ps_before.append((-10 * torch.log10(torch.nn.functional.mse_loss(out[j], gt[j]))).item())
    psnr_before = float(np.mean(ps_before))
    print(f'--- [v3] 分支 [{lo}:{hi}] ({pb:.2f}M, {ep} ep, {init_src}'
          f'{"+LoRA r%d" % args.lora_rank if use_lora else ""}, 底座段内均值 {psnr_before:.2f}dB) ---', flush=True)
    out_root = os.path.join(args.outf, f'{video.name}_{args.tag or f"v3_{args.preset}"}')
    br_train_pt = os.path.join(out_root, f'branch_train_{lo}.pt')
    if use_lora or use_qat:
        # LoRA/QAT 都要求: 先加载初始权重(标准键), 再注册参数化 —— 顺序不能反
        def _load_branch_init(model, init_sd):
            own = model.state_dict()
            init = {k: v for k, v in init_sd.items() if k in own and own[k].shape == v.shape}
            fe = 'frame_embed'
            if fe in init_sd and fe in own and init_sd[fe].shape != own[fe].shape:
                init[fe] = init_sd[fe][-1:].clone().repeat(own[fe].shape[0], 1, 1, 1)
            model.load_state_dict(init, strict=False)
        _load_branch_init(tr.model, init_sd)
    if use_lora:
        from modules import register_lora, lora_params, export_lora, load_lora, remove_lora
        n_lora = register_lora(tr.model, args.lora_rank)
        # ClipTrainer 构造优化器时还没有 U/V → 补挂参数组(train 的 LR 调度会覆盖所有组)
        tr.opt.add_param_group({'params': lora_params(tr.model), 'lr': 2.5e-4})
        lora_p = sum(p.numel() for p in lora_params(tr.model)) / 1e6
        print(f'    LoRA: {n_lora} 层, 因子 {lora_p:.3f}M ({lora_p/pb*100:.1f}% of {pb:.2f}M body)', flush=True)
        tr.train(epochs=ep, lr=2.5e-4, res_fn=res_fn, ckpt_path=br_train_pt, ckpt_every=100)
        # 导出因子并密集压缩(复用对称量化+熵编码路径); 解码端因子 + 复算投影 = 精确重建
        factors = export_lora(tr.model)
        for k in list(factors.keys()):        # NaN/Inf 清洗(极端情况)
            if torch.isnan(factors[k]).any() or torch.isinf(factors[k]).any():
                factors[k] = torch.where(torch.isnan(factors[k]) | torch.isinf(factors[k]),
                                         torch.zeros_like(factors[k]), factors[k])
        bits, dec_factors, _ = compress_state_dict(factors, bits=args.quant_bit, mode='dense')
        load_lora(tr.model, dec_factors)
        remove_lora(tr.model)                 # 就地保留 W+ΔW 合成权重(标准键名)
        dec_sd = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
    elif use_qat:
        # V4.1: 注册 STE 可学习 q_step, 率项(R+λD)推粗/失真推细 → 差分分布被训练成"好编码"形状
        from modules import register_qat, STEQuant
        n_q = register_qat(tr.model)
        tr._n_qat = n_q
        # 按权重自身尺度初始化 q_step(≈σ_w/8 → 起始熵 ~2-3bit):
        # 不能用 std(Δ) —— 链式初始化下 Δ≡0, 退化为 min-clamp 2.5e-7 → 压缩端
        # Δ/q_step 饱和截断, 解码差分全部削平(v4.1b 实测: 全段负增益-0.04~-0.56的根因)
        import math as _m
        with torch.no_grad():
            for name_, module_ in tr.model.named_modules():
                if not hasattr(module_, 'parametrizations') or 'weight' not in getattr(module_, 'parametrizations', {}):
                    continue
                ste_ = next((p_ for p_ in module_.parametrizations.weight if isinstance(p_, STEQuant)), None)
                if ste_ is None:
                    continue
                w_ = module_.parametrizations.weight.original
                qs0_ = float(w_.float().std().clamp(min=1e-6)) / 8.0
                ste_.q_step_raw.fill_(_m.log(_m.expm1(min(qs0_, 0.05))))
        # 构造优化器时还没有 q_step 参数 → 补挂(train 的 LR 调度会覆盖所有组)
        tr.opt.add_param_group({'params': [p_ for n_, p_ in tr.model.named_parameters()
                                           if 'q_step_raw' in n_], 'lr': 2.5e-4})
        tr.cfg.rate_weight = qat_lam          # 现在才激活率项(构造时为 0)
        tr.cfg.use_rate_dense = True
        print(f'    QAT: {n_q} 层可学习 q_step(幅值感知初始化), λ={qat_lam:g} (R+λD, 稠密差分)', flush=True)
        tr.train(epochs=ep, lr=2.5e-4, res_fn=res_fn, ref_state=init_sd,
                 ckpt_path=br_train_pt, ckpt_every=100)
        sd = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
        for k in sd:
            if torch.isnan(sd[k]).any() or torch.isinf(sd[k]).any():
                sd[k] = torch.where(torch.isnan(sd[k]) | torch.isinf(sd[k]),
                                    torch.zeros_like(sd[k]), sd[k])
        q_steps = tr.get_q_steps()
        # 差分用训练出的 q_step 量化(compress 的 q_steps 路径), 大多取整到 0/±1 → 熵编码极省
        bits, dec_sd, _ = compress_state_dict(sd, bits=args.quant_bit, mode='delta',
                                              ref_sd=init_sd, delta_keep=1.0, q_steps=q_steps)
    else:
        tr.train(init_state=init_sd, ref_state=init_sd, epochs=ep, lr=2.5e-4, res_fn=res_fn,
                 ckpt_path=br_train_pt, ckpt_every=100)
        sd = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
        # R+λD 训练可能产生 NaN/Inf(极端情况) → 压缩前清洗
        for k in sd:
            if torch.isnan(sd[k]).any() or torch.isinf(sd[k]).any():
                sd[k] = torch.where(torch.isnan(sd[k]) | torch.isinf(sd[k]),
                                    torch.zeros_like(sd[k]), sd[k])
        bits, dec_sd, _ = compress_state_dict(sd, bits=args.quant_bit, mode='delta',
                                              ref_sd=init_sd, delta_keep=1.0, delta_coder=args.delta_coder)
    tr.model.load_state_dict(dec_sd)
    psnr_after = tr.eval_stream(state_dict=dec_sd, res_fn=res_fn)[0]
    info = dict(lo=lo, hi=hi, n=n_g, params_m=pb, epochs=ep, bits=bits,
                init=init_src,
                delta=('lora_r%d' % args.lora_rank) if use_lora else ('qat_lam%g' % qat_lam if use_qat else 'sparse'),
                bpp_param=bits / (pb * 1e6),
                psnr_before=psnr_before, psnr_after=psnr_after)
    try:
        os.remove(br_train_pt)
    except OSError:
        pass
    print(f'--- [v3] 分支 [{lo}:{hi}] done: {bits/1e6:.2f}Mb ({info["bpp_param"]:.2f} b/p), '
          f'段内 {psnr_before:.2f}->{psnr_after:.2f}dB ---', flush=True)
    return bits, dec_sd, info




def select_adaptive_gops_v3(per_psnr, cos_dist, n_frames,
                            target_len=15, min_len=12, max_len=18, gap_db=0.8, bridge=4):
    """V3 重设计版自适应 GOP: TeNeRV 风格 + 12-18 帧目标。

    结合两个信号:
    1. 重建难度: 底座逐帧 PSNR 低于 (均值 - gap_db) 的帧 = "难帧"
    2. 内容变化: 相邻帧特征余弦距离的突变点 = 潜在切分点

    策略: 先用内容变化点确定候选边界, 再将难帧游程合并到最近的候选段,
    最后调整段长到 [min_len, max_len] 范围内(拆分过长段, 合并过短段)。
    """
    import numpy as _np
    p = _np.asarray(per_psnr)
    cd = _np.asarray(cos_dist) if cos_dist is not None else _np.zeros(n_frames)

    # 信号 1: 内容突变点(余弦距离的局部峰, 超过均值+1σ)
    if cd.max() > 0:
        cd_thresh = cd.mean() + cd.std()
        content_cuts = [i for i in range(1, n_frames) if cd[i] > cd_thresh and cd[i] >= cd[max(0,i-1)]]
    else:
        content_cuts = []

    # 信号 2: 难帧(重建 PSNR 低)
    hard = p < (p.mean() - gap_db) if len(p) == n_frames else _np.zeros(n_frames, bool)

    # 合并信号: 候选边界 = 内容突变点 + 难帧游程的首尾
    candidate_bounds = set([0, n_frames])
    candidate_bounds.update(content_cuts)
    # 难帧游程的首尾也作为候选
    i = 0
    while i < n_frames:
        if hard[i]:
            j = i
            gap_cnt = 0
            while j < n_frames - 1 and (hard[j+1] or gap_cnt < bridge):
                j += 1
                gap_cnt = 0 if hard[j] else gap_cnt + 1
            candidate_bounds.add(i)
            candidate_bounds.add(min(j + 1, n_frames))
            i = j + 1
        else:
            i += 1

    bounds = sorted(candidate_bounds)

    # 调整段长到 [min_len, max_len]
    adjusted = [bounds[0]]
    for b in bounds[1:]:
        seg_len = b - adjusted[-1]
        if seg_len < min_len and adjusted[-1] > 0:
            # 过短: 与前段合并
            continue
        elif seg_len > max_len:
            # 过长: 均匀拆分到 target_len 附近
            start = adjusted[-1]
            n_sub = max(1, round(seg_len / target_len))
            step = seg_len // n_sub
            for k in range(1, n_sub):
                adjusted.append(start + k * step)
        adjusted.append(b)
    # 确保最后一段也满足 min_len
    if len(adjusted) > 1 and adjusted[-1] - adjusted[-2] < min_len:
        adjusted.pop()

    segments = [(adjusted[i], adjusted[i+1]) for i in range(len(adjusted)-1)
                if adjusted[i+1] - adjusted[i] >= min_len // 2]  # 允许稍短的尾段
    return segments


def probe_lambdas(args, video, cfg, device='cuda'):
    """多出口 λ 选择: 一次训练、多个快照、零额外成本。
    底座用最大候选 λ 训练; 在 warmup 结束后的不同 ramp 阶段各取一个快照,
    压缩评估 RD 后选最优出口(等效于选了该时刻的有效 λ)。"""
    import copy as _copy
    from trainer import ClipTrainer, pe_encode, build_pe
    N = video.n_frames
    max_lam = max(float(x) for x in args.lambda_candidates.split(','))
    warm = cfg.rate_warm_frac                     # e.g. 0.5

    # 出口时刻: warmup 结束(λ=0) + ramp 的 1/3, 2/3, 末尾(λ=max)
    exit_fracs = [warm, warm + (1 - warm) * 0.33, warm + (1 - warm) * 0.67, 1.0]
    exit_names = ['λ=0 (warmup末)', 'λ=low (ramp 33%)', 'λ=mid (ramp 67%)', 'λ=max (满强度)']

    print(f'--- [v3-λ] 多出口选择: 训练用 max λ={max_lam:g}, '
          f'出口 at {[f"{f:.0%}" for f in exit_fracs]} ---', flush=True)

    # 训练(带快照收集)
    pc = _copy.copy(cfg)
    pc.mode = 'baseline'
    pc.use_crspe = False; pc.crspe_channels = 0
    pc.use_ms = False; pc.use_fw = False
    pc.use_fft = pc.use_tcl = pc.use_hfb = False
    pc.use_rate_dense = max_lam > 0
    pc.rate_weight = max_lam
    pc.prune = False                        # 探针不剪枝(纯率失真评估)

    tr = ClipTrainer(video, pc, (0, N), device=device)
    epochs = args.lambda_probe_epochs
    snapshots = []

    # 手动控制训练循环以在指定 epoch 取快照
    exit_eps = [max(1, int(f * epochs)) for f in exit_fracs]
    for ep in range(epochs):
        # 复用 trainer 的单 epoch 逻辑(简化: 直接调 train 但指定 epochs=ep+1)
        pass

    # 更简单的方式: 分段训练, 每段结束后取快照
    prev_ep = 0
    for exit_ep, name in zip(exit_eps, exit_names):
        seg = exit_ep - prev_ep
        if seg <= 0:
            continue
        tr.train(epochs=seg, verbose=False)
        prev_ep = exit_ep

        sd = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
        bits, _, _ = compress_state_dict(sd, bits=args.quant_bit, mode='dense')
        psnr = tr.quick_eval()

        # 逐帧 PSNR(P10 = 难帧)
        lbase_, levels_ = build_pe(cfg)
        per_psnr = []
        with torch.no_grad():
            for i in range(0, N, 4):
                ids = list(range(i, min(i + 4, N)))
                x = pe_encode(torch.tensor([g / N for g in ids], dtype=torch.float32, device=device), lbase_, levels_)
                fid = torch.tensor(ids, device=device)
                out = tr.model(x, fid)[-1].clamp(0, 1)
                gt = video.frame_tensor(ids, device)
                for j in range(len(ids)):
                    per_psnr.append((-10 * torch.log10(torch.nn.functional.mse_loss(out[j], gt[j]))).item())
        hard = float(np.percentile(per_psnr, 10))

        snapshots.append(dict(name=name, ep=exit_ep, psnr=psnr, bits=bits, hard_psnr=hard))
        print(f'  {name}: PSNR={psnr:.2f}dB, {bits/1e6:.2f}Mb, 难帧P10={hard:.2f}dB', flush=True)

    # 选择: 淘汰难帧 < 阈值的, 在剩余中选 PSNR×bit 效率最优的
    valid = [s for s in snapshots if s['hard_psnr'] >= args.lambda_min_hard_psnr]
    if not valid:
        best_snap = max(snapshots, key=lambda s: s['hard_psnr'])
        print(f'  ⚠ 全部出口难帧P10 < {args.lambda_min_hard_psnr}, 保底选 {best_snap["name"]}', flush=True)
    else:
        best_snap = max(valid, key=lambda s: s['psnr'] / max(s['bits'], 1))
        print(f'  ✓ 选定 {best_snap["name"]} (PSNR={best_snap["psnr"]:.2f}dB, {best_snap["bits"]/1e6:.2f}Mb)', flush=True)

    # 返回: 选中的等效 λ(根据出口位置插值) + 全部快照
    best_idx = snapshots.index(best_snap)
    best_lam = max_lam * (exit_fracs[best_idx] - warm) / max(1e-6, 1.0 - warm) if best_idx > 0 else 0.0
    return best_lam, snapshots


def run_v3(args, video, cfg):
    """V3 流水线(分支可并行):
    阶段1 底座训练 -> [底座压缩(CPU线程) ∥ 逐帧PSNR扫描(GPU)] -> 自适应GOP
    阶段2 各GOP残差分支: --branch-gpus 给定时多GPU子进程并行(星型, 分支互不依赖),
                         否则当前GPU串行; 分支=投影初始化+残差学习+步数对齐+RQEC-lite+阈值恢复
    阶段3 合成解码评估(底座 + 各分支)。"""
    import copy as _copy
    import concurrent.futures as _cf
    from modules import NeRVGenerator
    N, device = video.n_frames, args.device
    strides = cfg.strides
    t0 = time.time()
    out_root = os.path.join(args.outf, f'{video.name}_{args.tag or f"v3_{args.preset}"}')
    os.makedirs(out_root, exist_ok=True)
    base_done_pt = os.path.join(out_root, '_base_done.pt')    # 底座成品(训练完即存, 重启免重训)
    base_train_pt = os.path.join(out_root, '_base_train.pt')  # 底座训练中断点(周期落盘)

    # ---- 阶段 0: 自适应 λ 选择(短训探针) ----
    if args.lambda_candidates and args.lambda_probe_epochs > 0:
        best_lam, probe_results = probe_lambdas(args, video, cfg, device)
        args.base_rate_weight = best_lam
        # 把探针结果记入结果
        _probe_info = probe_results
    else:
        _probe_info = None

    # ---- 阶段 1: 宽底座(baseline 协议, 渐进剪枝, 用选定的 λ) ----
    base_cfg = _copy.copy(cfg)
    base_cfg.mode = 'baseline'
    base_cfg.use_crspe = False; base_cfg.crspe_channels = 0
    base_cfg.use_ms = False; base_cfg.use_fw = False
    base_cfg.use_fft = base_cfg.use_tcl = base_cfg.use_hfb = False
    base_cfg.use_rate_dense = args.base_rate_weight > 0
    base_cfg.rate_weight = args.base_rate_weight
    base_cfg.prune_keep = args.base_prune_keep
    tr_base = ClipTrainer(video, base_cfg, (0, N), device=device)
    pmb = sum(p.numel() for p in tr_base.model.parameters()) / 1e6
    if os.path.isfile(base_done_pt):
        # 断点续跑: 底座已训练完(上次进程在分支/评估阶段被杀) → 直接加载成品
        bd = torch.load(base_done_pt, map_location='cpu')
        base_sd = bd['base_sd']; q_steps_base = bd['q_steps']
        # 成品里是 QAT 收官后的标准键; 刚构造的模型还挂着 QAT 参数化键 → 先拆除再加载
        if getattr(tr_base, '_n_qat', 0) > 0:
            from modules import remove_qat
            remove_qat(tr_base.model)
        print(f'--- [v3] 底座成品已存在({pmb:.2f}M), 跳过训练直接进入压缩/GOP 扫描 ---', flush=True)
    else:
        print(f'--- [v3] 底座 (整段, baseline 协议, {pmb:.2f}M, {args.epochs_i} ep, '
              f'选定λ={args.base_rate_weight:g}, 剪枝保留={args.base_prune_keep:g}) ---', flush=True)
        tr_base.train(verbose=True, ckpt_path=base_train_pt, ckpt_every=20)
        q_steps_base = tr_base.get_q_steps() if hasattr(tr_base, 'q_steps') else {}
        base_sd = {k: v.detach().cpu().clone() for k, v in tr_base.model.state_dict().items()}
        torch.save(dict(base_sd=base_sd, q_steps=q_steps_base, preset=args.preset,
                        strides=strides, N=N, pmb=pmb), base_done_pt)
        try:
            os.remove(base_train_pt)
        except OSError:
            pass

    # 底座输出闭包(未压缩权重, 供 GOP 扫描并行使用)
    lbase_, levels_ = build_pe(cfg)
    def _res(model_ref):
        def fn(global_ids):
            idx = torch.tensor([g / N for g in global_ids], dtype=torch.float32, device=device)
            x = pe_encode(idx, lbase_, levels_)
            fid = torch.tensor([int(g) for g in global_ids], device=device)
            return model_ref(x, fid)[-1]
        return fn

    # 底座压缩 → 加载反量化权重 → 用压缩后模型做 GOP 扫描(与解码端完全一致)
    # TeNeRV: 相邻帧内容余弦距离(已缓存的 AlexNet 嵌入)
    base_bits, dec_base_sd, _ = compress_state_dict(base_sd, args.quant_bit, 'dense', q_steps=q_steps_base)
    tr_base.model.load_state_dict(dec_base_sd)
    base_model = tr_base.model
    base_model.eval()
    for p_ in base_model.parameters():
        p_.requires_grad_(False)
    fn_un = _res(base_model)
    per_psnr = []
    with torch.no_grad():
        for i in range(0, N, 4):
            ids = list(range(i, min(i + 4, N)))
            out = fn_un(ids).clamp(0, 1)
            gt = video.frame_tensor(ids, device)
            for j in range(len(ids)):
                per_psnr.append((-10 * torch.log10(torch.nn.functional.mse_loss(out[j], gt[j]))).item())
    base_psnr = float(np.mean(per_psnr))

    # TeNeRV: 用缓存的内容特征计算余弦距离
    embeds_cache = os.path.join('./cache', f'{video.name}_alex.npy')
    if os.path.isfile(embeds_cache):
        e = np.load(embeds_cache)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        cos_d = np.concatenate([[0.0], 1.0 - np.sum(e[1:] * e[:-1], axis=1)])
    else:
        cos_d = np.zeros(N)
    all_segments = select_adaptive_gops_v3(per_psnr, cos_d, N,
                                   target_len=args.min_gop + 3,
                                   min_len=args.min_gop, max_len=args.min_gop + 6,
                                   gap_db=args.gap_db, bridge=args.gap_bridge)
    # 只给底座表现低于均值的段加分支(星型: 底座管全局, 分支只管难段)
    mean_psnr = float(np.mean(per_psnr))
    if args.max_branches <= 0:
        gops = []  # 对照实验: 不加分支
    elif getattr(args, 'all_gops', False):
        gops = list(all_segments)   # E3: 铺满全部段(差分便宜后覆盖换质量)
    else:
        gops = [(lo, hi) for lo, hi in all_segments
            if float(np.mean(per_psnr[lo:hi])) < mean_psnr]
    print(f'[v3] 底座 PSNR={base_psnr:.2f} bits={base_bits/1e6:.2f}Mb; 自适应 GOP: {gops}', flush=True)

    # ---- 阶段 2: 分支(串行 或 多GPU子进程并行) ----
    branch_infos, branch_models = [], []
    total_bits = base_bits
    bgpus = [g.strip() for g in args.branch_gpus.split(',') if g.strip() != '']
    if args.branch_init == 'chain':
        bgpus = []   # 链式初始化要求按 GOP 顺序训练/解码(分支 k 参照分支 k-1), 禁用并行子进程
        print('[v3] chain 模式: 分支按 GOP 顺序串行训练', flush=True)
    if gops and bgpus:
        base_ckpt = os.path.join(out_root, '_base.pt')
        torch.save(dict(sd=dec_base_sd, preset=args.preset, strides=strides, N=N), base_ckpt)
        import subprocess as _sp
        pending = list(enumerate(gops))
        running = []          # (proc, bi, lo, hi, gpu, logfile)
        free = list(bgpus)
        os.makedirs('logs', exist_ok=True)
        while pending or running:
            while pending and free:
                bi, (lo, hi) = pending.pop(0)
                g = free.pop(0)
                lf = open(f'logs/v3branch_{video.name}_{args.tag}_{lo}.log', 'w')
                cmd = (f'env CUDA_VISIBLE_DEVICES={g} {sys.executable} {os.path.abspath(__file__)} '
                       f'--data {args.data} --mode v3 --branch-job --preset {args.preset} '
                       f'--gop-lo {lo} --gop-hi {hi} --base-ckpt {base_ckpt} '
                       f'--branch-ratio {args.branch_ratio} --epochs-i {args.epochs_i} '
                       f'--epochs-p {args.epochs_p} --max-epochs {args.max_epochs} '
                       f'--steps-parity {args.steps_parity} --quant-bit {args.quant_bit} '
                       f'--branch-init {args.branch_init} --branch-delta {args.branch_delta} '
                       f'--lora-rank {args.lora_rank} --delta-keep {args.delta_keep} '
                       f'--out-dir {out_root} --seed {args.seed}')
                running.append((_sp.Popen(cmd, shell=True, stdout=lf, stderr=_sp.STDOUT), bi, lo, hi, g, lf))
                print(f'[v3] 派发分支 [{lo}:{hi}] -> GPU {g}', flush=True)
            time.sleep(20)
            for r in list(running):
                proc, bi, lo, hi, g, lf = r
                if proc.poll() is not None:
                    running.remove(r); lf.close(); free.append(g)
                    print(f'[v3] 分支子进程 [{lo}:{hi}] 退出 rc={proc.returncode}', flush=True)
        # 收集子进程产物
        for bi, (lo, hi) in enumerate(gops):
            st = json.load(open(os.path.join(out_root, f'branch_{lo}.json')))
            ck = torch.load(os.path.join(out_root, f'branch_{lo}.pt'), map_location='cpu')
            bm = NeRVGenerator(embed_length=2 * levels_, stem_dim_num='256_1',
                               fc_hw_dim='9_16_8', lower_width=24,
                               n_frames=hi - lo, crspe_channels=0, multi_scale=False,
                               stride_list=strides, act='swish').to(device)
            bm.load_state_dict(ck['dec_sd']); bm.eval()
            for p_ in bm.parameters():
                p_.requires_grad_(False)
            branch_models.append((lo, hi, bm))
            branch_infos.append(st)
            total_bits += st['bits']
    else:
        import copy as _c2
        prev_dec = None                       # 链式模式: 分支 k 参照分支 k-1 的解压缩权重
        for lo, hi in gops:
            br_json = os.path.join(out_root, f'branch_{lo}.json')
            br_pt = os.path.join(out_root, f'branch_{lo}.pt')
            if os.path.isfile(br_json) and os.path.isfile(br_pt):
                # 断点续跑: 该分支上次已完成
                info = json.load(open(br_json))
                dec_sd = torch.load(br_pt, map_location='cpu')['dec_sd']
                bits = float(info['bits'])
                print(f'[v3] 分支 [{lo}:{hi}] 已完成, 跳过 (bits={bits/1e6:.2f}Mb)', flush=True)
            else:
                bits, dec_sd, info = train_one_branch(args, video, cfg, lo, hi, dec_base_sd, base_model,
                                                      device, prev_dec_sd=prev_dec)
                torch.save(dict(dec_sd=dec_sd), br_pt)
                with open(br_json, 'w') as f:
                    json.dump(info, f, default=float)
            prev_dec = dec_sd
            # 用分支宽度(非 base 宽度)构建评估用模型
            br_eval_cfg = _c2.copy(cfg)
            br_eval_cfg.stem_dim_num = '256_1'   # 与 train_one_branch 固定分支一致
            br_eval_cfg.fc_hw_dim = '9_16_8'
            br_eval_cfg.lower_width = 24
            br_eval_cfg.crspe_channels = 0
            br_eval_cfg.use_rate_dense = False    # 评估模型不注册 QAT(保持标准键名)
            br_eval_cfg.rate_weight = 0
            br_eval_cfg.residual_mode = True      # 与分支训练一致
            br_eval_cfg.use_ms = False            # 与分支一致(single head, 非 multi-scale)
            bm = ClipTrainer(video, br_eval_cfg, (lo, hi), device=device).model
            bm.load_state_dict(dec_sd); bm.eval()
            for p_ in bm.parameters():
                p_.requires_grad_(False)
            branch_models.append((lo, hi, bm))
            branch_infos.append(info)
            total_bits += bits

    # ---- 阶段 3: 整段合成评估 ----
    psnrs, msss, tl = [], [], []
    with torch.no_grad():
        for i in range(0, N, 4):
            ids = list(range(i, min(i + 4, N)))
            x = pe_encode(torch.tensor([g / N for g in ids], dtype=torch.float32, device=device), lbase_, levels_)
            fid = torch.tensor(ids, device=device)
            torch.cuda.synchronize(); t0f = time.time()
            out = base_model(x, fid)[-1]
            for j, g in enumerate(ids):
                bh = next(((l, h, m) for l, h, m in branch_models if l <= g < h), None)
                if bh is not None:
                    l, h, br = bh
                    xl = pe_encode(torch.tensor([(g - l) / (h - l)], dtype=torch.float32, device=device), lbase_, levels_)
                    out[j] = out[j] + br(xl, torch.tensor([g - l], device=device))[-1][0]
            torch.cuda.synchronize(); tl.append(len(ids) / (time.time() - t0f))
            out = out.clamp(0, 1)
            gt = video.frame_tensor(ids, device)
            for j in range(len(ids)):
                psnrs.append((-10 * torch.log10(torch.nn.functional.mse_loss(out[j], gt[j]))).item())
            from video_data import evaluate_msssim
            msss.append(evaluate_msssim(out.cpu(), gt.cpu(), batch=len(ids)))
    psnr = float(np.mean(psnrs)); msssim = float(np.mean(msss)); fps = float(np.mean(tl[:10])) if tl else None
    bpp = total_bits / (N * video.H * video.W)
    tag = args.tag or f'v3_{args.preset}'
    res = dict(video=video.name, tag=tag, mode='v3', preset=args.preset,
               branch_ratio=args.branch_ratio, gap_db=args.gap_db,
               branch_init=args.branch_init, branch_delta=args.branch_delta,
               lora_rank=args.lora_rank,
               base_rate_weight=args.base_rate_weight, base_prune_keep=args.base_prune_keep,
               lambda_probe=_probe_info,
               branch_gpus=bgpus, n_frames=N, base_bits=base_bits, base_psnr=base_psnr,
               base_params_m=pmb, gops=[[l, h] for l, h in gops],
               branches=branch_infos, total_bits=total_bits, bpp=bpp,
               psnr=psnr, msssim=msssim, fps=fps, wall_s=time.time() - t0)
    with open(os.path.join(out_root, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2, default=float)
    # 成功后清理训练断点; 保留 _base_done.pt(仅数十MB) — 分支超参实验可直接复用底座
    for f_ in [base_train_pt] + \
              [os.path.join(out_root, f'branch_train_{l}.pt') for l, _ in gops] + \
              ([base_ckpt] if gops and bgpus else []):
        try:
            os.remove(f_)
        except OSError:
            pass
    print(json.dumps({k: v for k, v in res.items() if k != 'branches'}, indent=2, default=str), flush=True)


def run_v3_branch_job(args, video):
    """[内部] --branch-job 子任务: 训练单个 GOP 的残差分支, 写出 branch_{lo}.pt/.json。"""
    from modules import NeRVGenerator
    device = args.device
    stem_p, fc_p, lw_p = PRESETS[args.preset]
    cfg = Cfg(mode='v3', strides=_strides_for(video), epochs=args.epochs_i, seed=args.seed,
              rate_warm_frac=args.rate_warm_frac,
              stem_dim_num=f'{stem_p}_1', fc_hw_dim=f'9_16_{fc_p}', lower_width=lw_p)
    ck = torch.load(args.base_ckpt, map_location='cpu')
    dec_base_sd, strides, N = ck['sd'], ck['strides'], ck['N']
    stem_s, fc_b, lw_b = PRESETS[ck['preset']]
    base_model = NeRVGenerator(embed_length=2 * 40, stem_dim_num=stem_s, fc_hw_dim=f'9_16_{fc_b}',
                               lower_width=lw_b, n_frames=N, crspe_channels=0, multi_scale=False,
                               stride_list=strides, act='swish').to(device)
    base_model.load_state_dict(dec_base_sd)
    base_model.eval()
    for p_ in base_model.parameters():
        p_.requires_grad_(False)
    args.preset = ck['preset']
    lo, hi = args.gop_lo, args.gop_hi
    bits, dec_sd, info = train_one_branch(args, video, cfg, lo, hi, dec_base_sd, base_model, device)
    torch.save(dict(dec_sd=dec_sd), os.path.join(args.out_dir, f'branch_{lo}.pt'))
    with open(os.path.join(args.out_dir, f'branch_{lo}.json'), 'w') as f:
        json.dump(info, f, default=float)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    video = VideoData(args.data, cache_dir='./cache')
    if args.mode == 'v3' and args.branch_job:
        return run_v3_branch_job(args, video)
    if args.mode == 'v3':
        stem_s, fc_b, lw_b = PRESETS[args.preset]
        cfg0 = Cfg(mode='v3', strides=_strides_for(video), epochs=args.epochs_i, seed=args.seed,
                   stem_dim_num=stem_s, fc_hw_dim=f'9_16_{fc_b}', lower_width=lw_b)
        return run_v3(args, video, cfg0)
    disable = set(x for x in args.disable.split(',') if x)
    cfg = Cfg(mode=args.mode, epochs=args.epochs_i, prune_keep=args.prune_keep,
              quant_bit=args.quant_bit, seed=args.seed,
              fft_weight=args.fft_weight, tcl_weight=args.tcl_weight,
              rate_weight=args.rate_weight, delta_l1=args.delta_l1,
              ms_lw=args.ms_lw, fw_mode=args.fw_mode, strides=_strides_for(video))
    for m in ['crspe', 'fft', 'tcl', 'fw', 'ms', 'hfb']:
        if m in disable:
            setattr(cfg, f'use_{m}', False)
    if 'crspe' in disable:
        cfg.crspe_channels = 0
    else:
        cfg.crspe_channels = args.crspe_channels
    use_clips = (args.mode == 'ours') and ('clips' not in disable)

    stem_s, fc, lower = PRESETS[args.preset]
    fc_hw = f'9_16_{fc}'

    # ---------------- 1. 预计算(缓存, 无光流) ----------------
    cache = './cache'
    os.makedirs(cache, exist_ok=True)
    embeds_path = os.path.join(cache, f'{video.name}_alex.npy')
    if not os.path.isfile(embeds_path):
        from embedder import video_content_embeds
        np.save(embeds_path, video_content_embeds(video, args.device))
    embeds = np.load(embeds_path)
    cos_d = content_cos_dist(video, embeds)

    grads_path = os.path.join(cache, f'{video.name}_probegrads.npy')
    if not os.path.isfile(grads_path):
        g = probe_dynamics(video, device=args.device)
        np.save(grads_path, g)
    grads = np.load(grads_path)

    # P4: TCL 静态场景门控 —— 远帧(间隔>N/2) GT 嵌入平均余弦相似度过高 → 关闭 TCL
    e = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    n_e = len(e)
    far_sim = float(np.mean([np.dot(e[i], e[i + n_e // 2]) for i in range(n_e // 2)]))
    tcl_enabled = not (args.tcl_gate and far_sim > 0.75)
    cfg.tcl_enabled = tcl_enabled
    print(f'TCL gate: far-frame GT embed sim={far_sim:.3f} -> tcl {"ON" if tcl_enabled else "OFF"}', flush=True)

    # ---------------- 2. 帧权重 + 切分(P0) ----------------
    N = video.n_frames
    if cfg.use_fw and args.mode == 'ours':
        if cfg.fw_mode == 'resource':
            fw_all = frame_weights(grads, w_min=0.6, w_max=1.8)   # 资源分配: 宽范围(决定采样比例)
        else:
            fw_all = frame_weights(grads)                          # 损失调制(v1 语义): 温和范围
    else:
        fw_all = None

    if (not use_clips) or args.clip_mode == 'single':
        bounds, i_flags = [0, N], [True]
    elif args.clip_mode == 'fixed':
        gop_len = args.gop_len or auto_gop_len(N)
        K = max(1, round(N / gop_len))
        bounds = sorted(set(round(i * N / K) for i in range(K + 1)))
        i_flags = [k % args.gom_size == 0 for k in range(len(bounds) - 1)]
    elif args.clip_mode == 'gop':
        gop_len = args.gop_len or auto_gop_len(N)
        scene = scene_split(cos_d, grads, N, target_len=max(32, gop_len // 2))
        bounds, i_flags = build_gop_plan(scene, N, gop_len, args.gom_size)
    else:   # 'auto' = v1 场景切分(对照保留)
        target = args.gop_len or (100 if N >= 400 else 48)
        bounds = scene_split(cos_d, grads, N, target_len=target)
        i_flags = [True] + [False] * (len(bounds) - 2)
    K = len(bounds) - 1
    print(f'clips={K}, bounds={bounds}, I-flags={i_flags}', flush=True)

    cfg.stem_dim_num = stem_s
    cfg.fc_hw_dim = fc_hw
    cfg.lower_width = lower
    cfg.delta_keep = args.delta_keep

    # ---------------- 3. 逐 clip 训练 + 压缩 ----------------
    t0 = time.time()
    total_bits = 0
    clip_infos = []
    prev_dec_sd = None
    i_mask = None
    for k in range(K):
        lo, hi = bounds[k], bounds[k + 1]
        n_clip = hi - lo
        is_I = i_flags[k]
        indep = args.independent_clips and not is_I
        # P5 步数对齐: clip 每 epoch ≈ 2·n_clip 前向 → epochs ≈ epochs_base·N/(2·n_clip)
        if use_clips and args.steps_parity and K > 1:
            e_base = args.epochs_i if is_I else args.epochs_p
            ep = int(min(args.max_epochs, max(60, e_base * N / (2 * n_clip))))
        else:
            ep = args.epochs_i if is_I else args.epochs_p
        if is_I or indep:
            cfg_k = cfg
        else:
            import copy
            cfg_k = copy.copy(cfg)
            cfg_k.prune = False          # P-model 继承 I 剪枝掩码, 不做新剪枝
        tr = ClipTrainer(video, cfg_k, (lo, hi), device=args.device)
        params_m = sum(p.numel() for p in tr.model.parameters()) / 1e6
        print(f'--- clip {k} [{lo}:{hi}] {"I" if is_I else "P"}-model '
              f'({params_m:.2f}M params, {ep} epochs) ---', flush=True)
        fwc = None if fw_all is None else fw_all[lo:hi]
        # P2: II 插值系数 ε(相邻 GOP 帧间 MSE 自适应)
        ii_eps = None
        if (not is_I) and (not indep) and (not args.no_ii):
            mse = adjacent_clip_mse(video, lo, hi)
            ii_eps = float(np.clip(args.ii_scale * np.sqrt(max(mse, 1e-6)), 0.15, 0.85))
            cfg_k.ii_eps = ii_eps
            print(f'    II-init: adj-MSE={mse:.4f} -> eps={ii_eps:.2f}', flush=True)
        tr.train(frame_weights=fwc,
                 init_state=None if (is_I or indep) else prev_dec_sd,
                 ref_state=None if (is_I or indep) else prev_dec_sd,
                 epochs=ep, lr=cfg.lr if is_I else 2.5e-4,
                 inherit_mask=None if (is_I or indep) else i_mask)
        if is_I and cfg.prune and tr.prune_mask is not None:
            i_mask = {kk: (m.cpu() if m is not None else None) for kk, m in tr.prune_mask.items()}
        sd = {kk: v.detach().cpu().clone() for kk, v in tr.model.state_dict().items()}
        mode = 'dense' if (is_I or indep) else 'delta'
        # P5: 差分已在训练内阈值化并恢复 → 压缩时不再重复阈值(delta_keep=1.0)
        bits, dec_sd, detail = compress_state_dict(
            sd, bits=args.quant_bit, mode=mode, ref_sd=prev_dec_sd, delta_keep=1.0,
            delta_coder=(args.delta_coder if mode == 'delta' else 'hist'))
        total_bits += bits
        prev_dec_sd = dec_sd
        # 压缩后流式评估(避免整段视频驻留内存)
        psnr_c, ms_c, fps_c = tr.eval_stream(state_dict=dec_sd,
                                             fps_reps=(args.fps_reps if k == min(1, K - 1) else 0))
        clip_infos.append(dict(k=k, lo=lo, hi=hi, role='I' if is_I else 'P', params_m=params_m,
                               bits=bits, psnr=psnr_c, msssim=ms_c, fps=fps_c, epochs=ep,
                               ii_eps=ii_eps))
        print(f'--- clip {k} done: bits={bits/1e6:.3f}Mb psnr={psnr_c:.2f} ({time.time()-t0:.0f}s) ---', flush=True)

    # ---------------- 4. 汇总(帧数加权平均) ----------------
    psnr = sum(c['psnr'] * (c['hi'] - c['lo']) for c in clip_infos) / N
    msssim = sum((c['msssim'] or 0) * (c['hi'] - c['lo']) for c in clip_infos) / N
    per_clip_psnr = [c['psnr'] for c in clip_infos]
    bpp = total_bits / (N * video.H * video.W)
    tag = args.tag or f'{args.mode}_{args.preset}'
    res = dict(video=video.name, tag=tag, mode=args.mode, preset=args.preset,
               clip_mode=args.clip_mode, gop_len=(args.gop_len or auto_gop_len(N)),
               gom_size=args.gom_size, disable=sorted(disable), n_frames=N, K=K,
               bounds=bounds, i_flags=i_flags, far_sim=far_sim, tcl_enabled=tcl_enabled,
               total_bits=total_bits, bpp=bpp, psnr=psnr, msssim=msssim,
               psnr_min=float(np.min(per_clip_psnr)), psnr_max=float(np.max(per_clip_psnr)),
               clips=clip_infos, params_per_clip=[c['params_m'] for c in clip_infos],
               fps=clip_infos[min(1, K - 1)]['fps'], wall_s=time.time() - t0)
    out_dir = os.path.join(args.outf, f'{video.name}_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ('clips',)}, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
