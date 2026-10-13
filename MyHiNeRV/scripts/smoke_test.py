"""
冒烟测试：用合成随机数据验证 MyHiNeRV 关键路径的正确性（不触碰真实数据集）。
覆盖：
1) MyHiNeRV 前向（训练 patch 模式 / 评估帧模式）与辅助输出形状；
2) 新损失（FFT/金字塔/高频注入/对比/光流）数值合理（有限、有梯度）；
3) 训练动态跟踪/帧权重/场景切分算法；
4) intra 与 residual 量化的压缩-解压往返（torchac 熵编码）一致性；
5) 光流伪标签 patch 裁剪。
运行：python scripts/smoke_test.py
"""
import os
import sys
import types
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger('smoke')


def base_args(**kw):
    """构造与命令行解析等价的 args（最小 MyHiNeRV 结构）。"""
    d = dict(
        model='MyHiNeRV', block_type='convnext', channels=64, channels_reduce=2.0,
        channels_reduce_base=1, channels_min=1, depths=[2, 1], exps=[2., 1.],
        stem_kernels=1, kernels=[3, 3], scales_t=[1, 1], scales_hw=[2, 2],
        stem_paddings=[-1, -1, -1], paddings=[-1, -1, -1],
        base_size=[-1, -1, -1], base_grid_size=[-1, -1, -1, 2], base_grid_level=1,
        base_grid_level_scale=[1., 1., 1., 1.], base_grid_init_scale=1e-3,
        enc_type='normalized+temp_local_grid', enc_align_corners=False,
        enc_pe=[1.2, 4, 1.2, 4], enc_pe_no_t=False, enc_grid_size=[-1, 4],
        enc_grid_level=1, enc_grid_level_scale=[1., 1.], enc_grid_depth_scale=[1., 1.],
        enc_grid_init_scale=1e-3,
        upsample_type='trilinear', upsample_config='matmul-th-w',
        upsample_norm='layernorm-no-affine', upsample_act='none',
        block_norm='layernorm-no-affine', block_act='gelu', block_bias=False,
        block_layerscale=0., block_dropout=0., block_droppath=0.,
        stem_type='conv_stem', stem_norm='none', stem_act='none', stem_bias=True,
        stem_layerscale=0., stem_dropout=0., stem_droppath=0.,
        head_type='linear_head', head_norm='none', head_act='sigmoid', head_bias=True,
        head_layerscale=0., head_dropout=0., head_droppath=0.,
        eval_patch_size=None,
        crspe_channels=8, crspe_init_scale=1e-2, crspe_content_init=True, crspe_content_size=16,
        multiscale=True, flow_head=True,
        quant_residual=False, quant_residual_deadzone=1, quant_scale_bits=0,
        debug=False,
    )
    d.update(kw)
    return types.SimpleNamespace(**d)


def make_model_and_input(T=8, H=32, W=32, patch=(1, 8, 8)):
    args = base_args()
    from models import myhinerbv
    inp = {'idx': torch.zeros(1, 3, dtype=int), 'idx_max': (T, H // patch[1], W // patch[2]),
           'video_size': (T, H, W), 'patch_size': patch, 'batch_size': 1.0}
    model = myhinerbv.build_model(args, logger, inp)
    return model, args, inp


def test_forward():
    print('== 1) forward (train patch mode / eval frame mode) ==')
    T, H, W = 8, 32, 32
    model, args, _ = make_model_and_input(T, H, W)

    # CRSPE content init
    model.init_from_content(torch.randn(T, 3 * 16 * 16))
    assert model.crspe.weight.abs().mean() > 0

    # Train patch mode
    model.train()
    inp = {'idx': torch.tensor([[1, 2, 2], [0, 0, 0]]), 'idx_max': (T, H // 8, W // 8),
           'video_size': (T, H, W), 'patch_size': (1, 8, 8)}
    out, aux = model.forward_with_aux(inp)
    assert out.shape == (2, 3, 1, 8, 8), out.shape
    assert len(aux['scales']) == 1 and aux['scales'][0].shape == (2, 3, 1, 4, 4)  # ratio 2
    assert aux['scale_ratios'] == [2]
    assert aux['flow'][0].shape == (2, 2, 1, 8, 8) and aux['flow'][2].shape == (2, 2, 1, 8, 8)
    assert aux['frame_idx'].shape == (2, 1)
    loss = out.mean()
    loss.backward()
    g = model.crspe.weight.grad
    assert g is not None and torch.isfinite(g).all()

    # Eval frame mode
    model.eval()
    inp = {'idx': torch.tensor([[0, 0, 0], [1, 0, 0]]), 'idx_max': (T, 1, 1),
           'video_size': (T, H, W), 'patch_size': (1, H, W)}
    with torch.no_grad():
        out = model(inp)
    assert out.shape == (2, 3, 1, H, W), out.shape

    # Eval patch mode (eval_patch_size)
    args2 = base_args(eval_patch_size=[1, 16, 16])
    from models import myhinerbv
    inp2 = {'idx': torch.tensor([[0, 0, 0]]), 'idx_max': (T, 1, 1),
            'video_size': (T, H, W), 'patch_size': (1, H, W)}
    m2 = myhinerbv.build_model(args2, logger, inp2)
    m2.eval()
    with torch.no_grad():
        out2 = m2(inp2)
    assert out2.shape == (1, 3, 1, H, W)
    print('   OK')


def test_losses():
    print('== 2) aux losses ==')
    from losses import (fft_loss, maxpool_pyramid, highpass, highfreq_enhanced_target,
                        temporal_contrastive_loss, flow_constraint_loss)
    x = torch.rand(2, 3, 1, 32, 32)
    y = torch.rand(2, 3, 1, 32, 32)
    l = fft_loss(x, y, high_freq_gamma=2.)
    assert l.shape == (2, 1) and torch.isfinite(l).all()
    pyr = maxpool_pyramid(y, [4, 8])
    assert pyr[0].shape == (2, 3, 1, 8, 8) and pyr[1].shape == (2, 3, 1, 4, 4)
    hp = highpass(x)
    assert hp.shape == x.shape
    tgt = highfreq_enhanced_target(y, x, beta=2.)
    assert tgt.shape == y.shape and torch.isfinite(tgt).all()

    emb = torch.randn(10, 8, requires_grad=True)
    l_tcl = temporal_contrastive_loss(emb, torch.tensor([1, 2, 5, 9]))
    assert l_tcl is not None and torch.isfinite(l_tcl) and l_tcl.requires_grad
    assert temporal_contrastive_loss(emb, torch.tensor([3])) is None

    pf = torch.randn(2, 2, 1, 32, 32)
    conf = torch.softmax(torch.randn(2, 2, 1, 32, 32), dim=1)
    valid = torch.tensor([[1.], [0.]])
    lf, le = flow_constraint_loss(pf, pf, conf, pf * 0.5, pf * 0.5, valid=valid)
    assert torch.isfinite(lf) and torch.isfinite(le)
    print('   OK')


def test_dynamics():
    print('== 3) dynamics / segmentation / patch dropout ==')
    from dynamics import (FrameDynamicsTracker, PatchDynamicsTracker, compute_frame_weights,
                          frame_patch_weights, segment_clips, compute_patch_drop_mask)
    T = 100
    tr = FrameDynamicsTracker(T, window=5, var_weight=2.0)
    rng = np.random.RandomState(0)
    for e in range(10):
        losses = {t: 0.1 + rng.rand() * (1. if t % 30 < 3 else 0.1) for t in range(T)}
        tr.update(e, losses)
    s = tr.get_scores()
    assert s.shape == (T,) and np.isfinite(s).all()
    fw = compute_frame_weights(s, temperature=0.5)
    assert fw.shape == (T,) and fw.min() >= 0.2 - 1e-9 and fw.max() <= 5.0 + 1e-9
    pw = frame_patch_weights(fw, (T, 4, 4))
    assert pw.shape == (T * 16,)

    # Patch 级动态 + dynamicAL 丢弃 25%
    pt = PatchDynamicsTracker((10, 4, 4), window=5, var_weight=2.0)
    P = 10 * 16
    for e in range(8):
        psum = np.abs(rng.randn(P)) * (1. + 2. * (np.arange(P) % 5 == 0))  # 部分 patch 损失高（高动态）
        pcnt = np.ones(P)
        pt.update(e, psum, pcnt)
    ps = pt.get_scores()
    assert ps.shape == (P,) and pt.get_frame_scores().shape == (10,)
    keep = compute_patch_drop_mask(ps, drop_ratio=0.25)
    assert abs(keep.mean() - 0.75) < 1e-9
    dropped = np.where(~keep)[0]
    kept = np.where(keep)[0]
    assert ps[dropped].max() <= ps[kept].min() + 1e-12, '必须丢的是最低动态 patch'
    pw2 = frame_patch_weights(np.ones(10), (10, 4, 4), keep_mask=keep)
    assert (pw2.numpy()[~keep] == 0).all() and (pw2.numpy()[keep] > 0).all()

    # 场景切分：在 t=33、66 注入突变
    div = np.abs(np.random.randn(T)) * 0.01
    div[33] = 2.0
    div[66] = 1.8
    clips = segment_clips(div, dynamics=s, num_clips=3, min_len=8)
    bounds = [c[1] for c in clips[:-1]]
    assert len(clips) == 3 and bounds[0] == 33 and bounds[1] == 66, clips
    auto = segment_clips(div, dynamics=s, num_clips=None, clip_len=25, min_len=8)
    assert all(b - a >= 8 for a, b in auto)
    print('   OK', clips)


def test_quantization_roundtrip():
    print('== 4) quantization & entropy coding roundtrip (intra + residual) ==')
    from compression import quant_utils, prune_utils, codec_utils
    from hinerv_compress import initial_parametrizations, compress_bitstream, decompress, get_bitstream_state_dict

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16)
            self.fc2 = nn.Linear(16, 4)
            self.bitstream_prefix = ('fc1', 'fc2')
            self.no_bitstream_prefix = ()
            self.no_prune_prefix = ()
            self.no_quant_prefix = ()
    torch.manual_seed(0)
    model = Tiny()
    args = base_args(quant_residual=False)
    initial_parametrizations(args, logger, model)

    class FakeAcc:
        is_main_process = True
        def broadcast_object_list(self, x, **k):
            return x
        def broadcast(self, x, **k):
            return x

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ref = get_bitstream_state_dict(model)
        nb = compress_bitstream(args, logger, FakeAcc(), model, td, 8)
        assert nb > 0
        w0 = {k: v.clone() for k, v in ref.items()}
        from hinerv_compress import set_zero
        set_zero(model)
        decompress(args, logger, FakeAcc(), model, td, 8)
        dec = get_bitstream_state_dict(model)
        for k in w0:
            assert torch.allclose(w0[k], dec[k], atol=1e-2), (k, (w0[k] - dec[k]).abs().max())

    # Residual roundtrip: 模型权重微扰后做残差压缩，参考 = 原 intra 解码权重
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    args_r = base_args(quant_residual=True, quant_residual_deadzone=2)
    with tempfile.TemporaryDirectory() as td:
        before = get_bitstream_state_dict(model)
        nb_r = compress_bitstream(args_r, logger, FakeAcc(), model, td, 8, reference_state_dict=ref)
        set_zero(model)
        decompress(args_r, logger, FakeAcc(), model, td, 8, reference_state_dict=ref)
        after = get_bitstream_state_dict(model)
        for k in before:
            assert torch.allclose(before[k], after[k], atol=1e-2), (k, (before[k] - after[k]).abs().max())
    print(f'   OK (intra {nb} bytes, residual {nb_r} bytes)')


def test_flow_provider():
    print('== 5) flow provider patch cropping ==')
    from flow_utils import FlowProvider
    import tempfile
    T, H, W = 6, 32, 32
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'f.pt')
        torch.save({'flow_fwd': torch.randn(T, 2, H // 2, W // 2, dtype=torch.float16),
                    'flow_bwd': torch.randn(T, 2, H // 2, W // 2, dtype=torch.float16),
                    'size': (H // 2, W // 2)}, p)
        fp = FlowProvider(p, video_size=(T, H, W), frame_offset=0)
        f, b, valid = fp.get_patch_flow((1, 1, 1), (1, 8, 8))
        assert f.shape == (2, 1, 8, 8) and b.shape == (2, 1, 8, 8) and valid
        f0, b0, v0 = fp.get_patch_flow((0, 0, 0), (1, 8, 8))
        assert not v0 and f0.abs().sum() == 0
    print('   OK')



def test_derive():
    print("== 6) derive: wide base -> small GOP model ==")
    from models import myhinerbv
    from derive import derive_state_dict
    import logging
    logger = logging.getLogger('derive_dbg'); logger.addHandler(logging.NullHandler())

    T, H, W = 12, 32, 32
    # 宽 base：channels 64, depths [2,2]；小模型：channels 32, depths [1,1]，clip 5 帧
    wide_args = base_args(channels=64, depths=[2, 2])
    inp = {'idx': torch.zeros(1, 3, dtype=int), 'idx_max': (T, 2, 2),
           'video_size': (T, H, W), 'patch_size': (1, 16, 16)}
    wide = myhinerbv.build_model(wide_args, logger, inp)

    small_args = base_args(channels=32, depths=[1, 1])
    inp_s = {'idx': torch.zeros(1, 3, dtype=int), 'idx_max': (5, 2, 2),
             'video_size': (5, H, W), 'patch_size': (1, 16, 16)}
    small = myhinerbv.build_model(small_args, logger, inp_s)

    wide_state = {k: v.detach().clone() for k, v in wide.state_dict().items()}
    n_params = sum(v.numel() for v in wide_state.values())
    derived = derive_state_dict(small, wide_state, logger=logger)
    small.load_state_dict(derived, strict=False)
    # 推导权重应与 base 对应切片一致（以 stem 为例）
    assert 'decoder.stem.conv.weight' in derived
    assert torch.allclose(derived['decoder.stem.conv.weight'],
                          wide_state['decoder.stem.conv.weight'][:derived['decoder.stem.conv.weight'].shape[0], :derived['decoder.stem.conv.weight'].shape[1]])
    # CRSPE 时间插值：小模型 [5, C] 从 base [12, C_w]
    assert derived['crspe.weight'].shape == small.state_dict()['crspe.weight'].shape
    # 推导后能正常前向（帧模式：每个样本一整帧）
    small.eval()
    inp_f = {'idx': torch.tensor([[0, 0, 0]]), 'idx_max': (5, 1, 1),
             'video_size': (5, H, W), 'patch_size': (1, H, W)}
    with torch.no_grad():
        out = small(inp_f)
    assert out.shape == (1, 3, 1, H, W)
    print(f'   OK (wide {n_params} params -> small {sum(v.numel() for v in small.state_dict().values())})')


def test_qec_roundtrip():
    print("== 7) QEC quantization & gaussian entropy coding roundtrip ==")
    from compression import quant_utils, codec_utils
    from hinerv_compress import initial_parametrizations, compress_bitstream, decompress, get_bitstream_state_dict, set_zero
    import tempfile

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16)
            self.fc2 = nn.Linear(16, 4)
            self.bitstream_prefix = ('fc1', 'fc2')
            self.no_bitstream_prefix = ()
            self.no_prune_prefix = ()
            self.no_quant_prefix = ()
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class FakeAcc:
        is_main_process = True
        def broadcast_object_list(self, x, **k):
            return x
        def broadcast(self, x, **k):
            return x

    torch.manual_seed(0)
    model = Tiny()
    args = base_args(quant_mode='qec', quant_residual=False)
    initial_parametrizations(args, logger, model)

    # 训练几步让 q_step 学习 + 启动
    from compression.quant_utils import start_quantization, get_rate_bpp
    quant_utils.set_quantization(args, logger, model, 8, 0.9, True)
    start_quantization(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, 16)
    model(x).sum().backward()
    opt.step()
    model.eval()

    with tempfile.TemporaryDirectory() as td:
        ref = get_bitstream_state_dict(model)
        nb = compress_bitstream(args, logger, FakeAcc(), model, td, 8)
        assert nb > 0
        w0 = {k: v.clone() for k, v in ref.items()}
        set_zero(model)
        decompress(args, logger, FakeAcc(), model, td, 8)
        dec = get_bitstream_state_dict(model)
        for k in w0:
            assert torch.allclose(w0[k], dec[k], atol=5e-2), (k, (w0[k] - dec[k]).abs().max())
    print(f'   OK (qec intra {nb} bytes)')



def test_hyper_roundtrip():
    print("== 8) hyper (DCVC-RT-style per-element mu/sigma) roundtrip ==")
    from compression import quant_utils
    from compression.hyperprior import attach_hyperprior, compute_hyper_rate_bpp
    from hinerv_compress import initial_parametrizations, compress_bitstream, decompress, get_bitstream_state_dict, set_zero
    import tempfile

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16)
            self.fc2 = nn.Linear(16, 4)
            self.bitstream_prefix = ('fc1', 'fc2')
            self.no_bitstream_prefix = ()
            self.no_prune_prefix = ()
            self.no_quant_prefix = ()
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class FakeAcc:
        is_main_process = True
        def broadcast_object_list(self, x, **k):
            return x
        def broadcast(self, x, **k):
            return x

    torch.manual_seed(0)
    model = Tiny()
    args = base_args(quant_mode='hyper')
    initial_parametrizations(args, logger, model)
    attach_hyperprior(args, logger, model)
    assert model.hyper is not None

    # QAT：启动量化并训练几步（率损失进入 task 层；此处直接验证 rate 估计可微）
    quant_utils.start_quantization(model)
    model.train()
    x = torch.randn(8, 16)
    out = model(x)
    loss = out.sum() + 0.01 * compute_hyper_rate_bpp(model, 8 * 16 * 16)
    loss.backward()
    g = model.hyper.mlp[0].weight.grad
    assert g is not None and torch.isfinite(g).all(), 'hyper rate loss must be differentiable'
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    opt.step()
    model.eval()

    with tempfile.TemporaryDirectory() as td:
        ref = get_bitstream_state_dict(model)
        nb = compress_bitstream(args, logger, FakeAcc(), model, td, 8)
        assert nb > 0
        w0 = {k: v.clone() for k, v in ref.items()}
        set_zero(model)
        decompress(args, logger, FakeAcc(), model, td, 8)
        dec = get_bitstream_state_dict(model)
        for k in w0:
            d = (w0[k] - dec[k]).abs().max()
            assert torch.allclose(w0[k], dec[k], atol=5e-2), (k, d)
    print(f'   OK (hyper intra {nb} bytes)')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_forward()
    test_losses()
    test_dynamics()
    test_flow_provider()
    test_quantization_roundtrip()
    test_derive()
    test_qec_roundtrip()
    test_hyper_roundtrip()
    print('\nALL SMOKE TESTS PASSED')
