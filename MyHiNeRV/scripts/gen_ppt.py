"""MyHiNeRV 核心机制与实验结果 PPT（4页）"""
import os, sys
sys.path.insert(0, '/remote-home/gdzj21/tmp/pylibs')
os.environ.setdefault('TMPDIR', '/remote-home/gdzj21/tmp/myhinerbv_tmp')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ===== 配色 =====
BG_DARK  = RGBColor(0x1A, 0x1A, 0x2E)
BG_LIGHT = RGBColor(0xF8, 0xF9, 0xFB)
PRIMARY  = RGBColor(0x4A, 0x90, 0xD9)
ACCENT   = RGBColor(0xE8, 0x5C, 0x3A)
TEXT_D   = RGBColor(0x2D, 0x2D, 0x2D)
TEXT_L   = RGBColor(0xFF, 0xFF, 0xFF)
MUTED    = RGBColor(0x8C, 0x8C, 0x8C)
CARD_BG  = RGBColor(0xEE, 0xF2, 0xF7)
GREEN    = RGBColor(0x27, 0xAE, 0x60)
FONT     = "Arial"
W, H, M = 13.33, 7.5, 0.5

prs = Presentation()
prs.slide_width  = Inches(W)
prs.slide_height = Inches(H)

def add_slide(bg):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid(); r.fill.fore_color.rgb = bg; r.line.fill.background()
    r.shadow.inherit = False
    return s

def tx(s, x, y, w, h, t, sz=18, c=TEXT_D, b=False, al=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left=0; tf.margin_right=0; tf.margin_top=0; tf.margin_bottom=0
    for i, line in enumerate(t.split('\n')):
        p = tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.alignment = al
        r = p.add_run(); r.text = line
        r.font.size = Pt(sz); r.font.color.rgb = c; r.font.bold = b; r.font.name = FONT
    return tb

def card(s, x, y, w, h, fc=CARD_BG):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fc
    sh.line.color.rgb = RGBColor(0xD0,0xD8,0xE0); sh.line.width = Pt(0.5)
    sh.shadow.inherit = False
    return sh

def arrow(s, x, y, w, h):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = PRIMARY; a.line.fill.background()
    a.shadow.inherit = False
    return a

def table(s, x, y, w, h, data, hdr_bg=PRIMARY, hdr_fg=TEXT_L, bsz=11, hsz=12):
    rows, cols = len(data), len(data[0])
    t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for i in range(rows):
        for j in range(cols):
            c = t.cell(i,j)
            c.margin_left = Inches(0.05); c.margin_right = Inches(0.05)
            c.margin_top = Inches(0.02); c.margin_bottom = Inches(0.02)
            if i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = hdr_bg
            else:
                c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFF,0xFF,0xFF) if i%2==1 else CARD_BG
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = data[i][j]
            r.font.name = FONT
            if i == 0:
                r.font.size = Pt(hsz); r.font.color.rgb = hdr_fg; r.font.bold = True
            else:
                r.font.size = Pt(bsz); r.font.color.rgb = TEXT_D
    return t

# ============ SLIDE 1: 标题 ============
s1 = add_slide(BG_DARK)
tx(s1, M, 1.2, W-2*M, 1.2, "MyHiNeRV", 54, TEXT_L, True, PP_ALIGN.CENTER)
tx(s1, M, 2.5, W-2*M, 0.8, "基于训练动态的双权重 INR 视频压缩", 28, PRIMARY, False, PP_ALIGN.CENTER)
tx(s1, M, 3.4, W-2*M, 0.6, "Patch Weighting · Jensen Gap · Two-Level Architecture", 16, MUTED, False, PP_ALIGN.CENTER)
for i, (v, l) in enumerate([("30.71 dB","基线浮点 PSNR\n(HiNeRV-S, 120ep)"),
                              ("3.3 dB","训练/评估差距\n(Jensen 不等式)"),
                              ("1.19M + 0.41M","base + GOP 参数\n(两级架构)")]):
    x = 1.5 + i*3.8
    tx(s1, x, 4.6, 3.2, 0.7, v, 32, ACCENT, True, PP_ALIGN.CENTER)
    tx(s1, x, 5.4, 3.2, 0.8, l, 13, TEXT_L, False, PP_ALIGN.CENTER)
tx(s1, M, 6.8, W-2*M, 0.4, "UVG ReadySetGo 1920×1080 × 600 frames · PyTorch · RTX A6000", 12, MUTED, False, PP_ALIGN.CENTER)

# ============ SLIDE 2: 双权重 + Jensen ============
s2 = add_slide(BG_LIGHT)
tx(s2, M, 0.3, W-2*M, 0.5, "Patch 双权重系统 & 训练/评估 PSNR 差距", 28, TEXT_D, True)

tx(s2, M, 1.0, 5.8, 0.4, "dynamicAL 双权重（同源，不同强度）", 16, PRIMARY, True)
card(s2, M, 1.5, 5.8, 1.2)
tx(s2, M+0.15, 1.6, 5.5, 0.35, "训练采样权重（激进）", 14, TEXT_D, True)
tx(s2, M+0.15, 1.95, 5.5, 0.6, "w = clip(exp(z_score), 0.5, 2.0)\n难 patch 采样频率 ×4  |  不丢弃任何 patch", 12, TEXT_D)
card(s2, M, 2.9, 5.8, 1.2)
tx(s2, M+0.15, 3.0, 5.5, 0.35, "损失权重（保守）", 14, TEXT_D, True)
tx(s2, M+0.15, 3.35, 5.5, 0.6, "w = clip(1 + 0.15·z_score, 0.8, 1.2)\n难 patch 梯度略增 → MSE 更均匀 → 评估 PSNR ↑", 12, TEXT_D)
tx(s2, M, 4.3, 5.8, 0.35, "机制：梯度从均匀分配 → 偏向高误差 patch", 13, ACCENT, True)
for i, t in enumerate(["L1 梯度对所有像素等大 → 难 patch 未被充分优先",
                         "损失加权补偿：温和地把 L1 向 MSE 方向拉",
                         "帧级 MSE 中难 patch 贡献最大 → 改善它们收益最高"]):
    tx(s2, M+0.2, 4.7+i*0.35, 5.5, 0.3, f"•  {t}", 12, TEXT_D)

tx(s2, 7.0, 1.0, 5.8, 0.4, "训练日志 PSNR vs 评估 PSNR（Jensen 偏移）", 16, PRIMARY, True)
card(s2, 7.0, 1.5, 5.8, 2.6)
tx(s2, 7.15, 1.6, 5.5, 0.35, "判别实验：同一权重，两种前向模式", 14, TEXT_D, True)
tx(s2, 7.15, 2.0, 5.5, 0.8, "帧模式 PSNR  = 30.708 dB\npatch 模式 PSNR = 30.708 dB\n差距 = 0.000 dB（逐位一致）", 13, GREEN, True)
tx(s2, 7.15, 2.9, 5.5, 0.9, "结论：前向计算无差异，差距完全来自统计口径\n（小窗口 log 平均 vs 大窗口 log 平均）", 12, TEXT_D)
card(s2, 7.0, 4.3, 5.8, 1.5)
tx(s2, 7.15, 4.4, 5.5, 0.35, "Jensen 不等式（凹函数 log）", 14, TEXT_D, True)
tx(s2, 7.15, 4.8, 5.5, 0.8, "训练：avg[log(MSE_patch)] = 几何平均（偏高）\n评估：log[avg(MSE_frame)] = 算术平均（基准真值）\n差距 ∝ patch 间 MSE 方差", 12, TEXT_D)
tx(s2, M, 6.3, W-2*M, 0.4, "A0 差距 3.3 dB（正常底噪）| A4 差距 5.9 dB（采样扰动 +2.6 dB）| A4R2 目标：收回 ~2 dB", 13, ACCENT, True, PP_ALIGN.CENTER)

# ============ SLIDE 3: 两级架构 + 实验数据 ============
s3 = add_slide(BG_LIGHT)
tx(s3, M, 0.3, W-2*M, 0.5, "两级架构 & 实验数据", 28, TEXT_D, True)
tx(s3, M, 1.0, 6.0, 0.4, "两级架构：base + GOP 残差", 16, PRIMARY, True)
card(s3, M, 1.5, 2.8, 1.5)
tx(s3, M+0.1, 1.6, 2.6, 0.35, "Base 宽模型", 14, TEXT_D, True)
tx(s3, M+0.1, 1.95, 2.6, 0.9, "1.19M 参数\n全视频训练 300ep\nintra 编码（编码一次\n全帧共享）", 11, TEXT_D)
arrow(s3, 3.5, 2.1, 0.6, 0.3)
card(s3, 4.3, 1.5, 2.8, 1.5)
tx(s3, 4.4, 1.6, 2.6, 0.35, "GOP 残差小模型 ×36", 14, TEXT_D, True)
tx(s3, 4.4, 1.95, 2.6, 0.9, "0.41M 参数 / 个\nzero-init 恒等起步\n只学对 base 的修正量\n残差编码（只传差值）", 11, TEXT_D)
tx(s3, M, 3.2, 6.0, 0.35, "关键设计决策", 14, ACCENT, True)
for i, t in enumerate(["所有 GOP 从 base 推导（通道/深度切片），彼此独立（非 D-init）",
                         "自适应切分 12~20 帧/GOP（内容散度 + 训练动态，TeNeRV 贪心）",
                         "权重压缩：I 模型 intra + P 模型残差（DCVC-RT 式 dead-zone + 熵编码）",
                         "QEC：每层可学习 q_step，前向用量化→反量化权重（率失真联合训练）"]):
    tx(s3, M+0.1, 3.6+i*0.38, 6.3, 0.35, f"•  {t}", 11, TEXT_D)

tx(s3, 7.0, 1.0, 5.8, 0.4, "消融实验结果（ReadySetGo 1080p, 120ep）", 16, PRIMARY, True)
table(s3, 7.0, 1.5, 5.8, 2.0, [
    ["变体","参数量","浮点\nPSNR","Q8\nbpp","Q8\nPSNR","说明"],
    ["A0 基线","3.19M","30.71","0.0177","30.18","HiNeRV-S\n等价"],
    ["A4 全模块","3.19M","28.81","0.0181","28.61","−1.9dB\n（调度问题）"],
    ["论文参考\n(300ep)","3.19M","31.53","—","—","HiNeRV-S\n原始结果"],
])
tx(s3, 7.0, 3.8, 5.8, 0.35, "A4 −1.9dB 归因（训练 34.7 vs 评估 28.8）", 14, ACCENT, True)
table(s3, 7.0, 4.2, 5.8, 1.5, [
    ["差距来源","贡献"],
    ["Patch 丢弃 25%（数据不对称）","~1.5 dB"],
    ["帧权重采样偏倚（读数偏高）","~0.7 dB"],
    ["HF 注入（目标偏移）","~0.4 dB"],
])
tx(s3, M, 6.5, W-2*M, 0.4, "A4R2 修复：不丢弃 patch + 保守损失加权 + 中段窗口调度 → 参数扫描 A4R2a~e 进行中", 13, GREEN, True, PP_ALIGN.CENTER)

# ============ SLIDE 4: 模块 + 总结 ============
s4 = add_slide(BG_LIGHT)
tx(s4, M, 0.3, W-2*M, 0.5, "辅助模块 & 总结", 28, TEXT_D, True)
tx(s4, M, 1.0, W-2*M, 0.4, "各模块正面效果", 16, PRIMARY, True)
mods = [("CRSPE","每帧内容嵌入\n内容感知初始化"),
        ("FFT 损失","频域监督\n保高频，零码率"),
        ("SFCM","RAFT 伪标签光流\n时序一致性"),
        ("TCL","时序对比损失\n帧嵌入平滑"),
        ("多尺度金字塔","频率尺度分解\n浅层学全局结构"),
        ("HF 注入","动态高频增强\n放大细节惩罚"),
        ("自适应 GOP","内容+动态切分\n场景对齐"),
        ("QEC/Hyper","可学习量化\n率失真联合")]
for i, (nm, ef) in enumerate(mods):
    col, row = i%4, i//4
    x, y = M+col*3.1, 1.5+row*1.8
    card(s4, x, y, 2.9, 1.6)
    tx(s4, x+0.1, y+0.1, 2.7, 0.35, nm, 14, PRIMARY, True)
    tx(s4, x+0.1, y+0.5, 2.7, 1.0, ef, 11, TEXT_D)

tx(s4, M, 5.3, W-2*M, 0.4, "总结与方向", 16, ACCENT, True)
for i, t in enumerate(["训练/评估 PSNR 差距由 Jensen 不等式解释（3.3 dB 底噪），A4 额外 2.6 dB 来自采样扰动",
                         "dynamicAL 双权重：采样激进（4:1）+ 损失保守（1.45:1），同源不同强度，不丢弃任何 patch",
                         "两级架构：base 1.19M + GOP 0.41M×36，zero-init 恒等起步 + 残差编码",
                         "下一步：A4R2 参数扫描 → 最佳 α；A1 300ep → 首个两级 RD 点"]):
    tx(s4, M+0.1, 5.7+i*0.38, W-2*M-0.2, 0.35, f"•  {t}", 12, TEXT_D)

out = '/remote-home/gdzj21/VideoCompression/MyHiNeRV/output/MyHiNeRV_Summary.pptx'
prs.save(out)
print(f'Saved: {out}')
