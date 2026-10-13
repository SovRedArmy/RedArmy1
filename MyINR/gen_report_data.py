# ==========================================================================================
# gen_report_data.py —— 汇总 v1 全部实验数据为单一 JSON + markdown 摘要
# 供 PPT / 文档生成使用; 诊断实验(W/X/Y 系列)的关键数字为实验记录, 以常量形式内置。
# 用法: python gen_report_data.py   →  输出 report_data.json + report_data.md
# ==========================================================================================
import glob
import json
import os
from collections import defaultdict

from bd_rate import bd_psnr, bd_rate

# ---- 诊断实验关键数字(bunny clip0, 同架构对照; 来自 logs/diag_*.log 实验记录) ----
DIAG = {
    'module_isolation_40ep': {   # 在多尺度基础上单独叠加各模块
        'ms_only': 24.07, 'fft': 23.50, 'flow_lk': 23.51, 'hfb': 23.41,
        'tcl_0.05': 19.88, 'tcl_0.005': 23.16, 'fw_wide_loss': 16.62,
        'full_fixed': 23.12,
    },
    'step_parity': {             # clip0 41帧, S 档全宽度
        'clip0_150ep': 25.59, 'clip0_450ep': 29.69, 'whole_video_150ep': 29.37,
    },
    'quant_engineering': {'asymmetric_perchan_bits': 7.18, 'symmetric_zerofirst': 4.69,
                          'entropy_gap_pct': 0.3},
    'tcl_gate': {'bunny_far_frame_sim': 0.945},
}

ABL_DESC = {
    'A1_ms': '仅多尺度 max-pool 监督(+pair 采样)', 'A2_crspe': '+CRSPE 每帧内容嵌入',
    'A3_fft': '+FFT 频域损失', 'A4_flow': '+LK 伪标签光流(带置信度)',
    'A5_tcl': '+TCL 时序对比', 'A6_fw': '+帧权重(损失调制)',
    'A7_hfb': '+高频增强目标 = 单模型全模块', 'ours': '场景切分多 clip I/P(完整 v1 方法)',
    'ourspar': '多 clip + 步数对齐', 'basedense': '不剪枝 dense 8bit(协议对照)',
}


def main():
    rows = defaultdict(dict)      # video -> tag -> results
    for p in glob.glob('runs/*/results.json'):
        r = json.load(open(p))
        rows[r['video']][r['tag']] = r

    def pts(v, prefix):
        out = []
        for pr in ['S', 'M', 'L']:
            t = f'{prefix}_{pr}'
            if t in rows.get(v, {}):
                r = rows[v][t]
                out.append(dict(preset=pr, psnr=r['psnr'], bpp=r['bpp'],
                                msssim=r.get('msssim'), bits_mb=r['total_bits'] / 1e6,
                                K=r.get('K', 1), fps=r.get('fps')))
        return out

    data = {'videos': {}, 'diagnostics': DIAG, 'abl_desc': ABL_DESC}
    md = ['# v1 实验数据汇总\n']
    for v in sorted(rows):
        vd = {}
        md.append(f'\n## {v}\n')
        md.append('| 方法 | 各码率点 PSNR@bpp | BD-rate vs base |')
        md.append('|---|---|---|')
        base_pts = [(p['psnr'], p['bpp']) for p in pts(v, 'base')]
        for prefix in ['base', 'basedense', 'A1_ms', 'A2_crspe', 'A3_fft', 'A4_flow',
                       'A5_tcl', 'A6_fw', 'A7_hfb', 'ours', 'ourspar']:
            p = pts(v, prefix)
            if not p:
                continue
            cells = ' | '.join(f"{x['psnr']:.2f}@{x['bpp']*1000:.2f}e-3" for x in p)
            br = bp = None
            if prefix != 'base' and len(base_pts) >= 3 and len(p) >= 3:
                try:
                    br = bd_rate(base_pts, [(x['psnr'], x['bpp']) for x in p])
                    bp = bd_psnr(base_pts, [(x['psnr'], x['bpp']) for x in p])
                except Exception:
                    pass
            desc = ABL_DESC.get(prefix, prefix)
            bds = f'{br:+.1f}% (BD-PSNR {bp:+.2f}dB)' if br is not None else \
                  ('anchor' if prefix == 'base' else '点数不足')
            md.append(f'| {prefix}<br>{desc} | {cells} | {bds} |')
            vd[prefix] = dict(points=p, bd_rate=br, bd_psnr=bp)
        # 多 clip 明细
        for prefix in ['ours', 'ourspar']:
            for pr in ['S', 'M', 'L']:
                t = f'{prefix}_{pr}'
                if t in rows[v] and rows[v][t].get('K', 1) > 1:
                    r = rows[v][t]
                    vd.setdefault(prefix, {}).setdefault('clips', {})[pr] = [
                        dict(role=c['role'], lo=c['lo'], hi=c['hi'], psnr=c['psnr'],
                             bits_mb=c['bits'] / 1e6, epochs=c.get('epochs'))
                        for c in r['clips']]
        data['videos'][v] = vd

    open('report_data.json', 'w').write(json.dumps(data, indent=1, default=float))
    open('report_data.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\n-> report_data.json / report_data.md  (videos: {len(data["videos"])})')


if __name__ == '__main__':
    main()
