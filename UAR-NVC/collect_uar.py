import re, glob

for cfg, label in [('ms09', 'UAR ms=0.9M  I60/P24'), ('ms18', 'UAR ms=1.8M  I60/P24')]:
    ps, bps, bpv = [], [], []
    per = []
    for f in sorted(glob.glob(f'logs_scaleup/{cfg}_g*.log')):
        for line in open(f, errors='ignore'):
            m = re.search(r'model_idx: (\d+).*after_psnr: ([0-9.]+), total_bpp: ([0-9.]+),.*bp_param: ([0-9.]+)', line)
            if m:
                per.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    per.sort()
    for idx, p, b, bp in per:
        ps.append(p); bps.append(b); bpv.append(bp)
    n = len(per)
    print(f'== {label}: {n} 模型')
    print(f'   平均 PSNR={sum(ps)/n:.2f} dB, 平均 total_bpp={sum(bps)/n:.4f}, 平均 bp_param={sum(bpv)/n:.3f}')
    im = [x for x in per if x[0] % 5 == 0]
    pm = [x for x in per if x[0] % 5 != 0]
    if im: print(f'   I模型({len(im)}个): 平均 {sum(x[1] for x in im)/len(im):.2f} dB, bpp {sum(x[2] for x in im)/len(im):.4f}')
    if pm: print(f'   P模型({len(pm)}个): 平均 {sum(x[1] for x in pm)/len(pm):.2f} dB, bpp {sum(x[2] for x in pm)/len(pm):.4f}')
