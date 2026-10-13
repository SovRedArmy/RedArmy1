import json, os
runs = ['bunny_v3_M', 'bunny_v3_M_e1600', 'bunny_v3_M_k50',
        'bunny_v4_M_lora4', 'bunny_v4_M_lora8', 'bunny_v4_M_lora16', 'bunny_v4_M_chain',
        'beauty_v3_S']
for r in runs:
    p = f'runs/{r}/results.json'
    if not os.path.isfile(p):
        print(f'{r}: 无 results.json'); continue
    d = json.load(open(p))
    print(f'== {r}: {d["psnr"]:.2f}dB @ {d["bpp"]:.4f}bpp, 底座 {d["base_psnr"]:.2f}dB, wall {d["wall_s"]/3600:.1f}h')
    for b in d['branches']:
        print(f'   [{b["lo"]:3d}:{b["hi"]:3d}] {b["psnr_before"]:.2f}->{b["psnr_after"]:.2f} (+{b["psnr_after"]-b["psnr_before"]:.2f})  {b["bits"]/1e6:.3f}Mb  {b.get("bpp_param", float("nan")):.2f}b/p  {b.get("init","?")[:12]}/{b.get("delta","?")}')
