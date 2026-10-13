# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ============================================================================
# DCVC-RT 测试入口: "编码 -> 写码流 -> 读码流 -> 解码 -> 计算失真" 的完整闭环。
#
# 运行方式(见 README):
#   python test_video.py --model_path_i ... --model_path_p ... --rate_num 4 \
#       --test_config ./dataset_config_example_yuv420.json --cuda 1 -w 1 \
#       --write_stream 1 --force_zero_thres 0.12 --output_path output.json ...
#
# 多进程结构: main() 用 ProcessPoolExecutor 把每个(序列, qp)组合提交为一个任务;
# 每个进程在 init_func 中加载 DMCI(I 帧)与 DMC(P 帧)模型, worker() 调
# run_one_point_with_stream() 完成一个测试点的全部编解码。
# ============================================================================

import argparse
import concurrent.futures
import io
import json
import multiprocessing
import os
import time

import torch
import numpy as np
from tqdm import tqdm

from src.layers.cuda_inference import replicate_pad
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import str2bool, create_folder, generate_log_json, get_state_dict, \
    dump_json, set_torch_env
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
from src.utils.video_reader import PNGReader, YUV420Reader
from src.utils.video_writer import PNGWriter, YUV420Writer
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np


def parse_args():
    """解析命令行参数(码率/模型路径/并行度/输出路径等)。"""
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--force_zero_thres', type=float, default=None, required=False)
    parser.add_argument('--model_path_i', type=str)
    parser.add_argument('--model_path_p',  type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--qp_i', type=int, nargs="+")
    parser.add_argument('--qp_p', type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='GPU indexes to use')
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--write_stream', type=str2bool, default=False)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img, device):
    """uint8 numpy 图 -> [0,1] float32 张量 (1,C,H,W)。"""
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):
    """按数据集配置构造读取器: png(RGB 序列)或 yuv420(裸 YUV 文件)。"""
    if args['src_type'] == 'png':
        src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])
    elif args['src_type'] == 'yuv420':
        src_reader = YUV420Reader(args['src_path'], args['src_width'], args['src_height'])
    return src_reader


def get_src_frame(args, src_reader, device):
    """读入一帧源图, 统一转为 YCbCr444 半精度张量 x(模型输入域)。

    YUV420: 上采样 UV 到 444; PNG: RGB 转 YCbCr。同时保留原始 numpy 帧(y,u,v/rgb)
    供失真计算(失真在原始分辨率与格式上算, 而非 padding 后)。"""
    if args['src_type'] == 'yuv420':
        y, uv = src_reader.read_one_frame()
        yuv = ycbcr420_to_444_np(y, uv)
        x = np_image_to_tensor(yuv, device)
        y = y[0, :, :]
        u = uv[0, :, :]
        v = uv[1, :, :]
        rgb = None
    else:
        assert args['src_type'] == 'png'
        rgb = src_reader.read_one_frame()
        x = np_image_to_tensor(rgb, device)
        x = rgb2ycbcr(x)
        y, u, v = None, None, None

    x = x.to(torch.float16)
    return x, y, u, v, rgb


def get_distortion(args, x_hat, y, u, v, rgb):
    """计算一帧的失真: 重建帧从 YCbCr444 转回原始格式后与源帧比对。

    YUV420: 转 420 后分别算 Y/U/V 的 PSNR(和 MS-SSIM), 按 6:1:1 加权;
    PNG: 转 RGB 后算 PSNR(和 MS-SSIM)。"""
    if args['src_type'] == 'yuv420':
        y_rec, uv_rec = yuv_444_to_420(x_hat)
        y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        y_rec = y_rec[0, :, :]
        u_rec = uv_rec[0, :, :]
        v_rec = uv_rec[1, :, :]
        psnr_y = calc_psnr(y, y_rec)
        psnr_u = calc_psnr(u, u_rec)
        psnr_v = calc_psnr(v, v_rec)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
        if args['calc_ssim']:
            ssim_y = calc_msssim(y, y_rec)
            ssim_u = calc_msssim(u, u_rec)
            ssim_v = calc_msssim(v, v_rec)
        else:
            ssim_y, ssim_u, ssim_v = 0., 0., 0.
        ssim = (6 * ssim_y + ssim_u + ssim_v) / 8

        curr_psnr = [psnr, psnr_y, psnr_u, psnr_v]
        curr_ssim = [ssim, ssim_y, ssim_u, ssim_v]
    else:
        assert args['src_type'] == 'png'
        rgb_rec = ycbcr2rgb(x_hat)
        rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        psnr = calc_psnr(rgb, rgb_rec)
        if args['calc_ssim']:
            msssim = calc_msssim_rgb(rgb, rgb_rec)
        else:
            msssim = 0.
        curr_psnr = [psnr]
        curr_ssim = [msssim]
    return curr_psnr, curr_ssim


def run_one_point_with_stream(p_frame_net, i_frame_net, args):
    """一个测试点(单序列 x 单 qp)的完整流程: 全序列编码 -> 写文件 -> 读文件 -> 全序列解码。

    阶段一(编码): 逐帧 get_src_frame -> replicate_pad -> I 帧 DMCI.compress /
      P 帧 DMC.compress(按 index_map 做 qp 偏移, 周期 reset_interval 触发 ada_i 重置) ->
      write_sps/write_ip 拼装码流。
    阶段二(解码): read_header 逐 NAL 解析 -> DMCI.decompress / DMC.decompress ->
      裁剪回原尺寸 -> get_distortion 计算失真。
    最后 generate_log_json 汇总并写 json。"""
    if args['check_existing'] and os.path.exists(args['curr_json_path']) and \
            os.path.exists(args['curr_bin_path']):
        with open(args['curr_json_path']) as f:
            log_result = json.load(f)
            if log_result['i_frame_num'] + log_result['p_frame_num'] == args['frame_num']:
                return log_result
            print(f"incorrect log for {args['curr_json_path']}, try to rerun.")

    frame_num = args['frame_num']
    save_decoded_frame = args['save_decoded_frame']
    verbose = args['verbose']
    reset_interval = args['reset_interval']
    intra_period = args['intra_period']
    verbose_json = args['verbose_json']
    device = next(i_frame_net.parameters()).device

    src_reader = get_src_reader(args)
    pic_height = args['src_height']
    pic_width = args['src_width']
    # 补齐到 16 的倍数: I 帧(1/16 隐变量)与 P 帧共用该 padding
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)

    # 大分辨率(>720p)启用双熵编码器, 加速 CPU 算术编码
    use_two_entropy_coders = pic_height * pic_width > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    frame_types = []
    psnrs = []
    msssims = []
    bits = []

    start_time = time.time()
    encoding_time = []
    decoding_time = []
    # 8 帧组的 qp 偏移索引: 配合 DMC.qp_shift=[0,8,4] 实现 [0,8,0,4,0,4,0,4] 分层质量
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    output_buff = io.BytesIO()
    sps_helper = SPSHelper()

    p_frame_net.set_curr_poc(0)
    with torch.no_grad():
        last_qp = 0
        for frame_idx in range(frame_num):
            x, y, u, v, rgb = get_src_frame(args, src_reader, device)

            torch.cuda.synchronize(device=device)
            frame_start_time = time.time()

            # pad if necessary
            x_padded = replicate_pad(x, padding_b, padding_r)

            is_i_frame = False
            if frame_idx == 0 or (intra_period > 0 and frame_idx % intra_period == 0):
                # ---- I 帧: 用 DMCI 压缩, 并把重建帧放入 P 帧模型的 DPB 作参考 ----
                is_i_frame = True
                curr_qp = args['qp_i']
                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'ec_part': 1 if use_two_entropy_coders else 0,
                    'use_ada_i': 0,
                }
                encoded = i_frame_net.compress(x_padded, args['qp_i'])
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, encoded['x_hat'])
                frame_types.append(0)
            else:
                # ---- P 帧: 按 8 帧组内位置偏移 qp(分层质量),
                #      每隔 reset_interval 帧触发一次 ada_i 像素域重置 ----
                fa_idx = index_map[frame_idx % 8]
                if reset_interval > 0 and frame_idx % reset_interval == 1:
                    use_ada_i = 1
                    p_frame_net.prepare_feature_adaptor_i(last_qp)
                else:
                    use_ada_i = 0
                curr_qp = p_frame_net.shift_qp(args['qp_p'], fa_idx)
                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'ec_part': 1 if use_two_entropy_coders else 0,
                    'use_ada_i': use_ada_i,
                }

                encoded = p_frame_net.compress(x_padded, curr_qp)
                last_qp = curr_qp
                frame_types.append(1)

            # SPS 去重: 内容未变则复用旧 id, 不重复写; 变化则写新 SPS
            sps_id, sps_new = sps_helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            sps_bytes = 0
            if sps_new:
                sps_bytes = write_sps(output_buff, sps)
                if verbose >= 2:
                    print("new sps", sps)
            stream_bytes = write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])
            bits.append(stream_bytes * 8 + sps_bytes * 8)

            torch.cuda.synchronize(device=device)
            frame_end_time = time.time()

            frame_time = frame_end_time - frame_start_time
            encoding_time.append(frame_time)

            if verbose >= 2:
                print(f"frame {frame_idx} encoded, {frame_time * 1000:.3f} ms, "
                      f"bits: {bits[-1]}")

    src_reader.close()
    with open(args['curr_bin_path'], "wb") as output_file:
        bytes_buffer = output_buff.getbuffer()
        output_file.write(bytes_buffer)
        total_bytes = bytes_buffer.nbytes
        bytes_buffer.release()
    total_kbps = int(total_bytes * 8 / (frame_num / 30) / 1000)  # assume 30 fps
    output_buff.close()
    sps_helper = SPSHelper()
    input_file = open(args['curr_bin_path'], "rb")
    with open(args['curr_bin_path'], "rb") as input_file:
        input_buff = io.BytesIO(input_file.read())
    decoded_frame_number = 0
    src_reader = get_src_reader(args)

    if save_decoded_frame:
        if args['src_type'] == 'png':
            recon_writer = PNGWriter(args['bin_folder'], args['src_width'], args['src_height'])
        elif args['src_type'] == 'yuv420':
            output_yuv_path = args['curr_rec_path'].replace('.yuv', f'_{total_kbps}kbps.yuv')
            recon_writer = YUV420Writer(output_yuv_path, args['src_width'], args['src_height'])

    p_frame_net.set_curr_poc(0)
    with torch.no_grad():
        while decoded_frame_number < frame_num:
            x, y, u, v, rgb = get_src_frame(args, src_reader, device)
            torch.cuda.synchronize(device=device)
            frame_start_time = time.time()

            # 解析 NAL 头: 先吃掉所有新出现的 SPS, 再处理 I/P 帧
            header = read_header(input_buff)
            while header['nal_type'] == NalType.NAL_SPS:
                sps = read_sps_remaining(input_buff, header['sps_id'])
                sps_helper.add_sps_by_id(sps)
                if verbose >= 2:
                    print("new sps", sps)
                header = read_header(input_buff)
                continue
            sps_id = header['sps_id']

            sps = sps_helper.get_sps_by_id(sps_id)
            qp, bit_stream = read_ip_remaining(input_buff)

            if header['nal_type'] == NalType.NAL_I:
                # I 帧: 解码后把重建帧放入 P 帧模型 DPB 作初始参考
                decoded = i_frame_net.decompress(bit_stream, sps, qp)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, decoded['x_hat'])
            elif header['nal_type'] == NalType.NAL_P:
                # P 帧: use_ada_i 标志要求先丢弃旧特征(改从像素域重提特征)
                if sps['use_ada_i']:
                    p_frame_net.reset_ref_feature()
                decoded = p_frame_net.decompress(bit_stream, sps, qp)

            recon_frame = decoded['x_hat']
            # 裁掉 padding, 回到原始分辨率后再算失真
            x_hat = recon_frame[:, :, :pic_height, :pic_width]

            torch.cuda.synchronize(device=device)
            frame_end_time = time.time()

            frame_time = frame_end_time - frame_start_time
            decoding_time.append(frame_time)

            curr_psnr, curr_ssim = get_distortion(args, x_hat, y, u, v, rgb)
            psnrs.append(curr_psnr)
            msssims.append(curr_ssim)

            if verbose >= 2:
                stream_length = 0 if bit_stream is None else len(bit_stream) * 8
                print(f"frame {decoded_frame_number} decoded, {frame_time * 1000:.3f} ms, "
                      f"bits: {stream_length}, PSNR: {curr_psnr[0]:.4f} ")

            if save_decoded_frame:
                if args['src_type'] == 'yuv420':
                    y_rec, uv_rec = yuv_444_to_420(x_hat)
                    y_rec = torch.clamp(y_rec * 255, 0, 255).round().to(dtype=torch.uint8)
                    y_rec = y_rec.squeeze(0).cpu().numpy()
                    uv_rec = torch.clamp(uv_rec * 255, 0, 255).to(dtype=torch.uint8)
                    uv_rec = uv_rec.squeeze(0).cpu().numpy()
                    recon_writer.write_one_frame(y_rec, uv_rec)
                else:
                    assert args['src_type'] == 'png'
                    rgb_rec = ycbcr2rgb(x_hat)
                    rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).round().to(dtype=torch.uint8)
                    rgb_rec = rgb_rec.squeeze(0).cpu().numpy()
                    recon_writer.write_one_frame(rgb_rec)
            decoded_frame_number += 1
    input_buff.close()
    src_reader.close()

    if save_decoded_frame:
        recon_writer.close()

    test_time = time.time() - start_time
    test_time_frame_numuber = len(encoding_time)
    time_bypass_frame_num = 10  # bypass the first 10 frames as warmup
    # verbose>=1 时统计平均编/解码耗时(跳过前 10 帧预热), I/O 与失真计算时间不计入
    if verbose >= 1 and test_time_frame_numuber > time_bypass_frame_num:
        encoding_time = encoding_time[time_bypass_frame_num:]
        decoding_time = decoding_time[time_bypass_frame_num:]
        avg_encoding_time = sum(encoding_time)/len(encoding_time)
        avg_decoding_time = sum(decoding_time)/len(decoding_time)
        print(f"encoding/decoding {test_time_frame_numuber} frames, "
              f"average encoding time {avg_encoding_time * 1000:.3f} ms, "
              f"average decoding time {avg_decoding_time * 1000:.3f} ms.")
    else:
        avg_encoding_time = None
        avg_decoding_time = None

    log_result = generate_log_json(frame_num, pic_height * pic_width, test_time,
                                   frame_types, bits, psnrs, msssims, verbose=verbose_json,
                                   avg_encoding_time=avg_encoding_time,
                                   avg_decoding_time=avg_decoding_time,)
    with open(args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)
    return log_result


i_frame_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess
p_frame_net = None


def worker(args):
    """进程池任务函数: 组织单个测试点的路径, 调 run_one_point_with_stream 并补充元信息。"""
    global i_frame_net
    global p_frame_net

    sub_dir_name = args['seq']
    bin_folder = os.path.join(args['stream_path'], args['ds_name'])
    assert args['write_stream'], ""
    create_folder(bin_folder, True)

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['curr_bin_path'] = os.path.join(bin_folder,
                                         f"{args['seq']}_q{args['qp_i']}.bin")
    args['curr_rec_path'] = args['curr_bin_path'].replace('.bin', '.yuv')
    args['curr_json_path'] = args['curr_bin_path'].replace('.bin', '.json')

    result = run_one_point_with_stream(p_frame_net, i_frame_net, args)

    result['ds_name'] = args['ds_name']
    result['seq'] = args['seq']
    result['rate_idx'] = args['rate_idx']
    result['qp_i'] = args['qp_i']
    result['qp_p'] = args['qp_p'] if 'qp_p' in args else args['qp_i']

    return result


def init_func(args, gpu_num):
    """进程池 initializer: 设置确定性环境, 按进程号分配 GPU, 加载并半精度化两个模型。

    update() 在此构建所有 qp 的 CDF 表并注册给 rANS 编解码器。"""
    set_torch_env()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        if args.cuda_idx is not None:
            gpu_id = args.cuda_idx[gpu_id]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = "cuda:0"
    else:
        device = "cpu"

    global i_frame_net
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    i_frame_net.update(args.force_zero_thres)
    i_frame_net.half()

    global p_frame_net
    p_frame_net = DMC()
    if not args.force_intra:
        p_state_dict = get_state_dict(args.model_path_p)
        p_frame_net.load_state_dict(p_state_dict)
        p_frame_net = p_frame_net.to(device)
        p_frame_net.eval()
        p_frame_net.update(args.force_zero_thres)
        p_frame_net.half()


def main():
    """主函数: 解析参数 -> 读数据集配置 -> 生成 (序列 x qp) 任务列表 -> 提交进程池

    -> 汇总所有结果写 output json。"""
    begin_time = time.time()

    args = parse_args()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    if args.cuda_idx is not None:
        cuda_device = ','.join([str(s) for s in args.cuda_idx])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num,
                                                                 initializer=init_func,
                                                                 initargs=(args, gpu_num))
    objs = []

    count_frames = 0
    count_sequences = 0

    # 生成 qp 列表: 未显式指定 qp_i 时, 在 [0, 63] 均匀取 rate_num 个码率点
    rate_num = args.rate_num
    qp_i = []
    if args.qp_i is not None:
        assert len(args.qp_i) == rate_num
        qp_i = args.qp_i
    else:
        assert 2 <= rate_num <= DMC.get_qp_num()
        for i in np.linspace(0, DMC.get_qp_num() - 1, num=rate_num):
            qp_i.append(int(i+0.5))

    if not args.force_intra:
        if args.qp_p is not None:
            assert len(args.qp_p) == rate_num
            qp_p = args.qp_p
        else:
            qp_p = qp_i

    print(f"testing {rate_num} rates, using qp: ", end='')
    for q in qp_i:
        print(f"{q}, ", end='')
    print()

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['qp_i'] = qp_i[rate_idx]
                if not args.force_intra:
                    cur_args['qp_p'] = qp_p[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['reset_interval'] = args.reset_interval
                cur_args['seq'] = seq
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq]['width']
                cur_args['intra_period'] = config[ds_name]['sequences'][seq]['intra_period']
                if args.force_intra:
                    cur_args['intra_period'] = 1
                if args.force_intra_period > 0:
                    cur_args['intra_period'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['calc_ssim'] = args.calc_ssim
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['check_existing'] = args.check_existing
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['verbose_json'] = args.verbose_json

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(worker, cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    # 按 数据集 -> 序列 -> 码率点 三级结构组织结果并写出
    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}

    for res in results:
        log_result[res['ds_name']][res['seq']][f"{res['rate_idx']:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
