# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# DCVC-UF 视频压缩测试主入口：
#   main() 解析参数与测试配置，用多进程（每进程绑定一张 GPU）并行测试多序列多码率；
#   每个任务 worker() → run_one_point_with_stream() 完成一个序列一个码率点的
#   完整“编码→写码流→读码流→解码→算指标”闭环，输出 json 日志。

import argparse
import concurrent.futures
import io
import json
import multiprocessing
import numpy as np
import os
import time
import torch

from tqdm import tqdm

from src.models.image_model import DMCI
from src.utils.common import ModelStructure, str2bool, create_folder, generate_log_json, \
    get_state_dict, dump_json, set_torch_env
from src.utils.metrics import calc_msssim, calc_msssim_rgb, calc_psnr
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np
from src.utils.video_reader import PNGReader, YUV420Reader
from src.utils.video_writer import PNGWriter, YUV420Writer


def finalize_model(net, device):
    # 测试前的最终化：转 FP16（half）、搬到设备、切换 channels_last 内存格式（提速）
    net = net.half().to(device)
    return net.to(memory_format=torch.channels_last)


def get_distortion(args, x_hat, y, u, v, rgb):
    # 计算一帧的失真指标：YUV420 输入时算 Y/U/V 分量 PSNR 与 MS-SSIM
    # （按 6:1:1 加权），RGB(PNG) 输入时转回 RGB 后计算。返回 (psnr 列表, ssim 列表)
    x_hat = x_hat + 0.5
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


def get_src_frame(args, src_reader, device, maximum_read, is_intra, np_only=False):
    # 从源读取器读 maximum_read 帧（HT 模型为 8 帧一个 chunk，I 帧为 1 帧），
    # 转成模型输入格式：YUV420→444 或 PNG→YCbCr，中心化 -0.5，沿通道拼成
    # [B, 3*N, H, W] 的 FP16 张量。chunk 不足 8 帧时复制末帧补齐。
    # np_only=True 时只返回 numpy 原帧（解码侧对齐源帧用）
    if args['src_type'] == 'yuv420':
        processed = 0
        x, y, u, v, rgb = [], [], [], [], []
        while processed < maximum_read:
            curr_y, curr_uv = src_reader.read_one_frame()
            curr_yuv = ycbcr420_to_444_np(curr_y, curr_uv)
            curr_x = torch.from_numpy(curr_yuv).unsqueeze(0)
            curr_y = curr_y[0, :, :]
            curr_u = curr_uv[0, :, :]
            curr_v = curr_uv[1, :, :]
            curr_rgb = None
            x.append(curr_x)
            y.append(curr_y)
            u.append(curr_u)
            v.append(curr_v)
            rgb.append(curr_rgb)
            processed += 1
    else:
        assert args['src_type'] == 'png'
        processed = 0
        x, y, u, v, rgb = [], [], [], [], []
        while processed < maximum_read:
            curr_rgb = src_reader.read_one_frame()
            curr_x = torch.from_numpy(curr_rgb).unsqueeze(0).to(device=device, non_blocking=True)
            curr_x = curr_x.float() / 255.0
            curr_x = rgb2ycbcr(curr_x)
            curr_y, curr_u, curr_v = None, None, None
            x.append(curr_x)
            y.append(curr_y)
            u.append(curr_u)
            v.append(curr_v)
            rgb.append(curr_rgb)
            processed += 1

    # P chunk 不足 g_frame_delay 帧时复制最后一帧补满（末 chunk 处理）
    while not is_intra and processed < g_frame_delay:
        x.append(x[-1])
        y.append(y[-1])
        u.append(u[-1])
        v.append(v[-1])
        rgb.append(rgb[-1])
        processed += 1

    if np_only:
        return y, u, v, rgb

    # 沿通道拼接为 chunk 张量并做数值/格式准备
    x = torch.cat(x, dim=1)
    if args['src_type'] == 'yuv420':
        x = x.to(device=device, non_blocking=True)
    x = x.half()
    if args['src_type'] == 'yuv420':
        x = x / 255.0
    x = x - 0.5
    x = x.to(memory_format=torch.channels_last)
    return x, y, u, v, rgb


def get_src_reader(args):
    # 按输入类型构造读取器：PNG 序列（RGB）或裸 YUV420 文件
    if args['src_type'] == 'png':
        src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])
    elif args['src_type'] == 'yuv420':
        src_reader = YUV420Reader(args['src_path'], args['src_width'], args['src_height'])
    else:
        assert False
    return src_reader


def parse_args():
    # 命令行参数：模型路径、QP、测试配置 json、模型结构（htl/hts/ld）、
    # intra period、reset_interval（记忆重置周期）、并行 worker 数等
    parser = argparse.ArgumentParser()

    parser.add_argument('--skip_thres', type=float, default=0)
    parser.add_argument('--model_path_i', type=str)
    parser.add_argument('--model_path_p',  type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--qp_i', type=int, nargs='+')
    parser.add_argument('--qp_p', type=int, nargs='+')
    parser.add_argument('--force_intra', type=str2bool, default=False)
    parser.add_argument('--force_frame_num', type=int, default=-1)
    parser.add_argument('--force_intra_period', type=int, default=-1)
    parser.add_argument('--reset_interval', type=int, default=32)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument('--worker', '-w', type=int, default=1, help='worker number')
    parser.add_argument('--model_structure', type=str, default='ld', choices=['htl', 'hts', 'ld'])
    parser.add_argument('--cuda_idx', type=int, nargs='+', help='GPU indexes to use')
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default='out_bin')
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


@torch.inference_mode()
def run_one_point_with_stream(p_net, i_net, args):
    # ★ 核心测试函数：一个序列 × 一个码率点的完整闭环 ★
    # 阶段一（编码）：按 chunk 读源 → I 帧用 i_net.compress / P chunk 用 p_net.compress
    #   → write_sps/write_ip 封装 NAL 写入内存码流 → 落盘 .bin；
    #   I 帧编码后调用 p_net.add_ref_feature_from_frame 建立 P 模型初始参考特征。
    # 阶段二（解码）：重新读 .bin → read_header/read_ip_remaining 解析 NAL →
    #   i_net.decompress / p_net.decompress 重建 → 逐帧算 PSNR/MS-SSIM →
    #   （可选）写重建帧 → 汇总 json 日志。
    # 同时用 CUDA Event 统计每个 chunk 的编/解码耗时（跳过前 4 个 chunk 热身）。
    if args['check_existing'] and os.path.exists(args['curr_json_path']) and \
            os.path.exists(args['curr_bin_path']):
        with open(args['curr_json_path']) as f:
            log_result = json.load(f)
            if log_result['i_frame_num'] + log_result['p_frame_num'] == args['frame_num']:
                return log_result
            print(f'incorrect log for {args['curr_json_path']}, try to rerun.')

    frame_num = args['frame_num']
    save_decoded_frame = args['save_decoded_frame']
    verbose = args['verbose']
    reset_interval = args['reset_interval']
    intra_period = args['intra_period']
    verbose_json = args['verbose_json']
    device = next(i_net.parameters()).device

    src_reader = get_src_reader(args)
    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)

    frame_types = []
    psnrs = []
    msssims = []
    bits = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_time = time.time()
    encoding_time = []
    decoding_time = []

    output_buff = io.BytesIO()
    sps_helper = SPSHelper()

    frame_idx = 0
    while frame_idx < frame_num:
        # 判断当前 chunk 是否为 I 帧：首帧、或按 intra_period 周期刷新
        is_intra = False
        if frame_idx == 0 or intra_period == 1:
            is_intra = True
        if intra_period > 1 and frame_idx != 1:
            assert intra_period % g_frame_delay == 0
            if frame_idx % intra_period == 1:
                is_intra = True

        maximum_read = min(g_frame_delay, frame_num - frame_idx)  # 本 chunk 帧数
        if is_intra:
            maximum_read = 1

        x, y, u, v, rgb = get_src_frame(args, src_reader, device, maximum_read, is_intra)

        torch.cuda.synchronize(device=device)
        start_event.record()

        if is_intra:
            # ---- I 帧：用帧内模型 DMCI 压缩，并把重建帧设为 P 模型的初始参考 ----
            curr_qp = args['qp_i']
            reset_feature_memory = 0
            encoded = i_net.compress(x, curr_qp, padding_b, padding_r)
            if not args['force_intra']:
                p_net.clear_dpb()
                p_net.add_ref_feature_from_frame(encoded['x_hat'])
            frame_types.append(0)
        else:
            # ---- P chunk：判断是否触发周期性记忆重置（默认每 32 帧）----
            if reset_interval > 0 and (frame_idx + g_frame_delay) % reset_interval == 1:
                reset_feature_memory = 1
            else:
                reset_feature_memory = 0
            curr_qp = args['qp_p']

            encoded = p_net.compress(x, curr_qp, reset_feature_memory, padding_b, padding_r)
            for _ in range(maximum_read):
                frame_types.append(1)

        # ---- SPS 去重 + NAL 封装写入码流 ----
        sps = {
            'sps_id': -1,
            'height': pic_height,
            'width': pic_width,
        }
        sps_id, sps_new = sps_helper.get_sps_id(sps)
        sps['sps_id'] = sps_id
        sps_bytes = 0
        if sps_new:
            sps_bytes = write_sps(output_buff, sps)
            if verbose >= 2:
                print('new sps', sps)
        stream_bytes = write_ip(output_buff, is_intra, sps_id, curr_qp,
                                encoded['ec_parallel'], reset_feature_memory,
                                encoded['bit_stream'])
        bits.append(stream_bytes * 8 + sps_bytes * 8)
        for _ in range(1, maximum_read):
            bits.append(0)

        end_event.record()
        torch.cuda.synchronize(device=device)

        frame_time = start_event.elapsed_time(end_event) / 1000.0
        encoding_time.append(frame_time)

        if verbose >= 2:
            print(f'frame {frame_idx} encoded, {frame_time * 1000:.3f} ms, '
                  f'bits: {bits[-maximum_read]}')

        frame_idx += maximum_read

    src_reader.close()
    # ---- 编码码流落盘 ----
    with open(args['curr_bin_path'], 'wb') as output_file:
        bytes_buffer = output_buff.getbuffer()
        output_file.write(bytes_buffer)
        total_bytes = bytes_buffer.nbytes
        bytes_buffer.release()
    total_kbps = int(total_bytes * 8 / (frame_num / 30) / 1000)  # assume 30 fps
    output_buff.close()
    sps_helper = SPSHelper()
    with open(args['curr_bin_path'], 'rb') as input_file:
        input_buff = io.BytesIO(input_file.read())
    decoded_frame_number = 0
    src_reader = get_src_reader(args)

    if save_decoded_frame:
        if args['src_type'] == 'png':
            recon_writer = PNGWriter(args['bin_folder'], args['src_width'], args['src_height'])
        elif args['src_type'] == 'yuv420':
            output_yuv_path = args['curr_rec_path'].replace('.yuv', f'_{total_kbps}kbps.yuv')
            recon_writer = YUV420Writer(output_yuv_path, args['src_width'], args['src_height'])

    while decoded_frame_number < frame_num:
        # ---- 解码循环：逐 NAL 解析并重建 ----
        torch.cuda.synchronize(device=device)
        start_event.record()

        header = read_header(input_buff)  # 读 NAL 头
        while header['nal_type'] == NalType.NAL_SPS:  # 先消费所有 SPS NAL
            sps = read_sps_remaining(input_buff, header['sps_id'])
            sps_helper.add_sps_by_id(sps)
            if verbose >= 2:
                print('new sps', sps)
            header = read_header(input_buff)
            continue
        sps_id = header['sps_id']

        sps = sps_helper.get_sps_by_id(sps_id)
        qp, ec_part, reset_feature_memory, bit_stream = read_ip_remaining(input_buff)

        if header['nal_type'] == NalType.NAL_I:
            # I 帧：帧内解压，并重建 P 模型初始参考特征
            decoded = i_net.decompress(bit_stream, sps, qp, ec_part)
            if not args['force_intra']:
                p_net.clear_dpb()
                p_net.add_ref_feature_from_frame(decoded['x_hat'], apply_feature_adaptor=False)
        elif header['nal_type'] == NalType.NAL_P:
            # P chunk：一次解出 8 帧（HT）或 1 帧（LD）
            decoded = p_net.decompress(bit_stream, sps, qp, ec_part, reset_feature_memory)

        recon_frame = decoded['x_hat']

        end_event.record()
        torch.cuda.synchronize(device=device)

        frame_time = start_event.elapsed_time(end_event) / 1000.0
        decoding_time.append(frame_time)

        # ---- 逐帧计算失真（裁掉 padding）----
        is_intra = header['nal_type'] == NalType.NAL_I
        maximum_read = min(g_frame_delay, frame_num - decoded_frame_number)
        if is_intra:
            maximum_read = 1
        y, u, v, rgb = get_src_frame(args, src_reader, device, maximum_read, is_intra, np_only=True)

        for i in range(maximum_read):
            if isinstance(recon_frame, list):
                x_hat = recon_frame[i]
            else:
                x_hat = recon_frame
            x_hat = x_hat[:, :, :pic_height, :pic_width]
            curr_psnr, curr_ssim = get_distortion(args, x_hat, y[i], u[i], v[i], rgb[i])
            psnrs.append(curr_psnr)
            msssims.append(curr_ssim)

        if verbose >= 2:
            stream_length = 0 if bit_stream is None else len(bit_stream) * 8
            print(f'frame {decoded_frame_number} decoded, {frame_time * 1000:.3f} ms, '
                  f'bits: {stream_length}, ', end='')
            for i in range(-maximum_read, 0, 1):
                print(f'PSNR: {psnrs[i][0]:.4f}, ', end='')
            print()

        if save_decoded_frame:
            # 可选：把重建帧写回 PNG/YUV 文件
            for i in range(maximum_read):
                if isinstance(recon_frame, list):
                    x_hat = recon_frame[i]
                else:
                    x_hat = recon_frame
                x_hat = x_hat[:, :, :pic_height, :pic_width]
                if args['src_type'] == 'yuv420':
                    y_rec, uv_rec = yuv_444_to_420(x_hat + 0.5)
                    y_rec = torch.clamp(y_rec * 255, 0, 255).round().byte()
                    y_rec = y_rec.squeeze(0).cpu().numpy()
                    uv_rec = torch.clamp(uv_rec * 255, 0, 255).byte()
                    uv_rec = uv_rec.squeeze(0).cpu().numpy()
                    recon_writer.write_one_frame(y_rec, uv_rec)
                else:
                    assert args['src_type'] == 'png'
                    rgb_rec = ycbcr2rgb(x_hat + 0.5)
                    rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).round().byte()
                    rgb_rec = rgb_rec.squeeze(0).cpu().numpy()
                    recon_writer.write_one_frame(rgb_rec)
        decoded_frame_number += maximum_read
    input_buff.close()
    src_reader.close()

    if save_decoded_frame:
        recon_writer.close()

    # ---- 统计平均编/解码耗时（跳过前 4 个 chunk 作为 GPU 热身）----
    test_time = time.time() - start_time
    test_time_frame_numuber = len(encoding_time)
    time_bypass_frame_num = 4  # bypass the first 4 * g_frame_delay frames as warmup
    if verbose >= 1 and test_time_frame_numuber > time_bypass_frame_num:
        encoding_time = encoding_time[time_bypass_frame_num:]
        decoding_time = decoding_time[time_bypass_frame_num:]
        avg_encoding_time = sum(encoding_time)/len(encoding_time)
        avg_decoding_time = sum(decoding_time)/len(decoding_time)
        print(f'encoding/decoding {len(encoding_time)} * {g_frame_delay} frames, '
              f'average encoding time {avg_encoding_time * 1000:.3f} ms, '
              f'average decoding time {avg_decoding_time * 1000:.3f} ms.')
    else:
        avg_encoding_time = None
        avg_decoding_time = None

    # 汇总日志 json 并写盘
    log_result = generate_log_json(frame_num, pic_height * pic_width, test_time,
                                   frame_types, bits, psnrs, msssims, verbose=verbose_json,
                                   avg_encoding_time=avg_encoding_time,
                                   avg_decoding_time=avg_decoding_time,)
    with open(args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)
    return log_result


g_i_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess
g_p_net = None
g_frame_delay = None  # depending on the model_structure (1 or 8)


def init_func(args, gpu_num):
    # ProcessPoolExecutor 的进程初始化器：每个 worker 进程绑定一张 GPU，
    # 加载 I/P 两个模型权重并 update()（构建 CDF），转 FP16 + channels_last。
    # 另外设置非默认 CUDA 流（cuda graph 不支持默认流 0）
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
        device = 'cuda:0'
    else:
        device = 'cpu'

    # one-time setup of default cuda stream. cuda graph does not support stream 0
    custom_stream = torch.cuda.Stream(device, 0)
    torch.cuda.set_stream(custom_stream)

    # 按模型结构导入对应 DMC（LD：N=1；HTL/HTS：N=8）
    model_structure = ModelStructure(args.model_structure)

    if model_structure == ModelStructure.LD:
        from src.models.video_model_ld import DMC, g_frame_delay as frame_delay
    else:
        from src.models.video_model_ht import DMC, g_frame_delay as frame_delay

    global g_i_net
    g_i_net = DMCI()          # 帧内模型
    g_i_net = g_i_net.eval()
    i_state_dict = get_state_dict(args.model_path_i)
    g_i_net.load_state_dict(i_state_dict)
    g_i_net.update(args.skip_thres)  # 构建 rANS CDF 表
    g_i_net = finalize_model(g_i_net, device)

    if not args.force_intra:
        # 帧间模型（纯图像测试 force_intra 时不加载）
        global g_p_net
        if model_structure == ModelStructure.LD:
            g_p_net = DMC()
        else:
            g_p_net = DMC(model_structure=model_structure)
        g_p_net = g_p_net.eval()
        p_state_dict = get_state_dict(args.model_path_p)
        g_p_net.load_state_dict(p_state_dict)
        g_p_net.update(args.skip_thres)
        g_p_net = finalize_model(g_p_net, device)

    global g_frame_delay
    g_frame_delay = frame_delay


def worker(args):
    # 进程池任务单元：为一个序列 × 一个码率点组织路径参数，
    # 调 run_one_point_with_stream 执行完整测试并回传结果
    sub_dir_name = args['seq']
    bin_folder = os.path.join(args['stream_path'], args['ds_name'])
    create_folder(bin_folder, True)

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['curr_bin_path'] = os.path.join(bin_folder, f'{args['seq']}_q{args['qp_i']}.bin')
    args['curr_rec_path'] = args['curr_bin_path'].replace('.bin', '.mp4')
    args['curr_json_path'] = args['curr_bin_path'].replace('.bin', '.json')

    result = run_one_point_with_stream(g_p_net, g_i_net, args)

    result['ds_name'] = args['ds_name']
    result['seq'] = args['seq']
    result['rate_idx'] = args['rate_idx']
    return result


def main():
    # 主入口：解析参数 → 读取测试配置 json → 按 (序列 × 码率点) 提交进程池任务
    # → 汇总所有结果写出到 output_path 的 json
    begin_time = time.time()

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('test_video.py requires CUDA (torch.cuda.is_available() is False)')

    args.skip_thres = max(0, args.skip_thres)
    if args.cuda_idx is not None:
        cuda_device = ','.join([str(s) for s in args.cuda_idx])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    gpu_num = torch.cuda.device_count()
    multiprocessing.set_start_method('spawn')
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num,
                                                                 initializer=init_func,
                                                                 initargs=(args, gpu_num))
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    # 生成 rate_num 个测试 QP（I 帧）：未指定时在 [0, 63] 均匀取点
    qp_i = []
    if args.qp_i is not None:
        assert len(args.qp_i) == rate_num
        qp_i = args.qp_i
    else:
        assert 2 <= rate_num <= DMCI.qp_num()
        for i in np.linspace(0, DMCI.qp_num() - 1, num=rate_num):
            qp_i.append(int(i+0.5))

    if not args.force_intra:
        if args.qp_p is not None:
            assert len(args.qp_p) == rate_num
            qp_p = args.qp_p
        else:
            qp_p = qp_i

    print(f'testing {rate_num} rates, using qp: {', '.join(str(q) for q in qp_i)}')

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

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}

    for res in results:
        log_result[res['ds_name']][res['seq']][f'{res['rate_idx']:03d}'] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == '__main__':
    main()
