# 训练速度测量：视频模型 DMC（htl/hts/ld），复用 train_video.train_one_epoch 真实训练路径。
# 用法: python bench_train_video.py <structure> <epoch_idx> <batch_size> [stage0|stage1|stage2|stage3]
import sys
import time
import torch

sys.path.insert(0, '.')
import train_video as tv
from src.datasets.video_dataset import VideoFolder
from src.models.image_model import DMCI
from src.utils.common import ModelStructure, get_training_lambdas, get_dataloader

structure = sys.argv[1]
epoch = int(sys.argv[2])
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
scheduling = sys.argv[4] if len(sys.argv) > 4 else 'stage0'

torch.manual_seed(0)
model_structure = ModelStructure(structure)
if model_structure == ModelStructure.LD:
    from src.models.video_model_ld import DMC, g_frame_delay
else:
    from src.models.video_model_ht import DMC, g_frame_delay

dataset = VideoFolder('test_data/train_video', 256, 256, DMCI.qp_num(),
                      get_training_lambdas([0.0023, 0.16], DMCI.qp_num()),
                      group_of_pictures=g_frame_delay)
loader = get_dataloader(dataset, -1, 1, batch_size, 4)

i_net = DMCI().cuda().eval()
if model_structure == ModelStructure.LD:
    p_net = DMC().cuda()
else:
    p_net = DMC(model_structure=model_structure).cuda()
optimizer = torch.optim.AdamW(p_net.parameters(), lr=1e-4)

args = type('A', (), {})()
args.training_scheduling = scheduling

strategy = tv.get_training_strategy(args.training_scheduling, g_frame_delay, model_structure)
row = strategy[min(epoch, len(strategy) - 1)]
_, lr, seq_len, cascaded, pw, ph = row
chunks_per_iter = (seq_len - 1) // g_frame_delay

torch.cuda.synchronize()
t0 = time.time()
tv.train_one_epoch(p_net, i_net, args, loader, optimizer, epoch, -1,
                   g_frame_delay, model_structure)
torch.cuda.synchronize()
elapsed = time.time() - t0

iters = len(loader)
if iters:
    print(f'[bench] model=DMC-{structure} epoch={epoch} seq_len={seq_len} cascade={int(cascaded)} '
          f'patch={pw}x{ph} batch={batch_size} ckpt={int(cascaded and pw > 256)} '
          f'iters={iters} total={elapsed:.1f}s per_iter={elapsed/iters:.2f}s '
          f'per_chunk_update={elapsed/iters/chunks_per_iter:.2f}s '
          f'max_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GiB')
