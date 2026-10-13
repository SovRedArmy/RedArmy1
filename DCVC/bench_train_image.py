# 训练速度测量：图像模型 DMCI，直接复用 train_image.train_one_epoch 的真实训练路径。
# 用法: python bench_train_image.py <epoch_idx>   （epoch 决定策略表中的 patch/lr 档位）
import sys
import time
import torch

sys.path.insert(0, '.')
import train_image as ti
from src.datasets.image_dataset import ImageFolder
from src.models.image_model import DMCI
from src.utils.common import get_training_lambdas, get_dataloader

epoch = int(sys.argv[1])
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16

torch.manual_seed(0)
dataset = ImageFolder('test_data/train_image', 256, 256, DMCI.qp_num(),
                      get_training_lambdas([0.0023, 0.16], DMCI.qp_num()))
loader = get_dataloader(dataset, -1, 1, batch_size, 4)

i_net = DMCI().cuda()
optimizer = torch.optim.AdamW(i_net.parameters(), lr=1e-4)

torch.cuda.synchronize()
t0 = time.time()
ti.train_one_epoch(i_net, loader, optimizer, epoch, -1)
torch.cuda.synchronize()
elapsed = time.time() - t0

iters = len(loader)
strategy = ti.get_training_strategy()[min(epoch, len(ti.get_training_strategy()) - 1)]
_, lr, pw, ph = strategy
print(f'[bench] model=DMCI epoch={epoch} patch={pw}x{ph} batch={batch_size} '
      f'iters={iters} total={elapsed:.1f}s per_iter={elapsed/iters:.3f}s '
      f'max_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GiB')
