import os
import math
import torch
import torch.nn                 as  nn
import torch.nn.functional      as  F
import torchvision.transforms   as  transforms
import numpy                    as  np

from PIL                    import  Image
from torch.utils.data       import  Dataset


class GlobalDataset:
    def __init__(self, main_dir, gop=3):
        self.gop        = gop
        self.main_dir   = main_dir
        self.all_imgs   = os.listdir(main_dir)
        self.all_imgs.sort()
        
        self.t_length   = len(self.all_imgs)
        
    def get_len(self):
        return self.t_length
        
    def get_local_dataset(self, index=0):
        max_idx = min((index + 1 ) * self.gop, len(self.all_imgs))
        return self.all_imgs[index * self.gop: max_idx]
    
class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    # Return actual data length
    def data_length(self):
        raise NotImplementedError("This inr compression framework needs data_length in training process for lr changing")

class CustomDataset(BaseDataset):
    def __init__(self, img_transforms, args, local_cfg, train=True):
        self.train          = train
        self.args           = args
        self.main_dir       = args['dataset_cfg']['data_path']
        self.patch_size     = args['dataset_cfg']['patch_size']
        self.gop            = args['dataset_cfg']['gop']
        self.data_multiple  = args['train_cfg']['data_multiple']
        self.local_dataset  = local_cfg['local_dataset']
        self.img_transforms = img_transforms
        self.video_size     = list([int(x) for x in args['dataset_cfg']['video_size'].split('_')])
        
        error_str = 'Height: {} or width: {} is not divisible by patch_size: {}'.format(self.video_size[-2], self.video_size[-1], self.patch_size)
        assert (self.video_size[-2] % self.patch_size == 0 and self.video_size[-1] % self.patch_size == 0), error_str
        self.t_length       = len(self.local_dataset)
        self.h_length       = self.video_size[-2] // self.patch_size
        self.w_length       = self.video_size[-1] // self.patch_size
                                            
        self.data_buffer    = None
        self.idx_buffer     = None
        self._data_prepare() 
    
    def _data_prepare(self):
        image_list = []
        for idx in range(len(self.local_dataset)):
            img_path        = os.path.join(self.main_dir, self.local_dataset[idx])
            image           = Image.open(img_path).convert("RGB")
            tensor_image    = self.img_transforms(image)
            image_list.append(tensor_image.unsqueeze(1))
        
        image_all           = torch.concat(image_list, dim=1) # [C, T, 720, 1280]
        C, T, H, W          = image_all.shape
        image_all           = image_all.reshape(C, T, self.h_length, self.patch_size, self.w_length, self.patch_size)
        image_all           = image_all.permute(1, 2, 4, 0, 3, 5)
        image_all           = image_all.reshape(T * self.h_length * self.w_length, C, self.patch_size, self.patch_size)
        self.data_buffer    = image_all
        
        idx_all             = [torch.tensor([t, x, y], dtype=torch.int32) 
                               for t in range(self.t_length) for x in range(self.h_length) for y in range(self.w_length)]
        self.idx_buffer     = idx_all
            
    def pixel_all(self):
        pixel_all       = len(self.local_dataset) * self.video_size[-2] * self.video_size[-1]
        return pixel_all
    
    def data_length(self):
        length = self.t_length * self.h_length * self.w_length
        return  length
    
    def __len__(self):
        length = self.t_length * self.h_length * self.w_length
        # Multiply dataset to reduce proportion of time spent on epoch preparation
        if self.train:
            length *= self.data_multiple
        else:
            length *= 1
        return  length
    
    def __getitem__(self, idx):
        idx = idx % (self.t_length * self.h_length * self.w_length)
        data = self.data_buffer[idx]
        patch_idx = self.idx_buffer[idx]
        return data, patch_idx


def create_data_loader(args, local_cfg):
    from model.hnerv_models.hnerv_model_all import VideoDataSet
    from .hinerv_dataset import VideoDataset as VD2
    img_transforms  = transforms.ToTensor()
    if args['model_cfg']['model_name'] in ['NeRV', 'ENeRV', 'HNeRV']:
        train_dataset   = VideoDataSet(args=args, local_cfg=local_cfg)
        val_dataset     = VideoDataSet(args=args, local_cfg=local_cfg, train=False)
    else:
        train_dataset   = VD2(args, local_cfg['local_logger'], root=args['dataset_cfg']['data_path'], local_dataset=local_cfg['local_dataset'], 
                              name=None, crop=args['dataset_cfg']['crop_size'], resize=args['dataset_cfg']['input_size'],
                              patch_size=args['dataset_cfg']['patch_size'],
                              cached=args['dataset_cfg']['cached'], train=True)
        val_dataset     = VD2(args, local_cfg['local_logger'], root=args['dataset_cfg']['data_path'], local_dataset=local_cfg['local_dataset'], 
                              name=None, crop=args['dataset_cfg']['crop_size'], resize=args['dataset_cfg']['input_size'],
                              patch_size=[args['dataset_cfg']['patch_size'][0], -1, -1],
                              cached=args['dataset_cfg']['cached'], train=False)
        # train_dataset   = CustomDataset(img_transforms=img_transforms, args=args, local_cfg=local_cfg)
        # val_dataset     = CustomDataset(img_transforms=img_transforms, args=args, local_cfg=local_cfg, train=False)
        # test_dataset = CustomDataset(main_dir=args.data_path, img_transforms=img_transforms)
    
    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=args['train_cfg']['batchsize'],
                                                shuffle=True, num_workers=args['train_cfg']['num_workers'],
                                                pin_memory=True,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                shuffle=False, num_workers=args['train_cfg']['num_workers'],
                                                pin_memory=True,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    
    return train_loader, val_loader
    

def parse_input(loader, batch, training=True):
    """
    Parse the input to the model during training/evaluation step
    """
    idx, x = batch
    assert idx.ndim == 2, 'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
    assert x.ndim == 5,  'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

    input = {
        'x': x if training else None,
        'idx': idx, 
        'idx_max': loader.dataset.num_patches,
        'batch_size': x.shape[0] * loader.dataset.patch_size[0] / (loader.dataset.num_patches[1] * loader.dataset.num_patches[2]), # in the number of images
        'video_size': loader.dataset.video_size,
        'patch_size': loader.dataset.patch_size
    }

    target = x

    return input, target
    
# def create_dataset(args, logger, training):
#     """
#     Create the dataset instance. Only apply the patch configuration for trainset.
#     """
#     return VideoDataset(logger, root=args.dataset, name=args.dataset_name,
#                         crop=args.crop_size, resize=args.input_size,
#                         patch_size=args.patch_size if training else [args.patch_size[0], -1, -1],
#                         cached=args.cached)


# def create_loader(args, logger, training, dataset):
#     """
#     Create the dataset loader.
#     """
#     loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=args.eval_batch_size if not training else args.batch_size,
#         shuffle=training,
#         num_workers=args.workers,
#         pin_memory=args.pin_mem,
#         persistent_workers=True,
#         drop_last=True,
#         prefetch_factor=2
#     )
#     return loader

    