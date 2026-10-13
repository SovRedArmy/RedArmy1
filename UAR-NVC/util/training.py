import os
import time
import math
import torch

from torch          import  optim
from tqdm           import  tqdm
from tensorboardX   import  SummaryWriter

from util.metrics       import  metrics, RateDistortionLoss
from util.utils         import  gpu_time_logger, get_dir_index

from util.dataloader            import parse_input

from compression.quant_utils import get_model_probs
import compression

class Tester:
    def __init__(self, net, global_cfg, local_cfg):
        super(Tester, self).__init__()
        self.net            = net
        train_cfg           = global_cfg.train_cfg
        self.model_name     = global_cfg.model_cfg.model
        
        self.device         = train_cfg.device
        self.metrics_set    = train_cfg.metrics_set
        
        self.val_loader     = local_cfg.val_loader
    
        self.data_length    = self.val_loader.dataset.data_length()

        self.metrics        = metrics(self.metrics_set)
        self.criterion      = RateDistortionLoss(num_pixels=self.val_loader.dataset.pixel_all(),
                                                 Lambda=global_cfg.train_cfg.Lambda)
        
    def network_forward(self, sig_batch_all, embedding=None, training=True):
        data_loader = self.train_loader if training else self.val_loader
        if self.model_name in ['NeRV', 'ENeRV', 'HNeRV']:
            sig_batch       = sig_batch_all['img'].to(self.device)
            idx             = sig_batch_all['idx'].to(self.device)
            norm_idx        = sig_batch_all['norm_idx'].to(self.device)
            if self.model_name == "HNeRV":
                result          = self.net(sig_batch, input_embed=embedding)
            elif self.model_name in ['NeRV', 'ENeRV']:
                result          = self.net(norm_idx)
        elif self.model_name == 'HiNeRV':
            inputs, sig_batch = parse_input(data_loader, sig_batch_all)
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            sig_batch = sig_batch.to(self.device)
            
            dec_start = time.time()
            img_out          = self.net(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dec_time = time.time() - dec_start
            
            result = {
                'x_hat'         : img_out,
                'dec_time'      : dec_time,
                'likelihoods'   : [],
                'side_loss'     : 0,
            }
        else:
            NotImplementedError, "error model name!"
        result['x_hat'] = result['x_hat'].clamp(0, 1)
        result['likelihoods'] = get_model_probs(self.net)
        out_criterion   = self.criterion(result, sig_batch)
        return result, out_criterion
    
    @torch.no_grad()
    def eval_only(self, embedding=None, return_img=False):
        self.metrics.reset()
        self.net.eval()
        embedding_list  = []
        dec_time_list   = []
        x_hat_list      = []
        for step, sig_batch_all in enumerate(self.val_loader):
            result, out_criterion = self.network_forward(sig_batch_all, 
                                    embedding=embedding[step:step+1] if embedding is not None else None, training=False)
            # print(out_criterion['psnr'])
            self.metrics.update(out_criterion)
            
            if self.model_name == 'HNeRV':
                embedding_list.append(result['embed_list'][0])
            dec_time_list.append(result['dec_time'])
            if return_img:
                x_hat_list.append(result['x_hat'])
        self.metrics.metrics_round(3)
        
        # print('')
        return {
            'metrics'   : self.metrics.stat_dict(stat='before quant'),
            'embedding' : torch.concat(embedding_list, dim=0) if self.model_name == 'HNeRV' else None,
            'dec_time'  : dec_time_list,
            'x_hat_list': x_hat_list,
        }
    
    def renew_model(self, model):
        self.metrics.reset()
        self.net = model

class Trainer:
    def __init__(self, accelerator, net, global_cfg, local_cfg, lr=None, epoch=None):
        super(Trainer, self).__init__()
        self.net            = net
        self.accelerator    = accelerator
        train_cfg           = global_cfg.train_cfg
        self.model_name     = global_cfg.model_cfg.model
        
        self.device         = train_cfg.device
        self.train_iters    = train_cfg.train_iters
        self.local_pbar     = train_cfg.local_pbar
        self.metrics_set    = train_cfg.metrics_set
        self.lr             = train_cfg.lr if (lr == None) else lr
        
        self.logger         = local_cfg.local_logger
        self.train_loader   = local_cfg.train_loader
        self.val_loader     = local_cfg.val_loader
        self.model_path     = local_cfg.model_path
        self.info_path      = local_cfg.info_path
        self.model_idx      = local_cfg.model_idx
    
        self.data_length    = self.train_loader.dataset.data_length()
    
        self.epochs         = local_cfg.epochs if (epoch == None) else epoch
        self.best_monitor   = 1e5
        self.iters          = 1

        self.metrics        = metrics(self.metrics_set)
        # self.trainwriter    = SummaryWriter('{}/{}'.format(local_cfg.log_dir, 'Train'))
        # self.evalwriter     = SummaryWriter('{}/{}'.format(local_cfg.log_dir, 'Eval'))

        self.criterion      = RateDistortionLoss(num_pixels=self.train_loader.dataset.pixel_all(),
                                                 Lambda=global_cfg.train_cfg.Lambda)
        self.optimizer      = optim.Adam(self.net.parameters(), lr=train_cfg.lr)
        self.lr_step_cnt    = 1
        self.ckpt_cnt       = 1
        self.lr_scheduler   = HNeRV_lr_scheduler(self.optimizer, lr=self.lr,
                                                 all_epoch=self.epochs * global_cfg.train_cfg.data_multiple)
        self.optimizer      = accelerator.prepare(self.optimizer)
        self.lr_scheduler   = accelerator.prepare(self.lr_scheduler)
        

    def loop(self):
        for self.iters in range(self.iters, self.epochs + 1):
            self.train_step()
            self.eval_step()
            
    def network_forward(self, sig_batch_all, training=True):
        data_loader = self.train_loader if training else self.val_loader
        if self.model_name in ['NeRV', 'ENeRV', 'HNeRV']:
            sig_batch       = sig_batch_all['img']
            idx             = sig_batch_all['idx']
            norm_idx        = sig_batch_all['norm_idx']
            if self.model_name == 'HNeRV':
                result          = self.net(sig_batch)
            elif self.model_name in ['NeRV', 'ENeRV']:
                result          = self.net(norm_idx)
                
        elif self.model_name == 'HiNeRV':
            inputs, sig_batch = parse_input(data_loader, sig_batch_all)
            img_out          = self.net(inputs)
            result = {
                'x_hat'         : img_out,
            }
        else:
            NotImplementedError, "error model name!"
        result['x_hat'] = result['x_hat'].clamp(0, 1)
        result['likelihoods'] = get_model_probs(self.net)
        out_criterion   = self.criterion(result, sig_batch)
        return out_criterion
        

    def train_step(self):
        self.metrics.reset()
        self.lr_step_cnt    = 1
        self.net.train()
        time_start = time.time()
        self.logger.info(f"Starting training epoch {self.iters}:")
        
        if self.local_pbar and self.accelerator.is_main_process:
            pbar = tqdm(total         = len(self.train_loader),
                        desc          = f'Epoch{self.iters}/{self.epochs}',
                        mininterval   = 0.5)
        step = 0
        for sig_batch_all in self.train_loader:
        # for step, sig_batch_all in enumerate(self.train_loader):
            aaaa = len(self.train_loader)
            if step > self.train_iters:
                break
            self.optimizer.zero_grad()
            
            out_criterion = self.network_forward(sig_batch_all)
            
            # out_criterion["loss"].backward()
            self.accelerator.backward(out_criterion["loss"])
            if self.accelerator.sync_gradients:
                self.accelerator.unscale_gradients(self.optimizer)
            torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm=1.0, norm_type=2.0)
            # self._print_grad()
            self.optimizer.step()
            
            self.metrics.update(out_criterion)
            
            # learning rate delay
            step += 1
            if step * self.train_loader.total_batch_size >= self.lr_step_cnt * self.data_length:
                self.lr_step_cnt += 1
                self.lr_scheduler.step(self.metrics.get_loss())
                
            if self.local_pbar and self.accelerator.is_main_process:
                    pbar.set_postfix(self.metrics.stat_dict(stat='train'))
                    pbar.update(1)

        self.metrics.metrics_round(3)
        self.logger.info('====> Epoch: {} Time: {:.2f} '.format(self.iters, time.time() - time_start) + \
                              str(self.metrics.stat_dict(stat='train')).strip("{}").replace("'", '') + \
                              ' lr: {:.6f}'.format(self.optimizer.param_groups[0]['lr'])
                              )
        # self.metrics.tensorboard_renew(self.trainwriter, self.iters)

    @torch.no_grad()
    def eval_step(self):
        self.metrics.reset()
        self.net.eval()
        time_start = time.time()
        self.logger.info(f"Starting test epoch {self.iters}:")
        
        if self.local_pbar and self.accelerator.is_main_process:
            pbar = tqdm(total         = len(self.val_loader),
                        desc          = f'Epoch{self.iters}/{self.epochs}',
                        mininterval   = 0.5,
                        colour        = 'BLUE')
            
        for step, sig_batch_all in enumerate(self.val_loader):
            out_criterion = self.network_forward(sig_batch_all, training=False)
            self.metrics.update(out_criterion)
            
            if self.local_pbar and self.accelerator.is_main_process:
                pbar.set_postfix(self.metrics.stat_dict(stat='test'))
                pbar.update(1)

        self.metrics.metrics_round(3)
        self.logger.info('====> Epoch: {} Time: {:.2f} '.format(self.iters, time.time() - time_start) + \
                              str(self.metrics.stat_dict(stat='test')).strip("{}").replace("'", '')
                              )
        # self.metrics.tensorboard_renew(self.evalwriter, self.iters)

        torch.save(compression.utils.unwrap_model(self.net).state_dict(), self.model_path)

    @torch.no_grad()
    def eval_only(self):
        self.metrics.reset()
        self.net.eval()
        for step, sig_batch_all in enumerate(self.val_loader):
            out_criterion = self.network_forward(sig_batch_all, training=False)
            self.metrics.update(out_criterion)
        self.metrics.metrics_round(3)
        
    # show each network parameter
    def _print_grad(self):
        dict0 = {}
        for name, param in self.net.named_parameters():
            dict0[name] = param.grad
            print(f"Gradient of {name}:")
            print(param.grad)
            print("\n")
            
class HNeRV_lr_scheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, lr, all_epoch, up_ratio=0.1, up_pow=1, min_lr=0.1) -> None:
        self.optimizer  = optimizer
        self.lr         = lr
        self.up_ratio   = up_ratio
        self.up_pow     = up_pow
        self.min_lr     = min_lr
        self.add_epoch  = 0
        self.all_epoch  = all_epoch

    def step(self, loss=None):
        cur_epoch   = self.add_epoch / self.all_epoch
        if cur_epoch < self.up_ratio:
            lr_mult = self.min_lr + (1. - self.min_lr) * (cur_epoch / self.up_ratio)** self.up_pow
        else:
            lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - self.up_ratio)/ (1 - self.up_ratio)) + 1.0)

        a = self.optimizer.param_groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.lr * lr_mult
            
        self.add_epoch += 1
        return self.lr * lr_mult
        