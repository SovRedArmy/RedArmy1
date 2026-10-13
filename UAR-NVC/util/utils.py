import os
import os.path
import random
import torch
import logging

import datetime
import uuid

import numpy    as np

from typing     import Any

def tree(logger, ele, deep=0):
    if isinstance(ele, dict):
        for key, value in ele.items():
            if isinstance(value, dict):
                logger.info('    |'*deep+'---'+key)
                tree(logger, value, deep+1)
            else:
                logger.info('    |'*deep+'---'+key + ': ' + str(value))
    else:
        logger.info('    |'*deep+'---'+str(ele))
        
def log_in_config(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 40)
    tree(logger, cfg)
    logger.info('=' * 80)


def create_logger(filename, file_handle=True, cmd_handle=True, rewirte=False):
    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    
    if cmd_handle:
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(stream_formatter)
        logger.addHandler(ch)
        
    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='w+' if rewirte else 'a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger


def fix_seed(seed): 
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    
        
def get_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = str(len(dirs))

    return log_dir_index

class Global_log_config:
    def __init__(self, args, train=True):
        self.dataset    = args['dataset_cfg']['dataset']
        self.model      = args['model_cfg']['model_name']
        self.base_dir   = args['log_cfg']['base_dir'] if train else 'inference_result'
        os.makedirs(self.base_dir, exist_ok=True)
        index = get_dir_index(self.base_dir)
        
        self.cfg_dir = self.base_dir + '/' + '-'.join([
            self.dataset,
            self.model,
            'init'      + str(args['train_cfg']['init_type']) + '_' + str(int(args['train_cfg']['ref_compress'])),
            'ms'        + str(args['model_cfg']['modelsize']) + 'MB',
            'gop'       + str(args['dataset_cfg']['gop']),
            'gom'       + str(args['dataset_cfg']['gom']),
            'qmode'     + str(args['compression_cfg']['q_mode']),
            'Lambda'    + str(args['train_cfg']['Lambda']),
            'epoch'     + str(args['train_cfg']['epochs']) + '_' + str(args['train_cfg']['local_epochs']),
            'lr'        + str(args['train_cfg']['lr']),
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            # str(uuid.uuid4()).split('-')[0]
        ])
        
        # self.cfg_dir    = '%s/%s' % (self.base_dir, self.dataset + '_' + self.model + '_' + index)
        if args.train_cfg.checkpoint != '':
            self.cfg_dir = args.train_cfg.checkpoint
        self.log_dir    = '%s/log' % self.cfg_dir
        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def get_dict(self, args):
        args.update({
            'log_cfg':{
                'cfg_dir'       : self.cfg_dir,
                'log_dir'       : self.log_dir,
            }
        })
        return args
        
# More accurate time analysis tool (cosidering GPU)
class gpu_time_logger(object):
    def __init__(self):
        self.start      = torch.cuda.Event(enable_timing=True)
        self.end        = torch.cuda.Event(enable_timing=True)
        self.time_list  = []
        
        self.start.record(stream=torch.cuda.current_stream())

    def renew(self):
        self.end.record(stream=torch.cuda.current_stream())
        self.end.synchronize()
        self.time_list.append(self.start.elapsed_time(self.end))
        
    def get_list(self):
        self.renew()
        return self.time_list
    
    