import torch, os, sys
from datetime import datetime
from loguru import logger

class Arguments():
    def __init__(self):  self.file = {}


def config(**kwargs):
    args = Arguments()
    task, model, dataset = kwargs.get('task'), kwargs.get('model'), kwargs.get('dataset')
    plm_dir, data_dir, cache_dir = kwargs.get('plm_dir'), kwargs.get('data_dir'), kwargs.get('cache_dir')

    ## 0. parameters for file
    args.file = {
        'plm_dir': plm_dir, # dir of plm model
        'data_dir': data_dir+f"{task}/", # dir of data
    }
    sys.path.append(args.file['data_dir']) # add data path to current


    ## 1. logging
    if cache_dir is not None: 
        args.file['cache_dir'] = cache_dir+f'{task}_{dataset}/' # dir of cache
        args.file['log'] = f'./logs/{task}_{dataset}/' # dir of log
        args.file['record'] = f'./records/{task}_{dataset}/' # dir of record

        if not os.path.exists(args.file['log']): os.makedirs(args.file['log']) # create log path
        if not os.path.exists(args.file['record']): os.makedirs(args.file['record']) # create record path
        if not os.path.exists(args.file['cache_dir']): os.makedirs(args.file['cache_dir']) # create cache path

        logger.remove() # do not show log in console
        logDir = os.path.expanduser(args.file['log']+datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(logDir): os.makedirs(logDir) # create log dir

        logger.add(os.path.join(logDir,'params.log'), filter=lambda record: record["extra"].get("name")=="params") # show params
        logger.add(os.path.join(logDir,'process.log'), filter=lambda record: record["extra"].get("name")=="process") # show process
        args.logger= {
            'params': logger.bind(name='params'),
            'process': logger.bind(name='process'),  
        } 

        args.logger['display'] = ['epochs', 'e_stop', 'batch_size', 'lr', 'lr_plm', 'seed']


    ## 2. parameters for training
    args.train = {
        'show':  False,
        'tasks': [task, dataset],

        'seed':    2025,
        'e_start': 0, # start epoch
        'e_stop':  64,
        'epochs':  64,
        'lr':      1e-2,
        'lr_plm':  3e-5,
        'batch_size': 32, 
        
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'device_ids': [0],
        'do_test':  True,
        'do_valid': True,
        'do_train': True,
        'save_model': False,
        'inference':  False, 
        'eval_rate':     1.0, # evaluate times per epoch
        'eval_rate_max': 3.0,
        'data_rate':     1.0,

        'l2reg': 0.01,
        'warmup_ratio':  0.3,
        'weight_decay':  1e-3,
        'adam_epsilon':  1e-8,
        'max_grad_norm': 5.0,
    }


    ## 3. parameters for model
    args.model = {
        'name': model,
        'drop_rate': 0.3,
    }

    return args