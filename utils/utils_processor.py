import torch, time, math, os, random
import numpy as np
from tqdm import tqdm

from torch.optim import Adam, AdamW, SGD
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def set_rng_seed(rng_seed: int=None):
    if rng_seed is None: rng_seed = int(math.modf(time.time())[0] * 1000000)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)    
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(rng_seed)  # forbid hash random 
    return rng_seed

def get_scheduler(args, optimizer, iter_total, method=None):
    scheduler = None
    if method is None: method = args.model['schedule']
    if method is None: return None

    warmup_ratio = args.train['warmup_ratio']
    if 'linear' in method:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=iter_total*warmup_ratio, 
            num_training_steps=iter_total
        )
    if 'cosine' in method:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_ratio*iter_total, 
            num_training_steps=iter_total
        )

    return scheduler

def get_optimizer(args, model, method=None):
    if isinstance(model, torch.nn.DataParallel): model = model.module # for multi-gpu mode
    if method is None: method = args.model['optimize']

    lr, lr_plm = args.train['lr'], args.train['lr_plm']
    weight_decay, adam_epsilon, l2reg = args.train['weight_decay'], args.train['adam_epsilon'], args.train['l2reg']

    no_decay = ['bias', 'LayerNorm.weight']
    if method == 'AdamW_': # lr for non-plm
        plm_params = list(map(id, model.plm_model.parameters()))
        model_params, warmup_params = [], []
        for name, model_param in model.named_parameters():
            weight_decay_ = 0 if any(nd in name for nd in no_decay) else weight_decay 
            lr_ = lr_plm if id(model_param) in plm_params else lr

            model_params.append({'params': model_param, 'lr': lr_, 'weight_decay': weight_decay_})
            warmup_params.append({'params': model_param, 'lr': lr_/4 if id(model_param) in plm_params else lr_, 'weight_decay': weight_decay_})
        
        model_params = sorted(model_params, key=lambda x: x['lr'])
        optimizer = AdamW(model_params)

    if method == 'AdamW': # lr_plm for non-plm and plm model
        model_params = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
        ]
        optimizer = AdamW(model_params, lr=lr_plm, eps=adam_epsilon)
    
    if method == 'Adam': 
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(model_params, lr=lr_plm, weight_decay=l2reg)

    if method == 'SGD':
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(model_params, lr=lr, weight_decay=l2reg)

    return optimizer


def init_weight(model, method='xavier_uniform_'):
    if method == 'xavier_uniform_': fc = torch.nn.init.xavier_uniform_
    if method == 'xavier_normal_':  fc = torch.nn.init.xavier_normal_
    if method == 'orthogonal_':     fc = torch.nn.init.orthogonal_

    for name, param in model.named_parameters():
        if 'plm' not in name: # do not init plm weights
            if param.requires_grad:
                if len(param.shape) == 0: continue
                if len(param.shape) <= 1: 
                    stdv = 1. / math.sqrt(param.shape[0])
                    torch.nn.init.uniform_(param, a=-stdv, b=stdv)
                else: fc(param)

def print_trainable_parameters(args, model):
    params_all, params_train = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"): num_params = param.ds_numel
        params_all += num_params
        if param.requires_grad: params_train += num_params
    
    p_train, p_all = f"{round(params_train/1000000, 2)} M", f"{round(params_all/1000000, 2)} M"
    if params_all > 0:
        train_rate = round(100*params_train/params_all, 2) 
    else: train_rate = None
    
    if hasattr(args, 'logger'): 
        args.logger['process'].info(f"train: {p_train} || all params: {p_all} || trainable: {train_rate} %")


class Processor():
    def __init__(self, args, model, dataset) -> None:
        self.args = args
        self.dataset = dataset
        self.model = model.to(args.train['device'])
        init_weight(self.model) # init non-plm weights
        print_trainable_parameters(args, self.model) # print trainable parameters

        if self.dataset.loader: self.dataloader = self.dataset.loader
        else: self.dataloader = self.dataset.get_dataloader(self.args.train['batch_size'])
        self.model.get_optimizer() # init optimizer and scheduler

        self.eval_rate, self.global_step = args.train['eval_rate'], 0
        self.log_step = int(len(self.dataloader['train']) / self.eval_rate)

        if hasattr(args, 'logger'): 
            for k, v in vars(args).items():
                for kk, vv in v.items(): args.logger['params'].info(f"{k}.{kk}: {vv}")
            args.logger['params'].info(f"\n {'='*160} \n")

            display = ''
            for item in args.logger['display']: 
                if item in args.train: display += f"{item}: {args.train[item]}, "
                if item in args.model: display += f"{item}: {args.model[item]}, "
            args.logger['process'].warning(display)

    def train_desc(self, epoch, ttime=None):
        args, metrics = self.args, self.dataset.metrics.results
        epochs, model_name, data_name = args.train['epochs'], args.model['name'], self.dataset.name[-1]
        m = self.dataset.metrics.base
        m_tr, m_vl, m_te = round(metrics['train'][m], 3), round(metrics['valid'][m], 3), round(metrics['test'][m], 3)
        m_tr_loss = round(metrics['train']['loss'], 3) if 'loss' in metrics['train'] else 0.0
        desc = f"training: {epoch}/{epochs} ({model_name}=>{data_name}: {str(m_tr)}/{str(m_vl)}/{str(m_te)}, loss: {str(m_tr_loss)}, time: {ttime})"
        self.tqdm_epochs.set_description(desc)
        if epoch>=0: self.tqdm_epochs.update()

    def train_stop(self, epoch=None):
        metric_valid = self.dataset.metrics.results['valid']
        early_threshold = epoch-metric_valid['epoch'] if 'epoch' in metric_valid else 0

        ## 0. reach the early stop threshold
        if early_threshold >= self.args.train['e_stop']:
            return True
        
        ## 1. stop update for a long time
        if early_threshold: 
            self.eval_rate = self.args.train['eval_rate']+early_threshold*0.5
            self.eval_rate = min(self.eval_rate, 3.0)
        else: self.eval_rate = self.args.train['eval_rate']
        self.log_step = int(len(self.dataloader['train']) / self.eval_rate)

    def train_batch(self, batch, bi=None):
        ## 0. evaluate the performance
        if self.global_step % self.log_step == 0:
            if self.args.train['do_valid']: self._evaluate(stage='valid')
            if self.args.train['do_test'] and self.model.valid_update: self._evaluate(stage='test')
        
        self.model.train()
        ## 1. model training
        for key, val in batch.items(): 
            if not isinstance(val, torch.Tensor): continue
            batch[key] = val.to(self.args.train['device'])
        outs = self.model.training_step(batch, bi)  
        
        ## 2. backward and update
        outs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train['max_grad_norm'])
        self.model.optimizer.step()
        if self.model.scheduler is not None: 
            self.model.scheduler.step() 
        self.model.optimizer.zero_grad()       

        self.global_step += 1 # increase global step

    def _train(self):
        epochs, e_start = self.args.train['epochs'], self.args.train['e_start'] if 'e_start' in self.args.train else 0
        self.tqdm_epochs = tqdm(total=epochs, position=0) # epochs process bar
        self.tqdm_epochs.update(e_start); self.train_desc(epoch=-1) # initialize process bar
        for epoch in range(e_start, epochs):
            s_time = time.time()
            self.model.cur_epoch = epoch
            
            torch.cuda.empty_cache()
            if self.args.train['show']: # show each epoch's process bar
                for batch in tqdm(self.dataloader['train'], smoothing=0.05):
                    self.train_batch(batch, bi=-1)
            else: 
                for bi, batch in enumerate(self.dataloader['train']):
                    self.train_batch(batch, bi)
            
            self.model.on_train_epoch_end()

            self.train_desc(epoch, round(time.time()-s_time, 1))
            if self.train_stop(epoch): break 
            
        self.tqdm_epochs.close()
        return self.dataset.metrics.results

    def _evaluate(self, stage='test'):
        self.model.eval()
        with torch.no_grad():
            if self.args.train['show']: # show each epoch's process bar
                for batch in tqdm(self.dataloader[stage], smoothing=0.05):
                    for key, val in batch.items(): 
                        if not isinstance(val, torch.Tensor): continue
                        batch[key] = val.to(self.args.train['device'])
                    if stage == 'valid': self.model.validation_step(batch, -1)
                    if stage == 'test': self.model.test_step(batch, -1)
            else:
                for bi, batch in enumerate(self.dataloader[stage]):
                    for key, val in batch.items(): 
                        if not isinstance(val, torch.Tensor): continue
                        batch[key] = val.to(self.args.train['device'])
                    if stage == 'valid': self.model.validation_step(batch, bi)
                    if stage == 'test': self.model.test_step(batch, bi)
            
        if stage == 'valid': self.model.on_validation_end()
        if stage == 'test': self.model.on_test_end()
        return self.dataset.metrics.results