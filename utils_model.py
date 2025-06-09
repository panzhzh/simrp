import json, torch
import torch.nn as nn
from utils_processor import get_optimizer, get_scheduler


class PoolerAll(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class ModelForClassification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.cur_epoch = 0

    def forward(self, inputs, stage='train'):
        raise NotImplementedError

    def get_optimizer(self):
        iter_total = int(len(self.dataset.loader['train'])*self.args.train['epochs'])
        self.optimizer = get_optimizer(args=self.args, model=self)
        self.scheduler = get_scheduler(self.args, self.optimizer, iter_total)

    def training_step(self, batch, batch_idx):
        output, cur_e = self(batch, stage='train'), self.cur_epoch,
        self.training_step_outputs.append(output)

        return {
            'loss': output['loss']
        }
    
    def on_train_epoch_end(self):
        outputs, metrics_tr = self.training_step_outputs, self.dataset.metrics.results['train']
        metrics_tr.update(self.dataset.metrics._score(outputs, stage='train'))
        metrics_tr['loss'] = round(torch.stack([output['loss'] for output in outputs]).mean().item(),3)
        metrics_tr['epoch'] = self.cur_epoch
        
        self.training_step_outputs = [] # init record
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_tr.items()})
        if hasattr(self.args, 'logger'):
            self.args.logger['process'].info(f"train_eval: {describe}")
    
    def validation_step(self, batch, batch_idx):
        output = self(batch, stage='valid')
        self.validation_step_outputs.append(output)

        return output

    def on_validation_end(self):
        outputs, metrics_vl = self.validation_step_outputs, self.dataset.metrics.results['valid']
        metrics = self.dataset.metrics._score(outputs, stage='valid')

        ## update best model
        mark, self.valid_update = self.dataset.metrics.base, False
        if metrics[mark] > metrics_vl[mark]: # bigger is better
            metrics_vl.update(metrics)
            metrics_vl['epoch'] = self.cur_epoch
            describe = json.dumps({k: round(float(v),4) for k,v in metrics_vl.items()})
            if hasattr(self.args, 'logger'): self.args.logger['process'].info(f"valid: {describe}")
            self.valid_update = True # execute test

            if self.args.train['save_model']: 
                if self.args.model['use_adapter']: 
                    self.save_checkpoint_peft()
                else: self.save_checkpoint() # 保存模型

        self.validation_step_outputs = [] # init record

    def save_checkpoint(self, save_path=None, mode='save'):
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            state = {
                'net': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])

    def save_checkpoint_peft(self, save_path=None, mode='save'):
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            training_params_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
            state = {
                'net': training_params_dict,
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'], strict=False)
            self.optimizer.load_state_dict(state['optimizer'])

    def test_step(self, batch, batch_idx):
        output = self(batch, stage='test')
        self.test_step_outputs.append(output)

    def on_test_end(self):
        outputs, metrics_te = self.test_step_outputs, self.dataset.metrics.results['test']
        metrics_te.update(self.dataset.metrics._score(outputs, stage='test'))
        metrics_te['epoch'] = self.cur_epoch
        
        self.test_step_outputs = []
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_te.items()})
        if hasattr(self.args, 'logger'):
            self.args.logger['process'].info(f"test: {describe}")


class ModelForGeneration(ModelForClassification):

    def validation_step(self, batch, batch_idx):
        output = self._eval(batch, stage='valid') # generative
        self.validation_step_outputs.append(output)
        
        return output

    def test_step(self, batch, batch_idx):
        output = self._eval(batch, stage='test') # generative
        self.test_step_outputs.append(output)

