import os, torch, json, copy
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score


class Metrics(object):
    def __init__(self, base_metric='f1', dataset=None) -> None:
        self.base = base_metric
        self.results = {
            'train': { self.base: 0, 'loss': 0 }, 
            'valid': { self.base: 0 }, 
            'test':  { self.base: 0 }
            }
        
        self.dataset = dataset # 可有可无

    def _score(self, results, stage='train'):
        preds = np.concatenate([rec['preds'].cpu().numpy() for rec in results])
        truthes = np.concatenate([rec['labels'].cpu().numpy() for rec in results])
        losses = [rec['loss'].item() for rec in results]

        score_f1 = round(f1_score(truthes, preds, average='macro')*100, 2)
        score_acc = round(accuracy_score(truthes, preds)*100, 2)
        score_loss = round(sum(losses)/len(losses), 4)

        return {
            'f1'  : score_f1,
            'acc' : score_acc,
            'loss': score_loss
        }


class ALSCDataModule(Dataset):
    def __init__(self, data_dir, batch_size=2, num_workers=0) -> None:
        super().__init__()
        self.name = ['alsc', data_dir.split('/')[-2]]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_init() # dataset initialize
        self.prepare_data() # prepare data
        self.get_tokenizer_()
        self.num_classes = len(self.tokenizer_['labels']['l2i'])
        self.metrics = Metrics(base_metric='f1')

        self.datas['valid'] = self.datas['test'] # no validation set

    def dataset_init(self):
        self.info = {
            'max_seq_token_num': {}, # 句子 最长长度
            'max_asp_token_num': {}, # aspect 最长长度
            'total_samples_num': {}, # 样本数量
            'class_category': {},    # 类别统计
        }

        # 初始化数据集要保存的内容 
        self.datas, self.loader = {}, {}
        self.tokenizer_ = {
            'labels': { 'l2i': {}, 'i2l': {} }
        }

    def get_tokenizer_(self, ): # statistics of labels
        sample_total = self.datas['train'] + self.datas['test']
        try:
            polarities = [sample['aspects'][0]['polarity'] for sample in sample_total]
        except: polarities = [sample['polarity'] for sample in sample_total]

        self.tokenizer_['labels']['l2i'] = {l: i for i, l in enumerate(set(polarities))}
        self.tokenizer_['labels']['i2l'] = {i: l for i, l in enumerate(set(polarities))}
        # self.tokenizer_['labels']['count'] = {l: sum([1 for p in polarities if p==l]) for l in set(polarities)}

    def prepare_data(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            raw_path = f'{self.data_dir}/{stage}_alsc.json'
            if not os.path.exists(raw_path): continue
            with open(raw_path, 'r', encoding='utf-8') as fp: samples = json.load(fp)
            self.datas[stage] = samples

    def prepare_data_(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            raw_path = f'{self.data_dir}/{stage}.multiple.json'
            if not os.path.exists(raw_path): return None
            with open(raw_path, 'r', encoding='utf-8') as fp: raw_samples, samples = json.load(fp), []

            for sample in tqdm(raw_samples):
                aspects = sample['aspects']
                for aspect in aspects:
                    temp = copy.deepcopy(sample)
                    temp['index'] = len(samples)
                    temp['aspect'] = ' '.join(aspect['term'])
                    temp['aspect_pos'] = [aspect['from'], aspect['to']]
                    if 'tokens' not in temp: temp['tokens'] = temp['token']
                    if 'sentence' not in temp: temp['sentence'] = ' '.join(temp['tokens'])
                    if ' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]]) != temp['aspect']:
                        print(f"{' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]])} -> {temp['aspect']}")
                    temp['polarity'] = aspect['polarity']
                    samples.append(temp)

            self.datas[stage] = samples

    def setup(self, tokenizer, stage=None, mode='cls'):
        self.tokenizer, self.mode = tokenizer, mode
        for stage, samples in self.datas.items():
            if samples is None: continue
            self.info['class_category'][stage] = {l: 0 for l in self.tokenizer_['labels']['i2l'].keys()}
            for sample in samples:
                if 'sentence' not in sample: sample['sentence'] = ' '.join(sample['tokens'])
                embedding = tokenizer.encode_plus(sample['sentence'], sample['aspect'], return_tensors='pt')
                sample['input_ids'] = embedding['input_ids'].squeeze(dim=0)
                sample['attention_mask'] = embedding['attention_mask'].squeeze(dim=0)
                sample['label'] = self.tokenizer_['labels']['l2i'][sample['polarity']]
                
                if mode == 'gen':
                    embedding = tokenizer.encode_plus(sample['input'], return_tensors='pt')
                    sample['input_ids'] = embedding['input_ids'].squeeze(dim=0)
                    embedding_t = tokenizer.encode_plus(sample['output'], return_tensors='pt')
                    sample['input_ids_t'] = embedding_t['input_ids'].squeeze(dim=0)


    def get_dataloader(self, batch_size=None):
        if batch_size: self.batch_size = batch_size
        for stage, _ in self.datas.items():
            if stage=='train': self.loader[stage] = self.train_dataloader()
            if stage=='valid': self.loader[stage] = self.val_dataloader()
            if stage=='test':  self.loader[stage] = self.test_dataloader()
        return self.loader

    def train_dataloader(self):
        return DataLoader(
            self.datas['train'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datas['valid'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datas['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
    def collate_fn(self, samples):
        pad_token_id = self.tokenizer.pad_token_id
        inputs = {
            'input_ids': pad_sequence([sample['input_ids'] for sample in samples], batch_first=True, padding_value=pad_token_id), 
            #'attention_mask': pad_sequence([sample['attention_mask'] for sample in samples], batch_first=True, padding_value=0),
            'label': torch.tensor([sample['label'] for sample in samples])
        }
        if self.mode == 'gen':
            inputs['input_ids_t'] = pad_sequence([sample['input_ids_t'] for sample in samples], batch_first=True, padding_value=pad_token_id)

        return inputs
    

if __name__ == '__main__':
    data_dir = '/home/jzq/My_Codes/CodeFrame/Datasets/Textual/absa/twi/'
    dataset = ALSCDataModule(data_dir)
    plm_dir = None
    tokenizer = AutoTokenizer.from_pretrained(plm_dir)
    dataset.setup(tokenizer)