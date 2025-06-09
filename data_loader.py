import os, torch, json
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class ASQPDataModule(Dataset):
    def __init__(self, data_dir, batch_size=32, num_workers=8, lower=False) -> None:
        super().__init__()
        self.name = ['asqp', data_dir.split('/')[-2]]
        self.lower = lower
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_init() # dataset initialize
        self.prepare_data(['train', 'valid', 'test']) # data parse
        
        from asqp_instruction import Instruct_Base as Instruct
        if 'acos' in self.name[-1]: from asqp_instruction import Instruct_ACOS as Instruct
        self.metrics = Instruct(base_metric='f1')

    def dataset_init(self):
        self.info = {
            'num_samp': {}, # 句子数量 
            'num_quad': {}, # 四元组数量
            'len_out': {}, # 输出长度
            'len_in': {}, # 输入长度
        }
        self.datas, self.loader = {}, {}

    def prepare_data(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            raw_path, samples = f'{self.data_dir}/{stage}.json', []
            if not os.path.exists(raw_path): return None
            with open(raw_path, 'r', encoding='utf-8-sig') as fp: lines = json.load(fp)

            for line in tqdm(lines):
                sample = {
                    'index': len(samples),
                    'sentence': line['sentence'],
                    'quads': line['quads'], # (a,o,s,c)
                }
                sample['quad_num'] = len(sample['quads'])
                samples.append(sample)

            if stage == 'train':
                self.info['category'] = list(set([q[0] for s in samples for q in s['quads']]))
                self.info['sentiment'] = list(set([q[-1] for s in samples for q in s['quads']]))
            self.datas[stage] = samples

    def setup(self, tokenizer=None, max_seq_len=256, mode='gen'):
        if tokenizer: self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        for stage, samples in self.datas.items():
            if samples is None: continue
            for sample in samples:
                embedding_in = self.tokenizer.encode_plus(sample['input'], return_tensors="pt")
                sample['input_ids'] = embedding_in['input_ids'].squeeze()
                sample['attention_mask'] = embedding_in['attention_mask'].squeeze()

                embedding_out = self.tokenizer.encode_plus(sample['output'], return_tensors="pt")
                sample['input_ids_out'] = embedding_out['input_ids'].squeeze()
                sample['attention_mask_out'] = embedding_out['attention_mask'].squeeze()
            
            self.info['num_samp'][stage] = len(samples)
            self.info['num_quad'][stage] = [s['quad_num'] for s in samples]
            self.info['len_in'][stage]   = [len(s['input_ids']) for s in samples]
            self.info['len_out'][stage]  = [len(s['input_ids_out']) for s in samples]

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
            batch_size=self.batch_size*2, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datas['test'], 
            batch_size=self.batch_size*2, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
    def collate_fn(self, samples):
        pad_token_id = self.tokenizer.pad_token_id
        inputs = {
            'input_ids': pad_sequence([sample['input_ids'] for sample in samples], batch_first=True, padding_value=pad_token_id),
            'attention_mask': pad_sequence([sample['attention_mask'] for sample in samples], batch_first=True, padding_value=0),
            'input_ids_out': pad_sequence([sample['input_ids_out'] for sample in samples], batch_first=True, padding_value=pad_token_id),
            'attention_mask_out': pad_sequence([sample['attention_mask_out'] for sample in samples], batch_first=True, padding_value=0),
        }
        return inputs



if __name__ == '__main__':
    data_dir = './rest16/' # './rest_15/'
    dataset = ASQPDataModule(data_dir)

    ## quad in sentence
    for stage, samples in dataset.datas.items():
        for sample in samples:
            for quad in sample['quads']:
                a, o, c, p = quad
                if a != 'NULL' and a not in sample['sentence']:
                    print(f"{stage}_{sample['index']}: aspect {a}' not in '{sample['sentence']}'")
                if o not in sample['sentence']:
                    print(f"{stage}_{sample['index']}: opinion '{o}' not in '{sample['sentence']}'")

    print('done')

