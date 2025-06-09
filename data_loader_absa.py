import os, torch, json
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 成分句法分析中的常见成分类别及其说明
constituency_categories = {
    "S": "Simple declarative clause",
    "SBAR": "Clause introduced by a (possibly empty) subordinating conjunction",
    "SBARQ": "Direct question introduced by a wh-word or a wh-phrase",
    "SINV": "Inverted declarative sentence",
    "SQ": "Inverted yes/no question",
    "ADJP": "Adjective Phrase",
    "ADVP": "Adverb Phrase",
    "CONJP": "Conjunction Phrase",
    "FRAG": "Fragment",
    "INTJ": "Interjection",
    "LST": "List marker",
    "NAC": "Not a Constituent; used to show the scope of certain prenominal modifiers within an NP",
    "NP": "Noun Phrase",
    "NX": "Used within certain complex NPs to mark the head of the NP",
    "PP": "Prepositional Phrase",
    "PRN": "Parenthetical",
    "PRT": "Particle",
    "QP": "Quantifier Phrase (i.e., complex measure/amount phrase); used within NP",
    "RRC": "Reduced Relative Clause",
    "UCP": "Unlike Coordinated Phrase",
    "VP": "Verb Phrase",
    "WHADJP": "Wh-adjective Phrase",
    "WHADVP": "Wh-adverb Phrase",
    "WHNP": "Wh-noun Phrase",
    "WHPP": "Wh-prepositional Phrase",
    "X": "Unknown, uncertain, or unbracketable",
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb"
}

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

    def dataset_init(self):
        self.info = {
            'seq_num': {}, # 句子数量 
            'seq_len': {}, # 句子长度
            'seq_quad_num': {}, # 句子中的四元组数量
        }

        # 初始化数据集要保存的内容 
        self.datas = {}
        self.loader = {}

        # # tokenizer
        # self.tokenizer_ = {
        #     'labels': { 'l2i': {}, 'i2l': {}, 'count': {} }
        # }

        # 评价指标
        self.met, default = 'f1', 0 # 主要评价指标
        self.metrics = {
            'train': { self.met: default, 'loss': -1e3}, 
            'valid': { self.met: default, 'loss': -1e3}, 
            'test':  { self.met: default, 'loss': -1e3},
            } # 训练指标
    
    def prepare_data(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            if '_' in self.data_dir: # rest15、rest16是json格式，rest_15、rest_16是txt格式
                raw_path, samples = f'{self.data_dir}/{stage}.txt', []
                if not os.path.exists(raw_path): return None
                with open(raw_path, 'r', encoding='utf-8') as fp: lines = fp.readlines()

                for line in tqdm(lines):
                    if self.lower: line = line.lower()
                    sentence, quads = line.strip().split('####')
                    sample = {
                        'index': len(samples),
                        'sentence': sentence.strip(),
                        'quads': eval(quads), # (a,c,p,o)
                        'quad_num': len(eval(quads))
                    }
                    samples.append(sample)
            else:
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

            self.info['seq_num'][stage] = len(samples)
            self.info['seq_len'][stage] = [len(s['sentence'].split()) for s in samples]
            self.info['seq_quad_num'][stage] = [s['quad_num'] for s in samples]
            self.datas[stage] = samples

    def setup(self, tokenizer, stage=None):
        self.tokenizer = tokenizer
        for stage, samples in self.datas.items():
            if samples is None: continue
            # self.info['class_category'][stage] = {l: 0 for l in self.tokenizer_['labels']['i2l'].keys()}
            for sample in samples:
                embedding = tokenizer.encode_plus(sample['sentence'], return_tensors='pt')
                sample['input_ids'] = embedding['input_ids'].squeeze(dim=0)
                sample['attention_mask'] = embedding['attention_mask'].squeeze(dim=0)
                sample['token_type_ids'] = embedding['token_type_ids'].squeeze(dim=0)
                # sample['label'] = self.tokenizer_['labels']['l2i'][sample['polarity']]
                
                # self.info['class_category'][stage][sample['label']] += 1

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
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def get_sentence_tag(file_path=None, dataset=None, mode='save'):




    
    if os.path.exists(file_path): 
        return torch.load(file_path)

    # 句法解析
    import re, spacy, benepar, math
    from collections import Counter

    def parse_split(s):
        stack, parentheses_contents = [], []
        for i, char in enumerate(s):
            if char == '(': stack.append(i)
            elif char == ')' and stack:
                start = stack.pop()
                parentheses_contents.append(s[start:i+1])
        return parentheses_contents

    def get_dependency(nlp, samples):
        for s in tqdm(samples):
            sent = list(nlp(s['sentence']).sents)[0]

            parse_str = sent._.parse_string
            parse_str_ = re.sub(r"\(([\w#$',.!?-]+) [\w#$',.!?-]+\)", r"\1", parse_str)
            parse_str_split = parse_split(parse_str_)

            s['parse_str'] = parse_str
            s['parse_str_'] = parse_str_
            s['parse_str_split'] = parse_str_split

            # print(parse_str)
            # tree = Tree.fromstring(parse_str)
            # tree.draw()

    nlp = spacy.load("en_core_web_lg")
    if not spacy.util.is_package("benepar_en3"): benepar.download('benepar_en3')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    for stage, samples in dataset.datas.items():
        # if stage != 'train': continue # 只统计训练集的
        samples = get_dependency(nlp, samples)

    parse_parts = [s['parse_str_split'] for s in dataset.datas['train']]
    N, parse_parts_set = len(parse_parts), []
    for part in parse_parts:
        parse_parts_set.extend(list(set(part))) # 已经将一个句子中重复的去掉了
    cot, idf = Counter(parse_parts_set), {}
    for token, num in cot.items():
        if num == 1: continue # 排除掉只出现在一个句子中的 
        idf[token] = math.log(N / num)

    dataset.idf = idf
    dataset.cot = cot

    # cot_sort = dict(sorted(cot.items(), key=lambda item: item[1], reverse=True))
    # parse_tag_list = list(cot_sort.keys())[0:256]
    # parse_tag_list_idf = [idf[token] for token in parse_tag_list]
    # dataset.parse_tag_list = parse_tag_list
    # dataset.parse_tag_list_idf = parse_tag_list_idf
    
    torch.save(dataset, file_path)
    return dataset





if __name__ == '__main__':



    
    ## 加载数据集 (rest15、rest16是json格式，rest_15、rest_16是txt格式)
    data_dir = './rest15/' # './rest_15/'
    dataset = ASQPDataModule(data_dir)

    print(dataset)
    # get_dataset_tag(dataset, mode='save')

