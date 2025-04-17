
import torch, os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

from transformers import logging
logging.set_verbosity_error()

from utils_processor import *
from utils_model import *

# config 中已经添加路径了
from data_loader import ASQPDataModule


class ASQPDataset_Gen(ASQPDataModule):
    def sem_syn_encode(self, syn_dim=128, mode='syn'):
        if mode != 'sem':
            vocab, weight = self.tag_vocab[0:syn_dim], self.tag_vocab_idf[0:syn_dim]
            for stage, samples in self.datas.items():
                for sample in samples:
                    vec = [0] * len(vocab)
                    for tag in sample['syntax']['tag_str_split']:
                        if tag in vocab: vec[vocab.index(tag)] += 1
                    sample['synvec'] = torch.tensor([v*w for v,w in zip(vec, weight)])
        
        if mode != 'syn':
            for stage, samples in self.datas.items():
                for sample in tqdm(samples):
                    sample['semvec'] = self.sbert.encode(sample['sentence'])

    def get_sentence_tag(self, dim=256):
        import re, spacy, benepar, math
        from collections import Counter
        from nltk import Tree

        def parse_split(s):
            stack, parentheses_contents = [], []
            for i, char in enumerate(s):
                if char == '(': stack.append(i)
                elif char == ')' and stack:
                    start = stack.pop()
                    parentheses_contents.append(s[start:i+1])
            return parentheses_contents

        def tag_prasing(sentence=None, sent_dep=None, nlp=None):
            syntax = {'tag_str': sent_dep}
            if sentence is not None:
                sent = list(nlp(sentence).sents)[0]
                syntax['tag_str'] = sent._.parse_string
                syntax['tag_str_clean'] = re.sub(r"\(([\w#$',.!?-]+) [\w#$',.!?-]+\)", r"\1", syntax['tag_str'])
                syntax['tag_str_split'] = parse_split(syntax['tag_str_clean'])

            syntax['tag_tree'] = Tree.fromstring(syntax['tag_str'])

            return syntax

        nlp = spacy.load("en_core_web_lg")
        if not spacy.util.is_package("benepar_en3"): benepar.download('benepar_en3')
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

        for stage, samples in self.datas.items():
            # if stage != 'train': continue # only trainset
            for sample in tqdm(samples): sample['syntax'] = tag_prasing(sentence=sample['sentence'], nlp=nlp)

        tags_list = [s['syntax']['tag_str_split'] for s in self.datas['train']]
        N, tags_set = len(tags_list), []
        for tags in tags_list: tags_set.extend(list(set(tags)))
        cot, idf = Counter(tags_set), {}
        cot_sort = dict(sorted(cot.items(), key=lambda item: item[1], reverse=True))
        for token, num in cot_sort.items(): idf[token] = math.log(N / num)
        tag_vocab = list(cot_sort.keys())[0:dim]
        tag_vocab_idf = [idf[token] for token in tag_vocab]

        self.tag_vocab = tag_vocab
        self.tag_vocab_idf = tag_vocab_idf

    def get_retrieval(self, args, sbert=None):
        if sbert is None: 
            self.sbert = SentenceTransformer(f"{args.file['plm_dir']}/sbert/all-roberta-large-v1") # 'all-distilroberta-v1'  
        else: self.sbert = sbert

        # 0. encodding
        self.get_sentence_tag()  # not related to aspect/opinion
        self.sem_syn_encode(mode='all') # synvec & semvec encoding
        # self = torch.load(f"{args.file['cache_dir']}dataset_tmp.pth")

        # 1. retrieval
        trainset_syn = torch.stack([s['synvec'] for s in self.datas['train']])
        trainset_sem = torch.tensor([s['semvec'] for s in self.datas['train']])
        for stage, samples in self.datas.items():
            if samples is None: continue
            for sample in tqdm(samples):
                sim_sem = util.cos_sim(sample['semvec'], trainset_sem)[0] # 1.1 sematic similarity
                if stage == 'train': sim_sem[sample['index']] = -1e8 # remove self
                sim_sem_max = sim_sem.argsort(descending=True) 
                sample['ret_sem'] = sim_sem_max.tolist()[0:10]
                
                sim_syn = torch.sum((sample['synvec']-trainset_syn)**2, dim=-1) # 1.2 syntax similarity
                if stage == 'train': sim_syn[sample['index']] = 1e8 # remove self
                sim_syn_max = sim_syn.argsort(descending=False) 
                sample['ret_syn'] = sim_syn_max.tolist()[0:10]

                ret_syn = sim_syn_max[0:10] # 1.3 syntax first, then semantic           
                semvec_ret_syn = torch.stack([trainset_sem[i] for i in ret_syn])
                sim_syn_sem = util.cos_sim(sample['semvec'], semvec_ret_syn)[0]
                ret_syn_sem = sim_syn_sem.argsort(descending=True)
                sample['ret_com'] = [ret_syn[idx].item() for idx in ret_syn_sem]


    def update_input(self, sample, ret_num=1):
        sample['output'] = '; '.join([f"({q[0]}, {q[1]}, {q[2]}, {q[3]})" for q in sample['quads']])
        
        tmp = ''
        for k in range(ret_num):
            ret_sample = self.datas['train'][sample['ret_com'][k]]
            out = '; '.join([f"({q[0]}, {q[1]}, {q[2]}, {q[3]})" for q in ret_sample['quads']])
            tmp += f"{ret_sample['sentence']} {self.tokenizer.eos_token} {out} {self.tokenizer.eos_token} "

        sample['input'] = tmp + sample['sentence']
        return sample


def import_model(args, task='absa'):
    ## 0. model backbone
    args.model['plm'] = args.file['plm_dir'] + f"{args.model['arch']}-{args.model['scale']}"

    ## 1. load dataset
    args.model['data'] = f"{args.file['cache_dir']}{args.model['name']}_dataset.pt"
    if os.path.exists(args.model['data']):
        dataset = torch.load(args.model['data'])
    else:
        data_path = f"{args.file['data_dir']}{args.train['tasks'][-1]}/"
        dataset = ASQPDataset_Gen(data_path)   
        dataset.get_retrieval(args) # syn2vec encode & retrieval
        torch.save(dataset, args.model['data'])

    # for stage, samples in dataset.datas.items():
    #     for sample in samples: sample = dataset.metrics._output(sample, mode='sft')
    dataset.tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
    dataset.batch_size = args.train['batch_size']
    dataset.max_seq_len = 256

    ## 2. change input
    for stage, samples in dataset.datas.items():
        for sample in samples:  # if few_shot, update input
            sample = dataset.update_input(sample, args.model['ret_num'])
    dataset.setup(None)
    
    ## 3. load model
    model = SimRP(
        args=args,
        dataset=dataset,
    )
    return model, dataset
   

class SimRP(ModelForGeneration):
    def __init__(self, args, dataset, plm=None):
        super().__init__() 
        self.args = args
        self.dataset = dataset

        self.plm_model = AutoModelForSeq2SeqLM.from_pretrained(plm if plm is not None else args.model['plm'])
        # self.plm_model.encoder.dropout.p = args.model['drop_rate']
        # self.plm_model.decoder.dropout.p = args.model['drop_rate']
        
    def forward(self, inputs, stage='train'):
        plm_outs = self.plm_model(
            inputs['input_ids'],
            labels=inputs['input_ids_out'], 
            )
        loss = plm_outs.loss

        return {
            'loss':   loss,
        }

    def _eval(self, inputs, stage='test'): # generate mode
        outputs = self.plm_model.generate(
            input_ids=inputs['input_ids'],
            # attention_mask=inputs['attention_mask'],
            max_length=self.dataset.max_seq_len,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=1
            )

        preds = [self.dataset.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs['sequences']]
        labels = [self.dataset.tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs['input_ids_out']]

        return {
            # 'inputs': inputs,  'outputs': outputs,
            'preds': preds, 'labels': labels
        }

    def validation_step(self, batch, batch_idx):
        if self.cur_epoch > 5:
            output = self._eval(batch, stage='valid')
            # output = self(batch, stage='valid')
        else: output = None
        self.validation_step_outputs.append(output)

        return output
