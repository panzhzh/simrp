import numpy as np


## Sentence-Transformers Semantic Retrieval
def get_retrieval(dataset, cover=['test']):
    import torch
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer, util
    model_path = 'sentence-transformers/all-distilroberta-v1'
    sbert = SentenceTransformer(model_path)

    trainset_sent = [s['sentence'] for s in dataset.datas['train']]
    trainset_sent_embedding = sbert.encode(trainset_sent)
    for stage, samples in dataset.datas.items():
        if stage not in cover: continue
        for s in tqdm(samples):
            sim = util.cos_sim(sbert.encode(s['sentence']), trainset_sent_embedding)[0]
            sim_index = torch.argsort(sim, descending=True).tolist()
            if stage == 'train': sim_index.pop(sim_index.index(s['index']))
            s['ret_index'] = sim_index[:10]
            s['retrieval'] = [dataset.datas['train'][i] for i in s['ret_index']]

    return dataset


## Basic LLM template
class Instruct_Base(object):
    def __init__(self, base_metric='f1', tokenizer=None) -> None:
        self.base = base_metric
        self.results = {
            'train': { self.base: 0, 'loss': 0 }, 
            'valid': { self.base: 0 }, 
            'test':  { self.base: 0 }
            }
        self.tokenizer = tokenizer

    def score_paraphrase(self, preds, golds):
        n_tp, n_gold, n_pred = 0, 0, 0
        for i in range(len(preds)):
            n_gold += len(golds[i])
            n_pred += len(preds[i])

            for t in preds[i]:
                if t in golds[i]: n_tp += 1

        precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        
        return {
            'p':  round(precision*100, 2),
            'r':  round(recall*100, 2),
            'f1': round(f1*100, 2)
        }

    def score_ele(self, preds, golds, mode=['a','o']):
        length = 1
        pre, rec, f1 = {k: 0 for k in mode}, {k: 0 for k in mode}, {k: 0 for k in mode}
        for m in mode:
            if m == 'a': ele_ps, ele_gs = [[q[0] for q in quad] for quad in preds], [[q[0] for q in quad] for quad in golds]
            if m == 'o': ele_ps, ele_gs = [[q[1] for q in quad] for quad in preds], [[q[1] for q in quad] for quad in golds]

            num_true_positive, num_label, num_pred = 0, 0, 0
            for i, (ps, gs) in enumerate(zip(ele_ps, ele_gs)):
                ps, gs = [e.lower() for e in ps if len(e.split(' '))==length], [e.lower() for e in gs if len(e.split(' '))==length]
                num_pred, num_label = num_pred+len(ps), num_label+len(gs)
                for ele in ps:
                    if ele in gs: 
                        num_true_positive += 1
                
            pre[m] = float(num_true_positive) / float(num_pred) if num_pred != 0 else 0.0
            rec[m] = float(num_true_positive) / float(num_label) if num_label != 0 else 0.0
            f1[m] = 2*pre[m]*rec[m] / (pre[m]+rec[m]) if pre[m]+rec[m] else 0.0
            print(f"{m}: {num_label} -> {round(f1[m]*100, 2)}")

        return f1

    def score(self, preds, labels, lower=True):
        num_true_positive, num_label, num_pred = 0, 0, 0
        # 每个 pred/truth 包含若干个四元组, 看看预测的四元组是否在真实的四元组中
        for i, (quad_p, quad_l) in enumerate(zip(preds, labels)):
            if lower: quad_p, quad_l = [[it.lower() for it in q] for q in quad_p], [[it.lower() for it in q] for q in quad_l]
            num_pred, num_label = num_pred+len(quad_p), num_label+len(quad_l)
            pred_temp = [] # avoid duplicate quadruples
            for quad in quad_p: # 预测中是否存在 真实的四元组？？
                if quad in pred_temp: continue
                else: pred_temp.append(quad)
                if quad in quad_l: 
                    num_true_positive += 1
            
        precision = float(num_true_positive) / float(num_pred) if num_pred != 0 else 0.0
        recall = float(num_true_positive) / float(num_label) if num_label != 0 else 0.0
        f1 = 2*precision*recall / (precision+recall) if precision+recall else 0.0
        
        return {
            'p':  round(precision*100, 2),
            'r':  round(recall*100, 2),
            'f1': round(f1*100, 2)
        }

    def _score(self, results, stage='valid'):
        if None in results or stage == 'train': return {'p': 0, 'r': 0, 'f1': 0}

        preds_str = np.concatenate([rec['preds'] for rec in results])
        labels_str = np.concatenate([rec['labels'] for rec in results])
        preds_str, labels_str = [p.lower() for p in preds_str], [l.lower() for l in labels_str]

        def _parse_seq(seq_str):
            tmp = seq_str[seq_str.find("(")+1:seq_str.rfind(")")].split(', ')
            if len(tmp) == 4: 
                quad = tmp
            else: quad = []

            return quad

        quad_preds, quad_labels = [], []
        for pred_str, label_str in zip(preds_str, labels_str):
            quad_preds.append( [_parse_seq(it_str) for it_str in  pred_str.split('; ')])
            quad_labels.append([_parse_seq(it_str) for it_str in label_str.split('; ')]) 

        score = self.score(quad_preds, quad_labels)
        return score

    def _output(self, sample, mode='icl'):
        if mode == 'icl':
            sample['input'], sample['output'] = sample['sentence'], str(sample['quads'])
        else: # mode == 'sft
            sample['input'] = sample['sentence']
            sample['output'] = '; '.join([f"({q[0]}, {q[1]}, {q[2]}, {q[3]})" for q in sample['quads']])
        
        return sample

    def _prompt(self, sample, trainset, ret_k=0, stage='train'):
        if not hasattr(self, 'asp_category'): 
            self.asp_category = list(set([q[0] for s in trainset for q in s['quads']]))
            self.asp_sentiment = list(set([q[-1] for s in trainset for q in s['quads']]))

        prompt = f"""###Perform the **Aspect Sentiment Quad Prediction** task: \nGiven a sentence, identify all possible quads in the form of **(Category, Aspect, Oopinion, Sentiment)**, where: \n- **Category** is selected from the predefined list: {self.asp_category}. \n- **Aspect** is a span from the sentence (usually a noun or a noun phrase) or 'NULL' if implicit. \n- **Opinion** is a span from the sentence expressing an evaluative statement (the shortest but most capable of reflecting the sentiment polarity). \n- **Sentiment** is chosen from {self.asp_sentiment}. \n"""  
        prompt += "###**Output Format:** \nReturn a Python list of tuples, where each tuple contains four strings in single quotes. Ensure that the response consists of the list only, without additional comments or symbols. \n"

        # **Example:**  
        # Given the sentence: *Gross food - Wow -*  
        # The potential quads are:  

        for i in range(ret_k):
            if i == 0: prompt += "\n###For example: \n"
            ret_sample = sample['retrieval'][i]
            prompt += f"Input sentence: {ret_sample['sentence']} \nThe potential quads are: {ret_sample['output']}\n"

        prompt += f"\n###Now, \nInput sentence: {sample['sentence']} \nThe potential quads are: "

        return prompt
    

## 2021EMNLP.GAS template
class Instruct_GAS(Instruct_Base):
    def __init__(self, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.sent_dict = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
    
    def _parse_seq(self, input_seq):
        try:
            aocs = input_seq.strip().split(', ')
            a, o, c, s = aocs[0], aocs[1], aocs[2], aocs[3]
        except: c, s, a, o = '', '', '', ''

        return [a.strip(), o.strip(), c.strip(), s.strip()]

    def get_score(self, results, stage='valid'):
        if None in results: return {'p': 0, 'r': 0, 'f1': 0}

        preds_str = np.concatenate(
            [[self.tokenizer.decode(ids, skip_special_tokens=True) for ids in rec['outputs']['sequences']] for rec in results]
            )
        labels_str = np.concatenate(
            [[self.tokenizer.decode(ids, skip_special_tokens=True) for ids in rec['inputs']['input_ids_out']] for rec in results]
            )
        preds_str, labels_str = [p.lower() for p in preds_str], [l.lower() for l in labels_str]

        quad_preds, quad_labels = [], []
        for i in range(len(preds_str)):
            quad_preds.append([self._parse_seq(seq[1:-1]) for seq in preds_str[i].split('; ')]) # 以 ; 分割
            quad_labels.append([self._parse_seq(seq[1:-1]) for seq in labels_str[i].split('; ')]) # 去掉首尾的括号

        score = self.score(quad_preds, quad_labels)
        score_ele = self.score_ele(quad_preds, quad_labels, mode=['a', 'o'])
        return score

    def get_output(self, datas, item='quad_ori'):
        for stage, samples in datas.items():
            for s in samples: 
                s['input'] = s['sentence']
                # 1. 替换 sentiment 为常见 word; 2. NULL 替换为 it
                # quads = [[q[0] if q[0]!='NULL' else 'it', q[1], q[2], self.sent_dict[q[3]]] for q in s['quads']] 
                s['output'] = '; '.join([f"({q[0]}, {q[1]}, {q[2]}, {q[3]})" for q in s[item]])

        return datas  
    

## 2021EMNLP.Paraphrase template
class Instruct_Paraphrase(Instruct_Base):
    def __init__(self, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.sent_dict = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
    
    def _parse_seq(self, input_seq):
        try:
            cs, ao = input_seq.strip().split(' because ')
            c_s, a_o = cs.strip().split(' is '), ao.strip().split(' is ')
            c, s, a, o = c_s[0], c_s[1], a_o[0], a_o[1]
        except: c, s, a, o = '', '', '', ''

        if a.strip() == 'it': a = 'null'
        return [a.strip(), o.strip(), c.strip(), s.strip()]

    def get_output(self, datas):
        idx = {'[A]': 0, '[O]': 1, '[S]': 3, '[C]': 2} # 原有quad顺序为 aocs
        for stage, samples in datas.items():
            for s in samples: 
                s['input'] = s['sentence']
                # 1. 替换 sentiment 为常见 word; 2. NULL 替换为 it
                quads = [[q[0] if q[0]!='NULL' else 'it', q[1], q[2], self.sent_dict[q[3]]] for q in s['quads']] 
                s['output'] = ' [SSEP] '.join([f"{q[2]} is {q[3]} because {q[0]} is {q[1]}" for q in quads])

        return datas  


## 2023ACL.MvP template
class Instruct_MvP(Instruct_Base):
    def __init__(self, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.sent_dict = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
    
    def _parse_seq(self, input_seq, locat_tokens=['[a]', '[o]', '[s]', '[c]']):
        seq_dict = {t: {
            'si': input_seq.find(t), # 开始位置
            'ei': -1,                # 结束位置
            'str': 'none',           # 提取字符串
            } for t in locat_tokens}
        
        keys = list(seq_dict.keys())
        begin_indexes = [seq_dict[k]['si'] for k in keys] + [len(input_seq)]
        for ki, key in enumerate(keys):
            cur_begin = seq_dict[key]['si']
            assert cur_begin == begin_indexes[ki]
            if cur_begin == -1: continue
            diff = [ind-cur_begin if ind>cur_begin else 1e8 for ind in begin_indexes]
            val, next = min(diff), diff.index(min(diff))
            if val > 1e7: continue
            seq_dict[key]['ei'] = begin_indexes[next]
            seq_dict[key]['str'] = input_seq[cur_begin+len(key):begin_indexes[next]].strip()

        if seq_dict['[a]']['str'].lower() == 'it': seq_dict['[a]']['str'] = 'null' 
        return [seq_dict[k]['str'] for k in keys]

    def get_output(self, datas, order='aosc'):
        idx = {'[A]': 0, '[O]': 1, '[S]': 3, '[C]': 2} # 原有quad顺序为 aocs
        if order == 'aosc': ord = ['[A]', '[O]', '[S]', '[C]']
        if order == 'aocs': ord = ['[A]', '[O]', '[C]', '[S]']
        if order == 'acos': ord = ['[A]', '[C]', '[O]', '[S]']

        for stage, samples in datas.items():
            for s in samples: 
                s['input'] = s['sentence'] + f" {' '.join(ord)} "
                quads = [[q[0], q[1], q[2], self.sent_dict[q[3]]] for q in s['quads']] # 替换 sentiment 为常见 word
                s['output'] = ' [SSEP] '.join([f"{ord[0]} {q[idx[ord[0]]]} {ord[1]} {q[idx[ord[1]]]} {ord[2]} {q[idx[ord[2]]]} {ord[3]} {q[idx[ord[3]]]}" for q in quads])

        return datas    


## Instruct-ABSA template
class Instruct_ABSA(Instruct_Base):
    def __init__(self, args, dataset) -> None:
        self.args = args
        self.dataset = dataset
    
    def get_output(self, dataset=None):
        if dataset is None: dataset = self.dataset
        for stage, samples in dataset.datas.items():
            for s in samples:

                s['target'] = ', '.join([f"{q[0]}:{q[1]}:{q[2]}:{q[3]}" for q in s['quads']])
        
        return dataset

    def get_prompt(self, s, trainset, stage='train'):
        if not hasattr(self, 'asp_category'): 
            self.asp_category = list(set([q[1] for s in trainset for q in s['quads']]))

        prompt = f"""Definition: The output will be the aspects (both implicit and explicit), the corresponding opinion/describing terms, the aspect category {self.asp_category}, and the sentiment polarity (positive, negative, neutral) of the opinion term. 
        The output should be in the format: aspect:opinion:sentiment:category, aspect:opinion:sentiment:category, ...
        In cases where there are no aspects the output should be NULL:NULL:NULL:NULL."""
        
        idxs = np.arrange(10)
        for i in idxs:
            prompt += f"\nInput: {trainset[i]['sentence']}\nOutput: {trainset[i]['target']}"

            prompt += f"\nInput: {s['sentence']} \nOutput: "
        s['prompt'] = prompt

        return s


## 2021ACL.ACOS template (opinion can be NULL)
class Instruct_ACOS(Instruct_Base):
    def _prompt(self, sample, trainset, ret_k=0, stage='train'):
        if not hasattr(self, 'asp_category'): 
            self.asp_category = list(set([q[0] for s in trainset for q in s['quads']]))
            self.asp_sentiment = list(set([q[-1] for s in trainset for q in s['quads']]))

        prompt = f"""###Perform the **Aspect Sentiment Quad Prediction** task: \nGiven a sentence, identify all possible quads in the form of **(Category, Aspect, Oopinion, Sentiment)**, where: \n- **Category** is selected from the predefined list: {self.asp_category}. \n- **Aspect** is a span from the sentence (usually a noun or a noun phrase) or 'NULL' if implicit. \n- **Opinion** is a span from the sentence expressing an evaluative statement (the shortest but most capable of reflecting the sentiment polarity) or 'NULL' if implicit. \n- **Sentiment** is chosen from {self.asp_sentiment}. \n"""  
        prompt += "###**Output Format:** \nReturn a list of tuples, where each tuple contains four strings in single quotes. Ensure that the response consists of the list only, without additional comments or symbols. \n"

        # **Example:**  
        # Given the sentence: *Gross food - Wow -*  
        # The potential quads are:  

        for i in range(ret_k):
            if i == 0: prompt += "\n###For example: \n"
            ret_sample = sample['retrieval'][i]
            prompt += f"Input sentence: {ret_sample['sentence']} \nThe potential quads are: {ret_sample['output']}\n"

        prompt += f"\n###Now, \nInput sentence: {sample['sentence']} \nThe potential quads are: "

        return prompt