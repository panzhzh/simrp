import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

## Sentence-Transformers Semantic Retrieval
def get_retrieval(dataset, cover=['test']):
    from sentence_transformers import SentenceTransformer, util
    model_path = 'sentence-transformers/all-distilroberta-v1'
    sbert = SentenceTransformer(model_path)

    trainset_sent = [f"sentence: '{s['sentence']}', aspect: '{s['aspect']}'" for s in dataset.datas['train']]
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
class ALSC_INS_Base(object):
    def __init__(self, base_metric='f1') -> None:
        self.base = base_metric
        self.results = {
            'train': { self.base: 0, 'loss': 0 }, 
            'valid': { self.base: 0 }, 
            'test':  { self.base: 0 }
            }
        
    def _score(self, results, stage='valid'):
        if None in results or stage == 'train': return {'f1': 0.0, 'acc': 0.0}
        
        preds = np.concatenate([rec['preds'] for rec in results])
        truthes = np.concatenate([rec['labels'] for rec in results])

        score_f1 = round(f1_score(truthes, preds, average='macro')*100, 2)
        score_acc = round(accuracy_score(truthes, preds)*100, 2)

        return {
            'f1'  : score_f1,
            'acc' : score_acc,
        }

    def _output(self, sample, mode='gas'):
        if mode == 'gas':
            ## GAS
            sample['input'] = f"The sentiment of '{sample['sentence']}' toward '{sample['aspect']}' is "
            sample['output'] = sample['polarity']
        else:
            ## Paraphrase
            # sample['input'] = f"'{sample['aspect']}' in '{sample['sentence']}' is "
            sample['input'] = f"For '{sample['sentence']}', '{sample['aspect']}' is "
            sample['output'] = {'positive': 'good', 'neutral': 'ok', 'negative': 'bad'}[sample['polarity']]

        return sample

    def _prompt(self, s, trainset, k=1):
        prompt = f"""Please perform the Aspect Level Sentiment Classification task: given a sentence and a specific aspect, predict the sentiment of this sentence toward this aspect. Sentiment must be selected from ['negative', 'neutral', 'positive']. Please return the string only, without any other comments or texts. \n"""

        # ## 每个类别选 k 个demonstrations
        if k > 0:
            prompt += '\n\nFor example: '
            for ret_s in s['retrieval'][0:k]:
                prompt += f"\nSentence: {ret_s['sentence']}\nAspect: {ret_s['aspect']}\nLabel: {ret_s['target']}"

        prompt += f"\n\nNow, complete the task: \nSentence: {s['sentence']} \nAspect: {s['aspect']}\nLabel: "

        return prompt
    

## Instruct-ABSA template
class ALSC_INS_2(ALSC_INS_Base):
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