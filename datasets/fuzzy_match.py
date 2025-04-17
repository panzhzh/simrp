import os, torch, re, argparse, copy, math
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from collections import Counter


def parse_split(s):
    stack, parentheses_contents = [], []
    for i, char in enumerate(s):
        if char == '(': stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            parentheses_contents.append(s[start:i+1])
    return parentheses_contents

def extract_leaf_nodes(parse_tree):
    stack = []
    leaf_nodes = []
    for i, char in enumerate(parse_tree):
        if char == '(':
            stack.append(i)
        elif char == ')':
            start = stack.pop()
            subtree = parse_tree[start:i+1]
            if '(' not in subtree[1:-1]:
                leaf_nodes.append(subtree[1:-1].strip())
    
    return [n.split(' ')[1].lower() for n in leaf_nodes]


def structure_encode(dataset, dim=None):
    # # 取出 idf 最大的前 dim 个词
    # idf_sort = dict(sorted(dataset.idf.items(), key=lambda item: item[1], reverse=True))
    # parse_tag_list = list(idf_sort.keys())
    # if dim is not None: parse_tag_list = parse_tag_list[0:dim]
    # parse_tag_list_idf = [dataset.idf[token] for token in parse_tag_list]

    # 取出 频率 最多的前 dim 个词
    cot_sort = dict(sorted(dataset.cot.items(), key=lambda item: item[1], reverse=True))
    parse_tag_list = list(cot_sort.keys())
    if dim is not None: parse_tag_list = parse_tag_list[0:dim]
    parse_tag_list_idf = [dataset.idf[token] for token in parse_tag_list]

    vocab, weight = parse_tag_list, parse_tag_list_idf
    for stage, samples in dataset.datas.items():
        if samples is None: continue
        for s in samples:
            vec = [0] * len(vocab)
            for str_split in s['sentence_tag_split']:
                if str_split in vocab:
                    vec[vocab.index(str_split)] += 1
            s['structure_vec'] = [v*w for v,w in zip(vec,weight)]

def get_retrieval(dataset, sbert=None, dim=None):
    import torch
    from sentence_transformers import SentenceTransformer, util
    if sbert is None:
        model_path = 'Y:/CodeFrame/Pretrained_Models/sbert/all-distilroberta-v1'
        sbert = SentenceTransformer(model_path)

    trainset = dataset.datas['train']
    if 'asqp' in dataset.name:
        structure_encode(dataset, dim=dim) 
        train_structure_embed = torch.tensor([s['structure_vec'] for s in trainset])
        train_sentences = [s['sentence'] for s in trainset]
        train_sentences_embed = sbert.encode(train_sentences)
        for stage, samples in dataset.datas.items():
            if samples is None: continue
            for s in tqdm(samples):
                # 1. 语义相似检索
                if stage == 'train':
                    sim_sentence = util.cos_sim(train_sentences_embed[s['index']], train_sentences_embed)[0]
                    sim_sentence[s['index']] = -1e8
                else: sim_sentence = util.cos_sim(sbert.encode(s['sentence']), train_sentences_embed)[0]
                sim_sentence_max_index = sim_sentence.argsort(descending=True) # 每种lab 取前十个最大的
                s['ret_sem'] = sim_sentence_max_index.tolist()[0:10]
                # 2. 结构相似检索
                if stage == 'train':
                    sim_structure = torch.sum((train_structure_embed[s['index']].unsqueeze(dim=0)-train_structure_embed)**2, dim=-1)
                    sim_structure[s['index']] = 1e8 # 越小越相似
                else: sim_structure = torch.sum((torch.tensor(s['structure_vec']).unsqueeze(dim=0)-train_structure_embed)**2, dim=-1)
                sim_structure_max_index = sim_structure.argsort(descending=False) # 每种lab 取前十个最小的
                s['ret_stu'] = sim_structure_max_index.tolist()[0:10]
                # 3. 先结构后语义检索
                ret_stu = sim_structure_max_index.tolist()[0:10]
                ret_stu_sem_embed = [train_sentences_embed[i] for i in ret_stu]
                ret_stu_sem_sim = util.cos_sim(sbert.encode(s['sentence']), ret_stu_sem_embed)[0]
                ret_stu_sem_sim_sort = ret_stu_sem_sim.argsort(descending=True) # 每种lab 取前十个最大的
                s['ret_com'] = [ret_stu[idx] for idx in ret_stu_sem_sim_sort][0:10]

def load_dataset(args, file_path=None):
    if file_path is None: file_path = f"./{args.dataset}/dataset_ret_asqp_tag.pt"
    if not os.path.exists(file_path):
        # dataset = ASQPDataModule(f"./{args.dataset}/") # 加载基本数据集
        # get_sentence_tag(dataset) # 获取句法成分内容
        dataset = torch.load(f"./{args.dataset}/dataset_tag.pt") # 加载已经 tag 的数据集
        get_retrieval(dataset, dim=args.tag_dim) # 获取检索
        torch.save(dataset, file_path)
    else: dataset = torch.load(file_path)

    return dataset

class QueryGPT(object):
    def __init__(self, args, dataset, settings=['0-shot']):
        self.args = args
        self.dataset = dataset
        self.testset = dataset.datas['test']
        self.trainset = dataset.datas['train']
        self.category = list(set([q[2] for s in self.trainset for q in s['quads']]))
        # 指定 生成输出的格式
        for s in self.trainset: 
            s['target'] = str([(q[0], q[1], q[3], q[2]) for q in s['quads']]) # aosc
        # for s in trainset: s['target'] = str([(q[0], q[3], q[2], q[1]) for q in s['quads']]) # aosc for rest_15

        self.api_key = args.api
        self.base_model = args.model
        self.add_tokens, self.sub_tokens = [], []

        ## 执行搜索
        for setting in settings:
            if setting not in self.testset[0]:
                self.forward(setting)

    def get_prompt(self, s, demo_num=0):
        prompt = f"""Please perform Aspect Sentiment Quad Prediction task. Given the sentence, tag all (aspect, opinion, sentiment, category) quadruples. Aspect and opinion should be substring of the sentence. Category should be selected from {self.category}. Sentiment should be selected from ['negative', 'neutral', 'positive']. Only aspect can be 'NULL', category, opinion and sentiment cannot be 'NULL'. Return a python list of tuples containing four strings in double quotes. Please return python list only, without any other comments or texts. \n\n"""

        for i in range(demo_num):
            # prompt += f"\nSemantic similar example {i} -"
            prompt += f"Sentence: {self.trainset[s['ret_com'][i]]['sentence']}\nLabel: {self.trainset[s['ret_com'][i]]['target']}\n"
        if demo_num: prompt += "\n"
        prompt += f"Sentence: {s['sentence']} \nLabel: "
        return prompt

    def query_chatgpt_model(self, prompt, system='', max_tokens=256, temperature=0):
        client = OpenAI(
            api_key=self.api_key,
            # base_url="https://api.xiaoai.plus/v1" # "https://api.chatanywhere.com.cn/v1"  # 国内转发需要
        )
        try:
            response = client.chat.completions.create(
                model=self.base_model,
                messages=[
                    {'role': 'system', "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens, n=1, stop=None, temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return False

    def forward(self, setting='10-shot'):
        for s in tqdm(self.testset):
            prompt = self.get_prompt(s, demo_num=int(setting.split('-')[0]))
            response = self.query_chatgpt_model(prompt)
            s[setting] = {'prompt': prompt, 'response': response}

    def process_results(self, samples, settings=['0-shot']):
        labels = [s['quads'] for s in samples]
        labels = [[(q[0], q[1], q[3], q[2]) for q in lab] for lab in labels] # aosc
        # labels = [[(q[0], q[3], q[2], q[1]) for q in lab] for lab in labels] # For rest_15
        
        results = {}
        for setting in settings:
            preds_str, preds = [s[setting]['response'] for s in samples], []
            for i, p_str in enumerate(preds_str):
                p_str = p_str.lower()
                if not p_str.startswith('['): p_str = p_str[p_str.find("["):p_str.find("]")+1]
                try:
                    p = eval(p_str)
                except:
                    tmp = [[tmp.strip() for tmp in it.replace("'", "").split(',')] for it in re.findall(r"\((.*?)\)", p_str)]
                    p = [[t[0], t[1], t[2], t[3]] for t in tmp]
                    # print(f"Error in {i}th sample: {p_str}")
                preds.append(p)

            for p in preds:
                for i, q in enumerate(p): p[i] = list(q)
            results[setting] = preds

        return results, labels

    def score(self, preds, labels, setting='0-shot', testset=None, lower=True):
        self.add_tokens, self.sub_tokens = [], []

        def ismark_tag(e_p, e_l, subtrees):
            if len(e_l.split(' ')) >= len(e_p.split(' ')) and e_p in e_l:
                if len(e_l.split(' ')) > len(e_p.split(' ')):
                    self.sub_tokens.append(len(e_l.split(' '))-len(e_p.split(' ')))
                return True
            if e_l == 'null' and e_p != 'null': return False
            if e_l != 'null' and e_p == 'null': return False

            subtrees_str, subtrees_e_l = [extract_leaf_nodes(st) for st in subtrees], []
            for ss in subtrees_str:
                if e_l in ' '.join(ss): subtrees_e_l.append(ss)

            e_l_len = len(e_l.split(' '))
            diff = [e_l_len-len(st) if len(st)>e_l_len else -100 for st in subtrees_e_l]
            if len(diff) == 0: return False
            idx = diff.index(max(diff))
            vague_e_l = ' '.join(subtrees_e_l[idx])

            self.add_tokens.append(len(subtrees_e_l[idx])-e_l_len)

            if e_p in vague_e_l: return True

            return False

        def ismark_exd(e_p, e_l, sentence):
            if len(e_l.split(' ')) >= len(e_p.split(' ')) and e_p in e_l:
                return True
            if e_l == 'null' and e_p != 'null': return False
            if e_l != 'null' and e_p == 'null': return False

            tokens, tokens_l, pos, poses = sentence.split(' '), e_l.split(' '), [None, None], []
            for i, token in enumerate(tokens):
                if token == tokens_l[0]: pos[0] = i
                if pos[0] is not None and tokens_l[-1] in token: pos[1] = i
                
                if pos[0] is not None and pos[1] is not None: 
                    poses.append(pos)
                    pos = [None, None]
            
            add = 5
            for pos in poses:
                if tokens[pos[0]:pos[1]+1] == tokens_l: 
                    pos_exd = [pos[0]-add if pos[0]-add>=0 else 0, pos[1]+add if pos[1]+add<len(tokens) else len(tokens)-1]
                    vague_e_l = ' '.join(tokens[pos_exd[0]:pos_exd[1]+1])
                    if e_p in vague_e_l: return True
            
            return False


        num_true_positive, num_label, num_pred = 0, 0, 0
        # 每个 pred/truth 包含若干个四元组, 看看预测的四元组是否在真实的四元组中
        for i, (quad_p, quad_l) in enumerate(zip(preds, labels)):
            if lower: quad_p, quad_l = [[it.lower() for it in q] for q in quad_p], [[it.lower() for it in q] for q in quad_l]
            num_pred, num_label = num_pred+len(quad_p), num_label+len(quad_l)
            testset[i][setting]['preds'], testset[i][setting]['labels'] = quad_p, quad_l
            testset[i][setting]['num_pred'], testset[i][setting]['num_label'] = len(quad_p), len(quad_l)
            testset[i][setting]['num_hitting'] = 0            

            for q_p in quad_p: # 预测中是否存在 真实的四元组？？
                mark = False
                for q_l in quad_l:
                    if mark==False and q_p[-2:] == q_l[-2:]: # 进一步模糊匹配 a,0
                        if mark==False and q_p[0:2] == q_l[0:2]: 
                            mark = True # 已经完全匹配上了
                        else: 
                            subtrees = parse_split(testset[i]['sentence_tag'])
                            mark = ismark_tag(q_p[0], q_l[0], subtrees) and ismark_tag(q_p[1], q_l[1], subtrees)
                            # sentence = testset[i]['sentence'].lower()
                            # mark = ismark_exd(q_p[0], q_l[0], sentence) and ismark_exd(q_p[1], q_l[1], sentence)

                            # print(f"{q_p[0]} vs {q_l[0]} -> {ismark(q_p[0], q_l[0], subtrees)}; {q_p[1]} vs {q_l[1]} -> {ismark(q_p[1], q_l[1], subtrees)}")

                if mark: # 匹配上了 ？
                    num_true_positive += 1
                    testset[i][setting]['num_hitting'] += 1
            
        precision = float(num_true_positive) / float(num_pred) if num_pred != 0 else 0.0
        recall = float(num_true_positive) / float(num_label) if num_label != 0 else 0.0
        f1 = 2*precision*recall / (precision+recall) if precision+recall else 0.0
        
        return {
            'p':  round(precision, 4),
            'r':  round(recall, 4),
            'f1': round(f1, 4)
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--shots", type=int, default=0, help="[0, 1, 5, 10]")
    parser.add_argument("--dataset", type=str, default='rest15', help="[rest_15, rest_16]")
    parser.add_argument("--tag_dim", type=int, default=256, help="[32,64,128,256,512,1024,2048]")
    parser.add_argument("--api", type=str, default=None, help="api key")
    parser.add_argument("--model", type=str, default="gpt-4o", help="[gpt-3.5-turbo, gpt-4o-mini, gpt-4o]")
    args = parser.parse_args()


    rec_path = f"./results_256_tag_{args.model}_aosc.pt"
    rec = torch.load(rec_path)

    results, labels = rec.process_results(rec.testset, settings=['1-shot', '10-shot'])
    for setting, preds in results.items():
        metric = rec.score(preds, labels, setting, rec.testset)
        print(f"{setting}: {metric}")

