

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


def score_fuzzy(preds, labels, testset=None, lower=True):

    def ismark_tag(e_p, e_l, subtrees):
        if len(e_l.split(' ')) >= len(e_p.split(' ')) and e_p in e_l:
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
        testset[i]['preds'], testset[i]['labels'] = quad_p, quad_l
        testset[i]['num_pred'], testset[i]['num_label'] = len(quad_p), len(quad_l)
        testset[i]['num_hitting'] = 0            

        for q_p in quad_p: # 预测中是否存在 真实的四元组？？
            mark = False
            for q_l in quad_l:
                if mark==False and q_p[-2:] == q_l[-2:]: # 进一步模糊匹配 a,0
                    if mark==False and q_p[0:2] == q_l[0:2]: 
                        mark = True # 已经完全匹配上了
                    else: 
                        subtrees = parse_split(testset[i]['parse_str_'])
                        mark = ismark_tag(q_p[0], q_l[0], subtrees) and ismark_tag(q_p[1], q_l[1], subtrees)
                        # sentence = testset[i]['sentence'].lower()
                        # mark = ismark_exd(q_p[0], q_l[0], sentence) and ismark_exd(q_p[1], q_l[1], sentence)

                        # print(f"{q_p[0]} vs {q_l[0]} -> {ismark(q_p[0], q_l[0], subtrees)}; {q_p[1]} vs {q_l[1]} -> {ismark(q_p[1], q_l[1], subtrees)}")

            if mark: # 匹配上了 ？
                num_true_positive += 1
                testset[i]['num_hitting'] += 1
        
    precision = float(num_true_positive) / float(num_pred) if num_pred != 0 else 0.0
    recall = float(num_true_positive) / float(num_label) if num_label != 0 else 0.0
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0.0
    
    return {
        'p':  round(precision, 4),
        'r':  round(recall, 4),
        'f1': round(f1, 4)
    }