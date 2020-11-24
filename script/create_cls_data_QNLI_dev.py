import string
import re
import json

# import nltk
# nltk.download()

from nltk.translate.bleu_score import sentence_bleu
import numpy as np

from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

meteor_scorer = Meteor()
rouge_scorer = Rouge()

def ans_score(ans, gold_list):
    ans = normalize_answer(ans)
    gold_list = [normalize_answer(ref) for ref in gold_list]
    bleu = sentence_bleu([_.split() for _ in gold_list], ans.split(), weights=(1,0,0,0))
    meteor, _ = meteor_scorer.compute_score({0:gold_list}, {0:[ans]})
    rouge, _ = rouge_scorer.compute_score({0:gold_list}, {0:[ans]})
    return {'bleu': bleu, 'meteor':meteor, 'rouge': rouge}

# def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
def evaluate(test_annotation_file, user_annotation_file):
    gold_file = test_annotation_file
    pred_file = user_annotation_file
    gold = json.load(open(gold_file))
    pred = json.load(open(pred_file))
    idx2gold = {item['qid']:item['Answer'] for item in gold}
    idx2pred = {item['qid']:item['Answer'] for item in pred}
    idx2scores = {}
    for id_ in idx2gold.keys():
        if isinstance(idx2pred[id_], list):
            pred_ans = idx2pred[id_][0]
        else:
            pred_ans = idx2pred[id_]
        idx2scores[id_] = ans_score(pred_ans, idx2gold[id_])
    bleus = [item['bleu'] for item in idx2scores.values()]
    meteors = [item['meteor'] for item in idx2scores.values()]
    rouges = [item['rouge'] for item in idx2scores.values()]
    print({'BLEU': np.mean(bleus), 'METEOR': np.mean(meteors), 'ROUGE': np.mean(rouges)})

    output = {}
    output['result'] = [
    {'test_split': 
        {
        'BLEU-1': np.mean(bleus),
        'METEOR': np.mean(meteors),
        'ROUGE': np.mean(rouges)
        }
    }
    ]

    return output


def create_cls_data(ex_file, ge_file, src_file=None, do_para=False):
    with open(ex_file, 'r') as f_obj:
        ex_preds = json.load(f_obj)
    with open(ge_file, 'r') as f_obj:
        ge_preds = json.load(f_obj)
    with open(src_file, 'r') as f_obj:
        src_data = json.load(f_obj)
    print(len(ex_preds), len(ge_preds), len(src_data))

    idx2gold = {item['qid']:item['Answer'] for item in src_data}
    idx2raw_tweet = {item['qid']:item['Tweet'] for item in src_data}
    idx2clean_tweet = {item['qid']:item['Cleaned_tweet'] for item in src_data}
    idx2question = {item['qid']:item['Question'] for item in src_data}
    idx2fake = {item['qid']:item['Fake_answer'] for item in src_data}

    idx2extract = {item['qid']:item['Answer'] for item in ex_preds}
    idx2generate = {item['qid']:item['Answer'] for item in ge_preds}

    idx_ex_scores = {}
    idx_ge_scores = {}
    idx2betters = {}

    idx2same = []

    # 抽取和生成分开(qid对齐)
    ex_cls_data = []
    ge_cls_data = []
    qids = []
    # 全部case个数
    eval_num = 0
    # 得分较高的case
    ex_num = 0
    ge_num = 0
    # 
    case_idx = 0
    

    for id_ in idx2gold.keys():
        pred_ex_ans = idx2extract[id_]
        pred_ge_ans = idx2generate[id_]

        # 计算得分
        ex_scores = ans_score(pred_ex_ans, idx2gold[id_])
        idx_ex_scores[id_] = ex_scores
        ge_scores = ans_score(pred_ge_ans, idx2gold[id_])
        idx_ge_scores[id_] = ge_scores
        # 得分比较
        ex_score = 0.7*ex_scores['bleu'] + 0.1*ex_scores['meteor'] + 0.2*ex_scores['rouge']
        ge_score = 0.7*ge_scores['bleu'] + 0.1*ge_scores['meteor'] + 0.2*ge_scores['rouge']
        
        if do_para:
            ex_item1 = idx2question[id_] + ' [SEP] ' + idx2clean_tweet[id_]
            ex_item2 = pred_ex_ans
            ge_item1 = idx2question[id_] + ' [SEP] ' + idx2clean_tweet[id_]
            ge_item2 = pred_ge_ans
        else:
            ex_item1 = idx2question[id_]
            ex_item2 = pred_ex_ans
            ge_item1 = idx2question[id_]
            ge_item2 = pred_ge_ans

        if ex_score > ge_score:
            item = str(case_idx) + '\t' + ex_item1 + '\t' + ex_item2 + '\t' + 'entailment'
            ex_cls_data.append(item)
            item = str(case_idx) + '\t' + ge_item1 + '\t' + ge_item2 + '\t' + 'not_entailment'
            ge_cls_data.append(item)
            case_idx += 1

            qids.append(id_)

            # 记录最好的
            idx2betters[id_] = ex_scores
            ex_num += 1

        elif ex_score < ge_score:            
            item = str(case_idx) + '\t' + ex_item1 + '\t' + ex_item2 + '\t' + 'not_entailment'
            ex_cls_data.append(item)
            item = str(case_idx) + '\t' + ge_item1 + '\t' + ge_item2 + '\t' + 'entailment'
            ge_cls_data.append(item)
            case_idx += 1

            qids.append(id_)

            # 记录最好的
            idx2betters[id_] = ge_scores
            ge_num += 1

        else:   # 两者预测的相同
            idx2same.append({
                'qid': id_,
                'Answer': pred_ge_ans
            })
            eval_num += 1
            # 记录最好的
            idx2betters[id_] = ex_scores


    # 输出得分
    bleus = [item['bleu'] for item in idx_ex_scores.values()]
    meteors = [item['meteor'] for item in idx_ex_scores.values()]
    rouges = [item['rouge'] for item in idx_ex_scores.values()]
    print('Ex', {'BLEU': np.mean(bleus), 'METEOR': np.mean(meteors), 'ROUGE': np.mean(rouges)})

    bleus = [item['bleu'] for item in idx_ge_scores.values()]
    meteors = [item['meteor'] for item in idx_ge_scores.values()]
    rouges = [item['rouge'] for item in idx_ge_scores.values()]
    print('Ge', {'BLEU': np.mean(bleus), 'METEOR': np.mean(meteors), 'ROUGE': np.mean(rouges)})

    bleus = [item['bleu'] for item in idx2betters.values()]
    meteors = [item['meteor'] for item in idx2betters.values()]
    rouges = [item['rouge'] for item in idx2betters.values()]
    print('Better', {'BLEU': np.mean(bleus), 'METEOR': np.mean(meteors), 'ROUGE': np.mean(rouges)})

    print(eval_num, ex_num, ge_num)

    return qids, ex_cls_data, ge_cls_data, idx2same


if  __name__ == "__main__":
    # train
    # ex_file = '/home/rthuang/debug/bert_finetuned_squad_denoising/predictions_train.json'
    # ge_file = '/home/rthuang/debug/multi_task/predictions_train.json'
    # src_file = '/home/rthuang/TweetQA/TweetQA_data/corenlp_norm_word+char/train.json'
    # dev
    ex_file = '/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/bert-large-wwm-finetuned/predictions_dev.json'
    ge_file = '/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/multi_task/predictions_dev.json'
    src_file = '/home/rthuang/TweetQA/TweetQA_data/corenlp_norm_word+char/dev.json'
    
    # query+ans
    qids, ex_cls_data, ge_cls_data, idx2same = create_cls_data(ex_file=ex_file, ge_file=ge_file, src_file=src_file)
    assert len(qids) == len(ex_cls_data) == len(ge_cls_data)
    print(len(qids))

    with open('cls_data/extract_ans/dev.tsv', 'w') as writer:
        writer.write('index\tquestion\tsentence\tlabel' + "\n")
        for item in ex_cls_data:
            writer.write(item + "\n")
    
    with open('cls_data/extract_ans/qid.txt', 'w') as writer:
        for qid in qids:
            writer.write(qid + "\n")

    with open('cls_data/generation_ans/dev.tsv', 'w') as writer:
        writer.write('index\tquestion\tsentence\tlabel' + "\n")
        for item in ge_cls_data:
            writer.write(item + "\n")

    with open('cls_data/generation_ans/qid.txt', 'w') as writer:
        for qid in qids:
            writer.write(qid + "\n")

    with open('cls_data/same.json', 'w') as writer:
        writer.write(json.dumps(idx2same, indent=4) + "\n")


    
    # # query + passage + ans 
    # cls_data = create_cls_data(ex_file=ex_file, ge_file=ge_file, src_file=src_file, do_para=True)
    # print(len(cls_data))
    # with open('cls_data_query_para_ans.json', 'w') as writer:
    #     writer.write(json.dumps(cls_data, indent=4) + "\n")