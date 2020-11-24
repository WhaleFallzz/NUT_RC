import string
import re
import json
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
import logging
import sys
import argparse
import pdb
import os

import nltk
# nltk.download()

from nltk.translate.bleu_score import sentence_bleu
import numpy as np

from stanfordcorenlp import StanfordCoreNLP
sys.path.append('/home/rthuang/script/normalise')
from normalise.normalisation import normalise

from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

from script.tweetqa_eval import evaluate as extract_eval

meteor_scorer = Meteor()
rouge_scorer = Rouge()

logger = logging.getLogger(__name__)


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


def read_data(file_in):
    with open(file_in, 'r') as f_obj:
        data = json.load(f_obj)
    return data


def write_data(new_data, file_out):
    with open(file_out, 'w') as f_obj:
        json.dump(new_data, f_obj)


def pre_process(raw_tweet, nlp):
    def split_hashTag_userId(tweet_tokens):
        """将#hashTag和@userId分隔开 -> ['#', 'hashTag'] ['@', 'userId']"""
        split_tokens = []
        for token in tweet_tokens:
            if token[0] == '#' or token[0] == '@' and len(token) > 1:
                split_tokens.extend([token[0], token[1:]])
            elif token[:2] == '.@':
                split_tokens.extend([token[:2], token[2:]])
            else:
                split_tokens.append(token)
        return split_tokens

    def is_whitespace(c):
        # '\r'：回车
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    # stanfordCoreNLP分词
    tweet_tokens = nlp.word_tokenize(raw_tweet)
    # pdb.set_trace()

    # 分割
    doc_tokens = split_hashTag_userId(tweet_tokens)
    # pdb.set_trace()

    # normalise - SPLT / (WDLK-None)
    doc_tokens = normalise(doc_tokens, verbose=False)
    # pdb.set_trace()

    # norm_tweet = ' '.join(doc_tokens)

    abb_list = ["'s", "n't", "'m", "'re", 'k']     
    norm_tweet = doc_tokens[0]
    for token in doc_tokens[1:]:
        if token in abb_list:
            norm_tweet += token
        else:
            norm_tweet += ' ' + token

    doc_tokens = []
    prev_is_whitespace = True
    for c in norm_tweet:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False

    return ' '.join(doc_tokens), doc_tokens

def find_fake_answer(sample, do_pre_process, do_char_split, nlp):
    # para = sample['Tweet']
    para = sample['Norm_tweet']
    answers = sample["Answer"]

    # 分词
    if do_pre_process:
        try:
            para, para_tokens = pre_process(para, nlp)
        except:
            print(para)
            para_tokens = para.split()
    else:
        para_tokens = para.split()
        

    if len(answers) > 1:
        max_ans_len = max(len(answers[0].split()), len(answers[1].split()))
    else:
        max_ans_len = len(answers[0].split())
    # normalise_ans:同评价
    gold_list = [normalize_answer(ref) for ref in answers]

    best_match_score = 0
    best_match_span = [0, 0]
    best_fake_answer = para_tokens[0]
    match_flag = False

    exclude = set(string.punctuation)
    # n-gram(词)比较
    for start_tidx in range(len(para_tokens)):
        if para_tokens[start_tidx] in exclude:
           continue
        for end_tidx in range(start_tidx, start_tidx + 2 * max_ans_len):
            if end_tidx > len(para_tokens):
                break
            span_tokens = para_tokens[start_tidx: end_tidx + 1]
            norm_span_string = normalize_answer(' '.join(span_tokens))

            # 比较
            bleu = sentence_bleu([_.split() for _ in gold_list], norm_span_string.split(), weights=(1, 0, 0, 0))
            meteor, _ = meteor_scorer.compute_score({0:gold_list}, {0:[norm_span_string]})
            rouge, _ = rouge_scorer.compute_score({0:gold_list}, {0:[norm_span_string]})
            match_score = 0.6 * bleu + 0.1 * meteor + 0.3 * rouge

            if match_score > best_match_score:
                match_flag = True
                best_match_span = [start_tidx, end_tidx]
                best_match_score = match_score
                best_fake_answer = ' '.join(span_tokens)
    
    # 输出得分较低的
    # if best_match_score != 1:
    #     print(para)
    #     print('Gloden:', gold_list)
    #     print('Fake_a:', best_fake_answer)
    #     print('M_socr:', best_match_score)
    #     # input()

    # 遍历n-gram(字符级别比较)
    if best_match_score != 1 and do_char_split:
        # 0 or best_match_score(word-level)
        c_best_match_score = 0
        # c_best_match_score = best_match_score

        c_best_match_span = [0, 0]
        c_best_fake_answer = best_fake_answer

        # 遍历
        for start_tidx in range(len(para_tokens)):
            if para_tokens[start_tidx] in exclude:
                continue
            for end_tidx in range(start_tidx, start_tidx + 2 * max_ans_len):
                if end_tidx > len(para_tokens):
                    break
                span_tokens = list(normalize_answer(' '.join(para_tokens[start_tidx: end_tidx + 1])))

                bleu = sentence_bleu([list(_) for _ in gold_list], span_tokens, weights=(1, 0, 0, 0))
                # meteor, _ = meteor_scorer.compute_score({0:[' '.join(list(_)) for _ in gold_list]}, {0:[' '.join(span_tokens)]})
                # rouge, _ = rouge_scorer.compute_score({0:[' '.join(list(_)) for _ in gold_list]}, {0:[' '.join(span_tokens)]})
                match_score = bleu

                if match_score > c_best_match_score:
                    c_best_match_span = [start_tidx, end_tidx]
                    c_best_fake_answer = ' '.join(para_tokens[start_tidx: end_tidx + 1])
                    c_best_match_score = match_score

        if c_best_match_score > 0 and c_best_match_span[0] != 0:
            # bug
            c_answer_start = len(' '.join(para_tokens[:c_best_match_span[0]])) + 1

            assert para[c_answer_start: c_answer_start + len(c_best_fake_answer)] == c_best_fake_answer
            # if para[c_answer_start: c_answer_start + len(c_best_fake_answer)] != c_best_fake_answer:
            #     print('Char')
            #     print(sample)
            #     print(para)
            #     print(para_tokens)
            #     print(para[c_answer_start: c_answer_start + len(c_best_fake_answer)])
            #     print(c_best_fake_answer)
            #     print()
        else:
            c_answer_start = 0

        return para, c_answer_start, c_best_fake_answer, c_best_match_score, 'char'

    if best_match_score > 0 and best_match_span[0] != 0:
        # bug
        answer_start = len(' '.join(para_tokens[:best_match_span[0]])) + 1
        assert para[answer_start: answer_start + len(best_fake_answer)] == best_fake_answer
        #if para[answer_start: answer_start + len(best_fake_answer)] != best_fake_answer:
        #    print('Word')
        #    print(sample)
        #    print(para)
        #    print(para_tokens)
        #    print(para[answer_start: answer_start + len(best_fake_answer)])
        #    print(best_fake_answer)
        #    print()
    else:
        answer_start = 0

    return para, answer_start, best_fake_answer, best_match_score, 'word'


def process(file_in, args):
    data = read_data(file_in)
    new_data = []
    bad_case = []
    count = 0
    try:
        nlp = StanfordCoreNLP(r'/home/rthuang/script/stanford-corenlp-full-2018-10-05/')

        for sample in tqdm(data):
            cleaned_tweet, ans_start, fake_answer, match_score, match_type\
                = find_fake_answer(sample, do_pre_process=args.do_pre_process, do_char_split=args.do_char_split, nlp=nlp)
            sample['Cleaned_tweet'] = cleaned_tweet
            sample['Answer_start'] = ans_start
            sample['Fake_answer'] = [fake_answer]
            sample['Match_score'] = match_score
            sample['Match_type'] = match_type
            # pdb.set_trace()

            new_data.append(sample)
            if match_score != 1:
                bad_case.append(sample)
                count += 1
        print('Non_Match:', count)

    finally:
        nlp.close()
    
    file_out = os.path.join(args.output_dir, args.file_suffix)
    file_non_match = os.path.join(args.output_dir, 'non_match_' + args.file_suffix)
    write_data(new_data, file_out)
    write_data(bad_case, file_non_match)

    # 打印结果
    for idx, entry in enumerate(new_data[:20]):
        print('Idx\t%s' % str(idx))
        print('R_Tweet:\t%s' % entry["Tweet"])
        print('N_Tweet:\t%s' % entry["Norm_tweet"])
        print('C_Tweet:\t%s' % entry["Cleaned_tweet"])
        print('Question:\t%s' % entry["Question"])
        print('G_Truth:\t%s' % entry["Answer"])
        print('Fake_Ans:\t%s' % entry["Fake_answer"])
        print('M_Score:\t%s' % entry["Match_score"])
        print('M_Type: \t%s' % entry["Match_type"])

    ex_results = extract_eval(file_in, file_out)
    print(ex_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the norm_data will be written.")
    parser.add_argument("--do_pre_process", default=True, type=bool, required=False,
                        help="Wether to do pre_process")
    parser.add_argument("--do_char_split", default=True, type=bool, required=False,
                        help="Wether to do char split")                        
    
    args = parser.parse_args()
    train_file = './data/original_version/train.json'
    dev_file = './data/original_version/dev.json'
    args.output_dir = './data/Nrom_Debug'

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print('Output directory ({}) already exists and is not empty. (input 1 continue)'.format(args.output_dir))
        x = input()
        if x != '1':
            sys.exit()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # args.file_suffix = 'train.json'
    # print(args)
    # process(train_file, args)
    # args.file_suffix = 'dev.json'
    # print(args)
    # process(dev_file, args)

    # Debug
    sample = {
        "Question": "who does brian windhorst work for?",
        "Answer": [ "espn", "espn" ],
        "Tweet": "LeBron James does not have a meeting set with Cavs at this time, sources say.\u2014 Brian Windhorst (@WindhorstESPN) July 9, 2014",
        "qid": "8707ed53557fc040403a7742fcf9b4b6",
    }
    para = "#8501qs why can't in-flight cock pit recorders be recorded real time electronically for instant answers?— david ogle (@dacogle) December 28, 2014"
    try:
        nlp = StanfordCoreNLP(r'/home/rthuang/script/stanford-corenlp-full-2018-10-05/')
        para, para_tokens = pre_process(para, nlp)
        print(para)
        print(para_tokens)
        # cleaned_tweet, ans_start, fake_answer, match_score, match_type = find_fake_answer(sample, True, True, nlp)
    finally:
        nlp.close()




