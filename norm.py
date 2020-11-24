import json
from stanfordcorenlp import StanfordCoreNLP
import logging
import sys
import argparse
import pdb
import os

import numpy as np

from stanfordcorenlp import StanfordCoreNLP

sys.path.append('./script/normalise')
from normalise.normalisation import normalise

logger = logging.getLogger(__name__)


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
    doc_tokens, norm_flag = normalise(doc_tokens, verbose=False)
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


if __name__ == '__main__':
    # raw_tweet = "#8501qs why can't in-flight cock pit recorders be recorded real time electronically for instant answers?— david ogle (@dacogle) December 28, 2014"
    raw_tweet = "#InTheUnlikelyEvent"
    try:
        nlp = StanfordCoreNLP(r'/home/rthuang/script/stanford-corenlp-full-2018-10-05/')
        para, para_tokens = pre_process(raw_tweet, nlp)
        print(para)
        print(para_tokens)
    finally:
        nlp.close()




