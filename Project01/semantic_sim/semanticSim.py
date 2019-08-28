#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
语义联想
@author:Administrator
@file:semanticSim.py
@time:2019/08/22
"""
import gensim
import pandas as pd
from keyword_extraction import keywordsExtract

def my_function(w):
    content_list = []
    model = gensim.models.Word2Vec.load('E:\pycharmproject\my_project\model\zhiwiki_news.word2vec')
    # result = pd.Series(model.most_similar(u'%s'%(w)))
    result = pd.Series(model.most_similar(u'{}'.format(w)))
    for item in result:
        sim_word = {'word': item[0], 'weight':"%.4f"%item[1]}
        content_list.append(sim_word)
    content_dict = {'keyword': w, 'array': content_list}
    return content_dict


def semantic_Similarity(text):
    semantic_list = []
    result = keywordsExtract.getKeywords_textrank(text)
    for words in result[:2]:
        source_word = words['word']
        sim_result = my_function(source_word)
        semantic_list.append(sim_result)
    return semantic_list

if __name__ == '__main__':

    data = """
            台湾工业总会是岛内最具影响力的工商团体之一，2008年以来，该团体连续12年发表对台当局政策的建言白皮书，集中反映岛内产业界的呼声。

            台湾工业总会指出，2015年的白皮书就特别提到台湾面临“五缺”（缺水、缺电、缺工、缺地、缺人才）困境，使台湾整体投资环境走向崩坏。然而四年过去，“五缺”未见改善，反而劳动法规日益僵化、两岸关系陷入紧张、对外关系更加孤立。该团体质疑，台当局面对每年的建言，“到底听进去多少，又真正改善了几多”？

            围绕当局两岸政策，工总认为，由数据来看，当前大陆不仅是台湾第一大出口市场，亦是第一大进口来源及首位对外投资地，建议台湾当局摒弃两岸对抗思维，在“求同存异”的现实基础上，以“合作”取代“对立”，为台湾多数民众谋福创利。

            工总现任理事长、同时也是台塑企业总裁的王文渊指出，过去几年，两岸关系紧张，不仅影响岛内观光、零售、饭店业及农渔蔬果产品的出口，也使得岛内外企业对投资台湾却步，2020年新任台湾领导人出炉后，应审慎思考两岸问题以及中国大陆市场。
                """
    semantic_Similarity(data)