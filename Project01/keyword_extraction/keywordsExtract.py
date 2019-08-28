#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:keywordsExtract.py
@time:2019/08/19
"""
# 采用TextRank方法提取文本关键词  jieba.analyse.textrank
"""
       TextRank权重：

            1、将待抽取关键词的文本进行分词、去停用词、筛选词性
            2、以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
            3、计算图中节点的PageRank，注意是无向带权图
"""
import sys
import pandas as pd
import os
import jieba.analyse

PATH = os.getcwd()
print(PATH)
# 处理标题和摘要，提取关键词
def getKeywords_textrank(text):
    topK = 10
    content_dict = []
    jieba.analyse.set_stop_words("E:\pycharmproject\my_project\worddict\stopword.txt")  # 加载自定义停用词表
    keywords = jieba.analyse.textrank(text, topK=topK, allowPOS=(
    'n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'),withWeight=True)  # TextRank关键词提取，词性筛选withWeight=True
    for item in keywords:
        # print(item[0], item[1]) #打印关键词及权重 参数必须设置withWeight=True
        result = {'word': item[0], 'weight':"%.4f"%item[1]}
        content_dict.append(result)
    return content_dict



if __name__ == '__main__':
    data = '国台办表示中国必然统一。会尽最大努力争取和平统一，但绝不承诺放弃使用武力。'
    data1 = """
        台湾工业总会是岛内最具影响力的工商团体之一，2008年以来，该团体连续12年发表对台当局政策的建言白皮书，集中反映岛内产业界的呼声。

        台湾工业总会指出，2015年的白皮书就特别提到台湾面临“五缺”（缺水、缺电、缺工、缺地、缺人才）困境，使台湾整体投资环境走向崩坏。然而四年过去，“五缺”未见改善，反而劳动法规日益僵化、两岸关系陷入紧张、对外关系更加孤立。该团体质疑，台当局面对每年的建言，“到底听进去多少，又真正改善了几多”？

        围绕当局两岸政策，工总认为，由数据来看，当前大陆不仅是台湾第一大出口市场，亦是第一大进口来源及首位对外投资地，建议台湾当局摒弃两岸对抗思维，在“求同存异”的现实基础上，以“合作”取代“对立”，为台湾多数民众谋福创利。

        工总现任理事长、同时也是台塑企业总裁的王文渊指出，过去几年，两岸关系紧张，不仅影响岛内观光、零售、饭店业及农渔蔬果产品的出口，也使得岛内外企业对投资台湾却步，2020年新任台湾领导人出炉后，应审慎思考两岸问题以及中国大陆市场。
            """
    #关键词提取
    result = getKeywords_textrank(data1)
    print(result)
