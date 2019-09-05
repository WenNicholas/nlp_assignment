# coding=utf-8
#读取excle数据，分词，词性筛选，文本向量化，基于Gensim实现LDA、LSI中文主题聚类
import os, sys
import importlib
importlib.reload(sys)
from gensim.models import LdaModel, TfidfModel, LsiModel
from gensim import similarities
from gensim import corpora
import sys,codecs
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse

# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos and len(i.word)>1:  # 去停用词 + 词性筛选
            l.append(i.word)
    return l

# #构建数据，先后使用doc2bow和tfidf model对文本进行向量表示
def getKeywords_tfidf(data,stopkey):
    idList, titleList, abstractList = data['id'], data['title'], data['content']
    sentences = [] # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        # text = '%s。%s' % (titleList[index], abstractList[index]) ####拼接标题和摘要
        text = '%s' % (abstractList[index])  ####一个字段摘要
        text = dataPrepos(text,stopkey) # 文本预处理
        sentences.append(text)
    # 对文本进行处理，得到文本集合中的词表
    dictionary = corpora.Dictionary(sentences)
    print(dictionary)
    # 利用词表，对文本进行cbow表示
    corpus = [dictionary.doc2bow(text) for text in sentences]
    # print(corpus)
    # 利用cbow，对文本进行tfidf表示
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return dictionary, corpus, corpus_tfidf

#lda模型，获取主题分布
def lda_model(dictionary, corpus, corpus_tfidf, cluster_keyword_lda):  # 使用lda模型，获取主题分布
    lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=20)
    f_keyword = open(cluster_keyword_lda, 'w+',encoding='utf-8')
    for topic in lda.print_topics(20, 20):
        print('****' * 5)
        words = []
        for word in topic[1].split('+'):
            word = word.split('*')[1].replace(' ', '')
            words.append(word)
        f_keyword.write(str(topic[0]) + '\t' + ','.join(words) + '\n')
    # 利用lsi模型，对文本进行向量表示，这相当于与tfidf文档向量表示进行了降维，维度大小是设定的主题数目
    corpus_lda = lda[corpus_tfidf]
    for doc in corpus_lda:
        print(len(doc), doc)
    return lda

#lsi模型，获取主题分布
def lsi_model(dictionary, corpus, corpus_tfidf, cluster_keyword_lsi):  # 使用lsi模型，获取主题分布
    lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=20)
    f_keyword = open(cluster_keyword_lsi, 'w+',encoding='utf-8')
    for topic in lsi.print_topics(20, 20):
        print(topic[0])
        words = []
        for word in topic[1].split('+'):
            word = word.split('*')[1].replace(' ', '')
            words.append(word)
        f_keyword.write(str(topic[0]) + '\t' + ','.join(words) + '\n')
    return lsi


if __name__ == "__main__":
    # 读取数据集
    # dataFile = 'data/2.csv'
    # data = pd.read_csv(dataFile)
    dataFile = 'E:/Pycharm/TopicCluster_master/data/corpus.xlsx'
    data = pd.read_excel(dataFile, sheet_name='Sheet1')
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('E:/Pycharm/TopicCluster_master/data/stopWord.txt', 'r', encoding='utf-8').readlines()]
    # tf-idf关键词抽取
    dictionary, corpus, corpus_tfidf = getKeywords_tfidf(data, stopkey)
    cluster_keyword_lda = 'E:/Pycharm/TopicCluster_master/result/cluster_keywords_lda1.txt'
    cluster_keyword_lsi = 'E:/Pycharm/TopicCluster_master/result/cluster_keywords_lsi1.txt'
    lsi_model(dictionary, corpus, corpus_tfidf, cluster_keyword_lsi)
    lda_model(dictionary, corpus, corpus_tfidf, cluster_keyword_lda)

