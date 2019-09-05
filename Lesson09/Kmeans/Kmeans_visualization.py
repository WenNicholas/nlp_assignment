#!/usr/bin/python
# coding=utf-8
# 聚类可视化#读取excle数据，分词，停用词，词性筛选，文本向量化，基于sklearn实现中文K-Means主题聚类，并可视化
#参考链接：https://www.jianshu.com/p/622222b96f76
import os, sys
import importlib
import codecs
importlib.reload(sys)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

"""
       TF-IDF权重：
           1、CountVectorizer 构建词频矩阵
           2、TfidfTransformer 构建tfidf权值计算
           3、文本的关键字
           4、对应的tfidf矩阵
"""
# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    return l

# tf-idf……
def getKeywords_tfidf(data,stopkey):
    idList, titleList, abstractList = data['id'], data['title'], data['content']
    corpus = [] # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        # text = '%s。%s' % (titleList[index], abstractList[index]) ####拼接标题和摘要
        text = '%s' % (abstractList[index])  ####一个字段摘要
        text = dataPrepos(text,stopkey) # 文本预处理
        text = " ".join(text) # 连接成字符串，空格分隔 如永磁 电机 驱动 电动 大巴车 坡道
        corpus.append(text)
    print(corpus)
    '''
        2、计算tf-idf设为权重
    '''

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频 (0, 140)    1  (第？文本，第？个词) 出现？次
    # print(X)
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # print(tfidf)
    ''' 
        3、获取词袋模型中的所有词语特征
        如果特征数量非常多的情况下可以按照权重降维
    '''
    word = vectorizer.get_feature_names()
    print("word feature length: {}".format(len(word)))

    ''' 
        4、导出权重，到这边就实现了将文字向量化的过程，矩阵中的每一行就是一个文档的向量表示
    '''
    tfidf_weight = tfidf.toarray()

    '''
        5、对向量进行聚类
    '''
    # 指定分成7个类
    kmeans = KMeans(n_clusters=15)
    kmeans.fit(tfidf_weight)

    # 打印出各个族的中心点
    print(kmeans.cluster_centers_)
    for index, label in enumerate(kmeans.labels_, 1):
        print("index: {}, label: {}".format(index, label))

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means的超参数n_clusters可以通过该值来评估
    print("inertia: {}".format(kmeans.inertia_))

    '''
        6、可视化
    '''
    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(tfidf_weight)
    x = []
    y = []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=kmeans.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig('./sample.png', aspect=1)



if __name__ == '__main__':
    # 读取数据集
    # dataFile = 'data/2.csv'
    # data = pd.read_csv(dataFile)
    dataFile = 'E:/Pycharm/TopicCluster_master/data/corpus.xlsx'
    data = pd.read_excel(dataFile, sheet_name='Sheet1')
    # 停用词表
    stopkey = [w.strip() for w in
               codecs.open('E:/Pycharm/TopicCluster_master/data/stopWord.txt', 'r', encoding='utf-8').readlines()]
    # tf-idf关键词抽取
    getKeywords_tfidf(data, stopkey)