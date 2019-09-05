# coding=utf-8
#读取excle数据，分词，停用词，词性筛选，文本向量化，基于sklearn实现中文K-Means主题聚类
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

# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos and len(i.word)>1:  # 去停用词 + 词性筛选
            l.append(i.word)
    return l

'''vectorize the input documents'''
def getKeywords_tfidf(data,stopkey):
    idList, titleList, abstractList = data['id'], data['title'], data['content']
    sentences = [] # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        # text = '%s。%s' % (titleList[index], abstractList[index]) ####拼接标题和摘要
        text = '%s' % (abstractList[index])  ####一个字段摘要
        text = dataPrepos(text,stopkey) # 文本预处理
        text = " ".join(text)  # 连接成字符串，空格分隔 格式如永磁 电机 驱动 电动 大巴车 坡道
        sentences.append(text)
    # print(sentences)
    print("build train-corpus done!!")
    count_v1 = CountVectorizer(max_df=0.4, min_df=0.01)
    counts_train = count_v1.fit_transform(sentences)

    word_dict = {}
    for index, word in enumerate(count_v1.get_feature_names()):
        word_dict[index] = word

    print("the shape of train is " + repr(counts_train.shape))
    tfidftransformer = TfidfTransformer()
    tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
    return tfidf_train, word_dict


'''topic cluster'''
def cluster_kmeans(tfidf_train, word_dict, cluster_docs, cluster_keywords, num_clusters):  # K均值分类
    f_docs = open(cluster_docs, 'w+', encoding='utf-8')
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_train)
    clusters = km.labels_.tolist()
    cluster_dict = {}
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    doc = 1
    for cluster in clusters:
        f_docs.write(str(str(doc)) + ',' + str(cluster) + '\n')
        doc += 1
        if cluster not in cluster_dict:
            cluster_dict[cluster] = 1
        else:
            cluster_dict[cluster] += 1
    f_docs.close()
    cluster = 1

    f_clusterwords = open(cluster_keywords, 'w+', encoding='utf-8')
    for ind in order_centroids:  # 每个聚类选 50 个词
        words = []
        for index in ind[:20]:
            words.append(word_dict[index])
        print(cluster, ','.join(words))
        f_clusterwords.write(str(cluster) + '\t' + ','.join(words) + '\n')
        cluster += 1
        print('*****' * 5)
    f_clusterwords.close()


'''select the best cluster num'''
def best_kmeans(tfidf_matrix, word_dict):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    import numpy as np
    K = range(1, 10)
    meandistortions = []
    for k in K:
        print(k, '****' * 5)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tfidf_matrix)
        meandistortions.append(
            sum(np.min(cdist(tfidf_matrix.toarray(), kmeans.cluster_centers_, 'euclidean'), axis=1)) /
            tfidf_matrix.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')  #平均簇内平方和
    plt.title('Elbow for Kmeans clustering')
    plt.show()


if __name__ == '__main__':
    # 读取数据集
    # dataFile = 'data/2.csv'
    # data = pd.read_csv(dataFile)
    dataFile = 'E:/Pycharm/TopicCluster_master/data/corpus.xlsx'
    data = pd.read_excel(dataFile, sheet_name='Sheet1')
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('E:/Pycharm/TopicCluster_master/data/stopWord.txt', 'r', encoding='utf-8').readlines()]
    # tf-idf关键词抽取
    tfidf_train, word_dict = getKeywords_tfidf(data, stopkey)
    # corpus_train = "./corpus_train.txt"
    cluster_docs = "E:/Pycharm/TopicCluster_master/result/cluster_kmeans_document.txt"
    cluster_keywords = "E:/Pycharm/TopicCluster_master/result/cluster_kmeans_keyword.txt"
    num_clusters = 20
    best_kmeans(tfidf_train, word_dict) #挑选最好的聚类数量
    cluster_kmeans(tfidf_train, word_dict, cluster_docs, cluster_keywords, num_clusters)
