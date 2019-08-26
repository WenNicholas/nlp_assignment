#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:text_classification.py
@time:2019/08/23
"""
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics  #计算准确率 精度 召回率 F值
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_extractors import bow_extractor, tfidf_extractor

warnings.filterwarnings('ignore')

stop_words = [line.strip() for line in open("text classification/stop/stopword.txt", 'r', encoding='utf-8-sig')]

#第一阶段，读取数据 新华社新闻8133 其他新闻社8133
def get_data():
    '''
    获取数据
    :return: 文本数据，对应的labels
    '''
    with open("text classification/data/ham_data.txt", encoding="utf-8-sig") as ham_f, open("text classification/data/spam_data.txt", encoding="utf-8-sig") as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()  #创建一个列表8133个1.0 [1.0,1.0,1.0，……]
        spam_label = np.zeros(len(spam_data)).tolist()  #创建一个列表8133个0.0 [0.0,0.0,0.0，……]
        #1.0代表新华社，0.0代表其他新闻社
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels

#过滤空文本，但本数据集中没有空文本
def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label) #len(filtered_corpus)=16266
    return filtered_corpus, filtered_labels

#划分数据集：将数据分为训练集和测试集
def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    '''
    :param corpus: 文本数据
    :param labels: label数据
    :param test_data_proportion:测试数据占比
    :return: 训练数据,测试数据，训练label,测试label
    '''
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y

#数据预处理：分词、去数字、特殊字符（停用词后面处理去除）
def deal_corpus(text):
    text_with_spaces = ''
    text = re.sub(r'\d+', ' ', text)  # 去除数字
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    # print(text_with_spaces)
    return text_with_spaces



#计算得分
def get_metrics(true_labels, predicted_labels):
    print('准确率:', np.round(metrics.accuracy_score(true_labels, predicted_labels), 2))
    print('精度:', np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),2))
    print('召回率:', np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),2))
    print('F1得分:', np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),2))


def train_predict_evaluate_model(classifier,train_features, train_labels,test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # 用模型预测
    predictions = classifier.predict(test_features)
    # 评估模型效果
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions



def main():
    corpus, labels = get_data()  # 获取数据集
    print("总的数据量:", len(labels))

    corpus, labels = remove_empty_docs(corpus, labels)

    print('样本之一:', corpus[10])
    print('样本的label:', labels[10])
    label_name_map = ["其他新闻报道单位", "新华社"]
    print('实际类型:', label_name_map[int(labels[10])], label_name_map[int(labels[20])]) #labels[0:8133]为1.0，labels[8133:16266]为0.0
    print('实际类型:', label_name_map[1], label_name_map[0])

    # 对数据进行划分
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,labels,test_data_proportion=0.3)

    #数据预处理
    norm_train_corpus = []
    for text in train_corpus:
        norm_train_corpus.append(deal_corpus(text))
    norm_test_corpus = []
    for text in test_corpus:
        norm_test_corpus.append(deal_corpus(text))
    print(len(norm_train_corpus),len(train_labels))
    print(len(norm_test_corpus), len(test_labels))


#朴素贝叶斯
    # 计算单词权重
    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

    train_features = tf.fit_transform(norm_train_corpus)
    # 上面fit过了，这里transform
    test_features = tf.transform(norm_test_corpus)

    # 多项式贝叶斯分类器
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)

    # 计算准确率
    print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))

    print('*'*90)

    # 词袋模型特征
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    # tfidf 特征
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)


    # 训练分类器
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    mnb = MultinomialNB()  # 朴素贝叶斯
    svm = SGDClassifier(loss='hinge', n_iter=100)  # 支持向量机
    lr = LogisticRegression()  # 逻辑回归
    knn = KNeighborsClassifier()

    # 基于词袋模型的KNN模型
    print("基于词袋模型特征的KNN模型")
    mnb_bow_predictions = train_predict_evaluate_model(classifier=knn,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)
    # 基于词袋模型的多项朴素贝叶斯
    print("基于词袋模型特征的贝叶斯分类器")
    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)
    # print(mnb_bow_predictions)  #返回的预测结果：[0. 0. 1. ... 0. 1. 0.]
    # 基于词袋模型特征的逻辑回归
    print("基于词袋模型特征的逻辑回归")
    lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                      train_features=bow_train_features,
                                                      train_labels=train_labels,
                                                      test_features=bow_test_features,
                                                      test_labels=test_labels)

    # 基于词袋模型的支持向量机方法
    print("基于词袋模型的支持向量机")
    svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)
    print('*'*100)
    # 基于tfidf的KNN模型
    print("基于tfidf的KNN模型")
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=knn,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)

    # 基于tfidf的多项式朴素贝叶斯模型
    print("基于tfidf的贝叶斯模型")
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)
    # 基于tfidf的逻辑回归模型
    print("基于tfidf的逻辑回归模型")
    lr_tfidf_predictions = train_predict_evaluate_model(classifier=lr,
                                                        train_features=tfidf_train_features,
                                                        train_labels=train_labels,
                                                        test_features=tfidf_test_features,
                                                        test_labels=test_labels)

    # 基于tfidf的支持向量机模型
    print("基于tfidf的支持向量机模型")
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)
    print('*' * 100)
    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        if label == 0 and predicted_label == 0:
            print('新闻报道单位:', label_name_map[int(label)])
            print('预测的新闻报道单位:', label_name_map[int(predicted_label)])
            print('文本:-')
            print(re.sub('\n', ' ', document))

            num += 1
            if num == 4:
                break
    # 部分分错邮件
    print('*' * 100)
    print("************部分涉嫌抄袭新闻:************")
    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        if label == 0 and predicted_label == 1:
            print('新闻报道单位:', label_name_map[int(label)])
            print('预测的新闻报道单位:', label_name_map[int(predicted_label)])
            print('文本:-')
            print(re.sub('\n', ' ', document))

            num += 1
            if num == 4:
                break




if __name__ == '__main__':
    main()