#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:Incremental_train1.py
@time:2019/08/05
"""
#增量训练法一：
#本代码主要是读取本地分词后的文件，对新增大语料进行增量训练，用本方法为宜
from gensim.models.word2vec import Word2Vec
class TrainWord2Vec:
    def __init__(self, num_features=50, min_word_count=1, context=5, incremental=True,old_path='./model1/news.word2vec'):
        """
        定义变量
        :param data: 用于训练胡语料
        :param stopword: 停用词表
        :param num_features:  返回的向量长度
        :param min_word_count:  最低词频
        :param context: 滑动窗口大小
        :param incremental: 是否进行增量训练
        :param old_path: 若进行增量训练，原始模型路径
        """
        # self.data = data
        # self.stopword = stopword
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.context = context
        self.incremental = incremental
        self.old_path = old_path

    def clean_text(self):
        corpus = []
        for line in open('./data/P2_keywords.txt', 'r', encoding='utf-8'):
            text = line.strip().split(' ')
            corpus.append(text)
        print(corpus)
        return corpus

    def get_model(self, text):
        """
        从头训练word2vec模型
        :param text: 经过清洗之后的语料数据
        :return: word2vec模型
        """
        model = Word2Vec(sentences = text, size=self.num_features, min_count=self.min_word_count, window=self.context)
        return model

    def update_model(self, text):
        """
        增量训练word2vec模型
        :param text: 经过清洗之后的新的语料数据
        :return: word2vec模型
        """
        model = Word2Vec.load(self.old_path)  # 加载旧模型
        model.build_vocab(text, update=True)  # 更新词汇表
        model.train(sentences = text, total_examples=model.corpus_count, epochs=model.iter)  # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
        return model

    def main(self):
        """
        主函数，保存模型
        """
        # 加入自定义分词词表
        # jieba.load_userdict("add_word.txt")
        text = self.clean_text()
        if self.incremental:
            model = self.update_model(text)
        else:
            model = self.get_model(text)
        # 保存模型
        model.save("./model1/word2vec_new.model")
        model.wv.save_word2vec_format('./model1/word2vec_format_new1.txt')

if __name__ == '__main__':
    trainmodel = TrainWord2Vec()
    trainmodel.main()
