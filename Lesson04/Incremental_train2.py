#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:Incremental_train2.py
@time:2019/08/06
"""
#增量训练法二：
#本代码主要是进行词向量增量训练，可以直接对新的语料实现在线分词并进行增量训练，不过这里也把分词后的文本存储到了本地。是否进行增量训练通过设置参数 incremental
#一般小语料可以在线分词训练，对于大语料，建议存储到本地，再读取进行训练为宜
import jieba
import re
from gensim.models.word2vec import Word2Vec

class TrainWord2Vec:
    """
    训练得到一个Word2Vec模型
    @author:xiaozhu
    @time:2018年10月12日
    """
    def __init__(self, num_features=50, min_word_count=1, context=4, incremental=False,old_path = './model1/news.word2vec'):
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
        """
        对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
        """
        #读取停用词
        stop_words = []
        with open('./data/stopwords.txt',"r",encoding="utf-8") as f:
            line = f.readline()
            while line:
                stop_words.append(line[:-1])
                line = f.readline()
        stop_words = set(stop_words)
        print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

        # 读取文本，预处理，分词，得到词典
        raw_word_list = []
        rules =  u"([\u4e00-\u9fff]+)" #只提取中文
        pattern =  re.compile(rules)
        with open('./data/P2.txt',"r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\r","").replace("\n","").strip()
                if line == "" or line is None:
                    continue
                line = ' '.join(jieba.cut(line))
                seg_list = pattern.findall(line)
                words_list = []
                for word in seg_list:
                    if word not in stop_words:
                        words_list.append(word)
                with open('./data/分词后的文本.txt', 'a', encoding='utf-8') as ff:
                    if len(words_list) > 0:
                        line = ' '.join(words_list) + "\n"
                        ff.write(line) # 词汇用空格分开
                        ff.flush()
                        raw_word_list.append(words_list)
        print(raw_word_list)
        return raw_word_list

    def get_model(self, text):
        """
        从头训练word2vec模型
        :param text: 经过清洗之后的语料数据
        :return: word2vec模型
        """
        model = Word2Vec(text, size=self.num_features, min_count=self.min_word_count, window=self.context)
        return model

    def update_model(self, text):
        """
        增量训练word2vec模型
        :param text: 经过清洗之后的新的语料数据
        :return: word2vec模型
        """
        model = Word2Vec.load(self.old_path)  # 加载旧模型
        model.build_vocab(text, update=True)  # 更新词汇表
        model.train(text, total_examples=model.corpus_count, epochs=model.iter)  # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
        return model

    def main(self):
        """
        主函数，保存模型
        """
        # 加入自定义分析词库
        # jieba.load_userdict("add_word.txt")
        text = self.clean_text()
        if self.incremental:
            model = self.update_model(text)
        else:
            model = self.get_model(text)
        # 保存模型
        model.save("./model1/word2vec_new3.model")
        model.wv.save_word2vec_format('./model1/word2vec_format_new3.txt')

if __name__ == '__main__':
    trainmodel = TrainWord2Vec()
    trainmodel.main()
