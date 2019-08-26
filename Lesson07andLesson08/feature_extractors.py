#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:feature_extractors.py
@time:2019/08/23
"""
from sklearn.feature_extraction.text import CountVectorizer

def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())  #词表是以逗号、句号分割的
    return vectorizer, features


from sklearn.feature_extraction.text import TfidfTransformer

def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus) #每一个列表元素中的词语的词频（权重）一样
    return vectorizer, features





