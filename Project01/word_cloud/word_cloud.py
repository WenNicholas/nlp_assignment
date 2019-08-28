#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:word_cloud.py
@time:2019/08/14
"""
import os
from os import path
import numpy as np
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
from matplotlib import pyplot as plt
import jieba
import re
# from cut_word import getCutWord

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
stopword = [line.rstrip() for line in open("E:\pycharmproject\my_project\worddict\stopword.txt", 'r', encoding='utf-8')]
def clean_text(text):
    newtext = []
    text = re.sub(r'\d+', ' ', text) #去除数字
    text = jieba.lcut(text)  # 分词
    for word in text:
        if word not in stopword:  # 去停用词 + 词性筛选
            newtext.append(word)
    lineswords=' '.join(newtext)
    return lineswords


def wc_chinese(text):
    text = clean_text(text)
    print(text)
    # 设置中文字体
    font_path = 'D:\Fonts\simkai.ttf'
    # 读取背景图片
    background_Image = np.array(Image.open(path.join(d, "E:\pycharmproject\my_project\data\\1.jpg")))
    # 提取背景图片颜色
    img_colors = ImageColorGenerator(background_Image)
    # 设置中文停止词
    stopwords = set('')
    wc = WordCloud(
            font_path = font_path, # 中文需设置路径 字体设置
            margin = 2, # 页面边缘
            mask = background_Image,
            scale = 2,
            max_words = 200, # 最多词个数
            min_font_size = 4, #
            stopwords = stopwords,
            random_state = 42,
            background_color = 'white', # 背景颜色
            # background_color = '#C3481A', # 背景颜色
            max_font_size = 100,
            )
    wc.generate(text)
    # 获取文本词排序，可调整 stopwords
    # process_word = WordCloud.process_text(wc,text)
    # sort = sorted(process_word.items(),key=lambda e:e[1],reverse=True)
    # print(sort[:50]) # 获取文本词频最高的前50个词
    # 设置为背景色，若不想要背景图片颜色，就注释掉
    wc.recolor(color_func=img_colors)
    # 存储图像
    wc.to_file('E:\pycharmproject\my_project\data\heart.png')
    # 显示图像
    plt.imshow(wc,interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    string = '台湾工业总会是岛内最具影响力的工商团体之一，2008年以来，该团体连续12年发表对台当局政策的建言白皮书，集中反映岛内产业界的呼声。  台湾工业总会指出，2015年的白皮书就特别提到台湾面临“五缺”（缺水、缺电、缺工、缺地、缺人才）困境，使台湾整体投资环境走向崩坏。然而四年过去，“五缺”未见改善，反而劳动法规日益僵化、两岸关系陷入紧张、对外关系更加孤立。该团体质疑，台当局面对每年的建言，“到底听进去多少，又真正改善了几多”？  围绕当局两岸政策，工总认为，由数据来看，当前大陆不仅是台湾第一大出口市场，亦是第一大进口来源及首位对外投资地，建议台湾当局摒弃两岸对抗思维，在“求同存异”的现实基础上，以“合作”取代“对立”，为台湾多数民众谋福创利。  工总现任理事长、同时也是台塑企业总裁的王文渊指出，过去几年，两岸关系紧张，不仅影响岛内观光、零售、饭店业及农渔蔬果产品的出口，也使得岛内外企业对投资台湾却步，2020年新任台湾领导人出炉后，应审慎思考两岸问题以及中国大陆市场。'
    wc_chinese(string)