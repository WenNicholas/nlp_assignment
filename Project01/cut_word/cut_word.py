#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:cut_word.py
@time:2019/08/14
"""

import jieba
import jieba.analyse
import jieba.posseg

# jieba.load_userdict("./worddict/200W-jiebauserdict.txt")
stopword = [line.rstrip() for line in open("E:\pycharmproject\my_project\worddict\stopword.txt", 'r', encoding='utf-8')]
def getCutWord(q):
    try:
        lineword = []
        cutword = []
        if len(q)<1:
            print('请输入数据')
        else:
            items = jieba.lcut(q)
            for item in items:
                lineword.append(item)
            cutword.append('|'.join(x for x in lineword))
            if lineword:
                return cutword
    except  Exception as e:
        print("系统内部错误")

if __name__ == '__main__':
    string = '毛主席在天安门城楼上宣布中华人民共和国今天成立了！'
    print(getCutWord(string))