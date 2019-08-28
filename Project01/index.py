#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:index.py
@time:2019/08/19
"""

import os
import json
from flask_cors import *
from flask import Flask,request
from speech_extraction import speechExtract
from keyword_extraction import keywordsExtract
from semantic_sim import semanticSim
from word_cloud import word_statics
from conf.GetConfParams import GetConfParams

app = Flask(__name__)
CORS(app, supports_credentials=True)

PATH = os.getcwd()

ConfParams = GetConfParams()
logger = ConfParams.logger# 日志记录函数

###新闻言论提取
@app.route('/System/speechExtract', methods=['POST'])
def get_person_speechExtract():
    try:
        prefix = request.json.get('prefix')
        result=speechExtract.del_sentences(prefix)
        res = {'code': 1, 'message': '数据获取成功', 'data': result}
    except Exception as e:
        logger.error(e)
        res = {'code': 0, 'message': '系统内部错误，请联系管理员', 'data': ''}
    return json.dumps(res, ensure_ascii=False,indent=4)


###关键词提取
@app.route('/System/keywordExtract', methods=['POST'])
def get_keyword_Extract():
    try:
        prefix = request.json.get('prefix')
        result = keywordsExtract.getKeywords_textrank(prefix)
        res = {'code': 1, 'message': '数据获取成功', 'data': result}
    except Exception as e:
        logger.error(e)
        res = {'code': 0, 'message': '系统内部错误，请联系管理员', 'data': ''}
    return json.dumps(res, ensure_ascii=False, indent=4)


###语义联想
@app.route('/System/semanticSimilarity', methods=['POST'])
def get_semanticSimlarity():
    try:
        prefix = request.json.get('prefix')
        result = semanticSim.semantic_Similarity(prefix)
        res = {'code': 1, 'message': '数据获取成功', 'data': result}
    except Exception as e:
        logger.error(e)
        res = {'code': 0, 'message': '系统内部错误，请联系管理员', 'data': ''}
    return json.dumps(res, ensure_ascii=False, indent=4)


###词云
@app.route('/System/wordCloud', methods=['POST'])
def get_wordCloud():
    try:
        prefix = request.json.get('prefix')
        result = word_statics.cut_word(prefix)
        res = {'code': 1, 'message': '数据获取成功', 'data': result}
    except Exception as e:
        logger.error(e)
        res = {'code': 0, 'message': '系统内部错误，请联系管理员', 'data': ''}
    return json.dumps(res, ensure_ascii=False, indent=4)



######合并
@app.route('/System/allExtract', methods=['POST'])
def get_keyword_Extract1():
    try:
        prefix = request.json.get('prefix')
        result1 = speechExtract.del_sentences(prefix)
        result2 = keywordsExtract.getKeywords_textrank(prefix)
        result3 = semanticSim.semantic_Similarity(prefix)
        result4 = word_statics.cut_word(prefix)
        res = {'code': 1, 'message': '数据获取成功', 'data1': result1, 'data2': result2, 'data3': result3,'data4': result4}
        # print(res)
    except Exception as e:
        logger.error(e)
        res = {'code': 0, 'message': '系统内部错误，请联系管理员', 'data': ''}
    return json.dumps(res, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8056)
    # string = '国台办表示中国必然统一。会尽最大努力争取和平统一，但绝不承诺放弃使用武力。'
    # result1 = keyword_Extract1(string)
    # print(result1)