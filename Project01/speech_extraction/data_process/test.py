#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
提取数据库新闻数据
@author:Administrator
@file:test.py
@time:2019/08/28
"""
from speech_extraction.data_process import DbUtil
from conf.GetConfParams import GetConfParams


ConfParams = GetConfParams()
logger = ConfParams.logger

def getdata(num):
    try:
        dbConn = DbUtil.DataBase(ConfParams.NEWSDBNAME)
    except Exception as e:
        logger.error(e)
        errCode = {'code': 0, 'message': '系统内部错误，请联系管理员'}
        return errCode  # oracle连接错误
    try:
        sql = "SELECT id,title,content FROM {} LIMIT '{}'".format(ConfParams.DOC_TYPE_NEWS, num)
        print(sql)
        # sqlview = "update {} set viewcount=viewcount+1 where id ='{}'".format(ConfParams.DOC_TYPE_QA, qaid)
        # cur.execute(sql)
        dbConn.queryAll(sql)
        errCode = {'code': 1, 'message': '提取数据完成'}
        return errCode
    except Exception as e:
        logger.error(e)
        errCode = {'code': 0, 'message': '系统内部错误，请联系管理员'}
        return errCode
    finally:
        dbConn.closeConn()

if __name__ == '__main__':
    print(getdata(10))