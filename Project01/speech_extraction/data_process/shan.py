#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:shan.py
@time:2019/08/28
"""
import psycopg2
import pymysql
import os

def getdata1():
    print(2222222222)
    conn = pymysql.connect(db="stu_db", user="root",passwd="AI@2019@ai", host="rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com", port="3306", charset='utf8')
    print(111111111)
    cur = conn.cursor()
    sql ="SELECT id,title,content FROM news_chinese limit 10"
    print(cur(sql))

if __name__ == '__main__':
    getdata1()