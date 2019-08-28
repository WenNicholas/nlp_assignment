#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Administrator
@file:DbUtil.py
@time:2019/08/28
"""

import psycopg2
from DBUtils.PooledDB import PooledDB
import math
from conf.GetConfParams import GetConfParams

ConfParams = GetConfParams()
logger = ConfParams.logger

class DataBase():
    #连接
    def __init__(self, match):
        try:
            self.psycopg_pool = PooledDB(psycopg2, mincached=5, blocking=True, user=ConfParams.USER,
                                    password=ConfParams.PWD, database=match, host=ConfParams.HOST,
                                    port=ConfParams.PORT)
            self.connection = self.psycopg_pool.connection()
            self.cursor = self.connection.cursor()
        except Exception as e:
            logger.error('数据库连接错误:%s'%e)
            raise

    #取单条数据
    def queryOne(self,sql,flag=False):
        cur = self.cursor
        cur.execute(sql)
        if flag==True:
            return cur.fetchone(),cur.description
        elif flag==False:
            return cur.fetchone()

    #取多条数据
    def queryAll(self,sql,flag=False):
        cur = self.cursor
        cur.execute(sql)
        if flag==True:
            print(cur.fetchall(),cur.description)
            return cur.fetchall(),cur.description
        elif flag==False:
            print(cur.fetchall())
            return cur.fetchall()


    #插入/更新数据
    def insertSql(self,sql,com=True):
        try:
            cur = self.cursor
            cur.execute(sql)
            if com:
                self.connection.commit()
        except Exception as e:
            logger.error(e)
            self.connection.rollback()
            raise

    '''
    批量插入 
    data：为dataframe数据，size：为批量大小
    sql示例：  "insert into table(username,password,userid) values(%s,%s,%s)"
    '''
    def batchInsert(self,sql, data, size):
        try:
            cur = self.cursor
            cycles = math.ceil(data.shape[0] / size)
            for i in range(cycles):
                val = data[i * size:(i + 1) * size].values
                cur.executemany(sql, val)
                self.connection.commit()
        except Exception as e:
            logger.error(e)
            self.connection.rollback()
            raise

    # 删除数据
    def deleteSql(self, sql, com=True):
        try:
            cur = self.cursor
            cur.execute(sql)
            if com:
                self.connection.commit()
        except Exception as e:
            logger.error(e)
            self.connection.rollback()
            raise

    # 提交
    def commitConn(self):
        try:
            self.connection.commit()
        except Exception as e:
            logger.error(e)
            self.connection.rollback()
            raise

    #关闭连接
    def closeConn(self):
        self.connection.close()