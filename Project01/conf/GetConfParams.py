# -*- coding:utf-8 -*-
"""
功能：读取配置文件
"""
import logging
import logging.config
import os
import yaml

class GetConfParams:
    PATH = os.getcwd()  # 获取上级目录
    logging.config.fileConfig(PATH + '\conf\logging.conf')
    def __init__(self):
        PATH = os.getcwd()  # 获取上级目录
        stream = open(PATH + '\conf\conf.yaml', 'r')
        params = yaml.load(stream, Loader=yaml.FullLoader)
        self.logger = logging.getLogger('root')
        # pgsql连接地址、端口号、数据库名、用户名、密码
        self.USER = params['postgresqlParam']['USER']
        self.PWD = params['postgresqlParam']['PWD']
        self.HOST = params['postgresqlParam']['HOST']
        self.PORT = params['postgresqlParam']['PORT']
        #新闻数据库
        self.NEWSDBNAME = params['postgresqlParam']['NEWSDBNAME']
        self.DOC_TYPE_NEWS = params['postgresqlParam']['doc_type_news']