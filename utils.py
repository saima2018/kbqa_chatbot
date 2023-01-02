#!/usr/bin/env python
# -*- coding: utf-8 -*-


from elasticsearch import Elasticsearch
from elasticsearch import helpers


class ElasticSearch_Util(object):
    def __init__(self,ES_ConnectStr='http://58.56.131.11:9200',ES_USER='elastic',ES_PWD='bat100*#'):
        self.client=Elasticsearch(ES_ConnectStr, http_auth=(ES_USER,ES_PWD),)

    def insert(self,entities):
        actions = []
        for entity in entities:
            action = {'_op_type': 'index',  # 操作 index update create delete  
                      '_index': 'kgqa',  # index

                      '_source': {'content': entity.rstrip()}}
            actions.append(action)
        #print(actions)
        helpers.bulk(self.client, actions)

    def search(self,entity,size=1):
        query = {'query': {'match_phrase': {'content':{'query':entity,'slop':3}}}}
        data_es = self.client.search(index='kgqa', body=query, size=size)
        data_source = data_es['hits']['hits']
        result = {}
        result['entity'] = entity
        result['datasource'] = []
        [result['datasource'].append(data['_source']) for data in data_source]
        return result


if __name__=='__main__':

    try:
        import json
        import math
        es=ElasticSearch_Util()
        data_path='corpus/allSubjectSort.txt'
        subjects=[]
        with open(data_path,'r',encoding='utf-8') as fread:
            for line in fread:
                subjects.append(line.rstrip().split(':')[0])
                print(len(subjects))
        epoches=math.ceil(len(subjects)/1000)
        print('一共执行{}轮'.format(epoches))
        for i in range(epoches):
            #print('**开始导入es数据**')
            es.insert(subjects[1000*i:1000*(i+1)])
            print('**es数据导入1000条**')
    except Exception as ex:
        print(ex)


    #查询
    es=ElasticSearch_Util()
    print(es.search('感冒'))