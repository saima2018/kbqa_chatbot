#!/usr/bin/env python
# -*- coding: utf-8 -*-


from neo4j import GraphDatabase
import time
from py2neo import Node, Relationship, Graph
import os
import re
# import chardet

neo4j_url = os.getenv('NEO4J_BASE')
if neo4j_url == 'http://keylab.jios.org:7474':
    neo4j_url = "127.0.0.1:7474"
    toInt = 'toInteger'
else:
    neo4j_url = "bolt://58.56.131.11:7474"
    toInt = 'toInteger'
print('连接neo4j数据库')
session = Graph(host='58.56.131.11', auth=("neo4j", "bat100"))
# session = Graph(neo4j_url, username="neo4j", password="bat100")
print('neo4j数据库连接完成,使用服务的地址为：%s' % (neo4j_url))


class Query(object):
    def __init__(self,):
        self.string = 'none'

    # 直接查询，根据实体和属性查询属性值，带方向
    def gen_spp_o(self):
        string = """
        match (s:Subject)-[r1]->(o:Object)-[r2]->(o2:Object) 
        where type(r1)<>'歧义关系' and type(r1)<>'歧义权重' 
        and type(r2)<>'歧义关系' and type(r2)<>'歧义权重'  
        and type(r1)<>'标签' and type(r2)<>'标签'  
        and type(r1)<>'描述' and type(r2)<>'描述'  
        return  s.name,type(r1),o.name,type(r2),o2.name  limit 200000;
        """
        rel = session.run(string).data()
        ans =[]
        for i in range(0, len(rel)):
            temp = rel[i]
            ans.append((temp['s.name'],temp['type(r1)'],temp['type(r2)']))
        return ans
    # 直接查询，根据两个属性值直接查一步关系



if __name__=='__main__':
    import os
    query=Query()
    ans=query.gen_spp_o()
    #print(ans[:10])
    root_dir = os.getcwd()
    with open(os.path.join(root_dir,'../corpus','sppo.csv'),'w',encoding='utf-8') as fwrite:
        for spp in ans:
            fwrite.write('{},{},{}\n'.format(spp[0],spp[1],spp[2]))
