from neo4j import GraphDatabase
import time
from py2neo import Node, Relationship, Graph
import os
import re
# import chardet
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

neo4j_url = os.getenv('NEO4J_BASE')
if neo4j_url == 'http://keylab.jios.org:7474':
    neo4j_url = "192.168.1.111:7474"
    toInt = 'toInteger'
else:
    neo4j_url = "bolt://192.168.1.111:7474"
    toInt = 'toInteger'
print('连接neo4j数据库')
session = Graph(host='192.168.1.111', auth=("username", "password"))
print('neo4j数据库连接完成,使用服务的地址为：%s' % (neo4j_url))


class Query(object):
    def __init__(self,):
        self.string = 'none'

    # 直接查询，根据实体和属性查询属性值，带方向
    def SP_O(self, s, p):
        string = "MATCH (a:Subject{name:'%s'})-[r:`%s`]->(b) RETURN b.name" % (
            s, p)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['b.name']
        return ans
    # 直接查询，根据两个属性值直接查一步关系

    def SO_P(self, s, o):
        string = "MATCH (a:Subject{name:'%s'})-[r]->(b:Object{name:'%s'}) RETURN r " % (
            s, o)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['r.name']
        return ans

    # 关系查询，查询与实体相关的关系，以列表形式返回
    def S_P(self, s):
        string = "MATCH (n:Subject{name:'%s'})-[r]->(m) RETURN type(r),m.name" % (
            s)
        # rel = session.run(string).data()
        try:
            rel = session.run(string).data()
        except:
            pass
        return rel

    # 直接查询，根据实体+属性1+属性2，直接查询结果，字典形式返回
    def SPP_O(self, s1, p1, p2):
        string = "MATCH (a:Subject{name:'{}'})-[r1:`{}`]->(b:Object), (b:Subject)-[r2:`{}`}]->(c:Object) RETURN c.name".format(
            s1, p1, p2)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['c.name']
        # print(ans)
        return ans

    # 查询实体相关的歧义关系，带有方向
    def qiyiByEntity(self, s):
        p_qiyi = '歧义关系'
        string = "MATCH (a:Subject{name:'{}'})-[r:`{}`]->(b:Object) RETURN b.name".format(
            s, p_qiyi)
        rel = session.run(string).data()
        qiyi_entity_list = []
        for i in range(len(rel)):
            temp = rel[i]
            qiyi_entity_list.append(temp['b.name'])
        if s not in qiyi_entity_list:
            qiyi_entity_list.insert(0, s)
        return qiyi_entity_list

    # 查询实体的歧义实体，并根据歧义实体的权重对实体进行排序，返回歧义实体的所有属性和属性值
    def Q_Z_ByEntity(self, s):
        rel = []
        # modified cypher query due to different neo4j node property initialisation
        # string = "match (m:Entity)-[r:Relation{name:'歧义关系'}]->(n:Entity)-[r1:Relation{name:'歧义权重'}]->(n1:Entity) where m.name='%s' return m.name,r.name,n.name, %s(n1.name) as weight order by  %s(n1.name) desc"%(s,toInt,toInt)
        string = "match (m:Subject)-[r:`歧义关系`]->(n:Object)-[r1:`歧义权重`]->(n1:Object), (n:Subject)-[r2]->(n2) where m.name='%s' return m.name,type(r) as r,n.name, %s(n1.name) as weight,type(r2) as r2, n2.name order by  %s(n1.name) desc" % (s, toInt, toInt)
        try:
            rel = session.run(string).data()
        except:
            pass
        # print('rrrrr',rel)
        return rel
    # 属性值的前属性查询，返回与属性值相关的属性和实体

    def OP_S_ByEntity(self, s):
        rel = []
        string = "match (m:Subject)-[r]->(n:Object{name:'%s'}) return m.name,type(r)" % (
            s)
        try:
            rel = session.run(string).data()
        except:
            pass
        return rel
    # 根据属性和属性值，直接查询实体s,字典形式返回

    def OP_S(self, p, o):
        rel = []
        string = "match (m:Subject)-[r:`{}`]->(n:Object{name:'{}'}) return m.name".format(
            p, o)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['m.name']
        return ans

    # 最小路径查询，查询两个节点之间五步关系的最小路径
    def P_By_SO(self, s, o):
        rel = []
        string = "match (m:Subject{name:'%s'}),(n:Object{name:'%s'}),p=shortestpath((m)-[*..5]-(n)) return p" % (
            s, o)
        rel = session.run(string).data()
        if len(rel)>0:
            ans = rel[0]['p']
            ans = str(ans).replace(':','').replace('{}', '')
            ans_list = ans.split('-')
            for i in range(len(ans_list)):
                if 'u' in ans_list[i]:
                    ans_list[i] = ans_list[i].encode().decode('unicode_escape')
            res = '-'.join(ans_list)
            return res
        else:
            return ''


# q = Query()
