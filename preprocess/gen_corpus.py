#!/usr/bin/env python
# -*- coding: utf-8 -*-

#生成训练模型的语料

import os
import json
import traceback
import pandas as pd
from tqdm import  tqdm
from random import shuffle

def get_totalSubjects(filte_count=20):
    root_dir=os.getcwd()
    data_path=os.path.join(root_dir,'../corpus','allSubjectSort.txt')
    with open(data_path,'r',encoding='utf-8') as fread:
        keys=[]
        for line in fread:
            k_v=line.rstrip().split(':')
            k=k_v[0]
            if len(k_v[1:])>1:
                k=':'.format(k_v[:-1])
            v=k_v[-1]
            if int(v)>filte_count:
                keys.append(k)
        # print('过滤后的实体个数：{}'.format(key_counts))  #filte_count:16   total_subject:159887
    with open(os.path.join(root_dir,'../corpus','subject.json'),'w',encoding='utf-8') as fwrite:
        json.dump(keys,fwrite,ensure_ascii=False)

def get_totalObjects(filte_count=20):
    root_dir=os.getcwd()
    data_path=os.path.join(root_dir,'../corpus','allObjectSort.txt')
    with open(data_path,'r',encoding='utf-8') as fread:
        keys=[]
        for line in fread:
            k_v=line.rstrip().split(':')
            k=k_v[0]
            if len(k_v[1:])>1:
                k=':'.format(k_v[:-1])
            v=k_v[-1]
            if int(v)>filte_count:
                keys.append(k)
        # print('过滤后的实体个数：{}'.format(key_counts))  #filte_count:16   total_subject:159887
    with open(os.path.join(root_dir,'../corpus','object.json'),'w',encoding='utf-8') as fwrite:
        json.dump(keys,fwrite,ensure_ascii=False)

def get_totalProperties(filte_count=20):
    root_dir = os.getcwd()
    data_path = os.path.join(root_dir, '../corpus', 'allPropSort.txt')
    with open(data_path, 'r', encoding='utf-8') as fread:
        props = []
        for line in fread:
            k_v = line.rstrip().split(':')
            k = k_v[0]
            v = k_v[1]
            if int(v) > filte_count:
                props.append(k)
        # print('过滤后的实体个数：{}'.format(key_counts))  #filte_count:16   total_subject:159887
    with open(os.path.join(root_dir, '../corpus', 'properties.json'),'w',encoding='utf-8') as fwrite:
        json.dump(props, fwrite, ensure_ascii=False)


def random_select(arr:list,topN=1):
    '''
    列表中随机返回N条
    '''
    shuffle(arr)
    return arr[:topN]


def check_file(func):
    def wrapper(*args,**kwargs):
        #实体json/属性json是否存在
        root_dir = os.getcwd()
        if not os.path.exists(os.path.join(root_dir,'../corpus','subject.json')):
            get_totalSubjects()
        if not os.path.exists(os.path.join(root_dir,'../corpus','object.json')):
            get_totalObjects()
        if not os.path.exists(os.path.join(root_dir, '../corpus', 'properties.json')):
            get_totalProperties()
        return func()
    return wrapper


#@check_file
def get_spDatas(spo,corpus_cnt=100000):
    root_dir = os.getcwd()
    indexs = list(range(spo.shape[0]))
    shuffle(indexs)
    indexs_shuffled = indexs[:corpus_cnt]
    params1=['的','','的','的']
    params2 = ['','是','是谁','是什么','有啥','有哪些','有','有多大','是啥时候','是什么时候','是多少','是哪','有多少页','在哪','是多大','是多高']
    params3 =  ['','?','？']
    # results=[]
    with open(os.path.join(root_dir,'../corpus','data.txt'),'w',encoding='utf-8') as fwrite:
        for row in spo.iloc[indexs_shuffled].iterrows():
            sub=row[1][0]
            prop=row[1][1]
            s='{}{}{}{}{}\t{}\t{}\t0'.format(sub,random_select(params1)[0],prop,random_select(params2)[0],random_select(params3)[0],sub,prop)
            #print(s)
            fwrite.write('{}\n'.format(s))


#@check_file
def get_sppDatas(spp,corpus_cnt=100000):
    root_dir = os.getcwd()
    #properties = json.load(open(os.path.join(root_dir, '../corpus', 'properties.json')))
    indexs = list(range(spp.shape[0]))
    shuffle(indexs)
    indexs_shuffled = indexs[:corpus_cnt]
    params1 = ['的','','的','的']
    params2 = ['','','','', '是', '是谁', '是什么', '有啥', '有哪些', '有', '有多大', '是啥时候', '是什么时候', '是多少', '是哪', '有多少页', '在哪', '叫啥', '叫什么']
    params3 = ['', '?', '？']
    # results=[]
    with open(os.path.join(root_dir, '../corpus', 'data.txt'), 'a', encoding='utf-8') as fwrite:
        for row in spp.iloc[indexs_shuffled].iterrows():
            sub = row[1][0]
            prop1=row[1][1]
            prop2 =row[1][2]
            s = '{}{}{}{}{}{}{}\t{}\t{}\t{}\t1'.format(sub, random_select(params1)[0], prop1,random_select(params1)[0],prop2,random_select(params2)[0],
                                               random_select(params3)[0], sub, prop1,prop2)
            #print(s)
            fwrite.write('{}\n'.format(s))


#@check_file
def get_poDatas(spo,corpus_cnt=100000):
    root_dir = os.getcwd()
    indexs = list(range(spo.shape[0]))
    shuffle(indexs)
    indexs_shuffled = indexs[:corpus_cnt]
    params1 = ['谁的', '什么机构', '哪个', '哪些', '什么书籍', '什么作品', '什么', '什么人物', '啥地方']
    params2 = ['的','','的','的']
    params3 = ['', '是', '为', '在']
    params4 = ['', '?', '？']
    # results=[]
    with open(os.path.join(root_dir, '../corpus', 'data.txt'), 'a', encoding='utf-8') as fwrite:
        for row in spo.iloc[indexs_shuffled].iterrows():
            param1 = random_select(params1)[0]
            param2 = random_select(params2)[0]
            prop = row[1][1]
            param3 = random_select(params3)[0]
            obj = row[1][2]
            param4 = random_select(params4)[0]
            s = '{}{}{}{}{}{}\t{}\t{}\t2'.format(param1, param2, prop, param3,
                                                 obj, param4, prop, obj)
            #print(s)
            fwrite.write('{}\n'.format(s))

#@check_file
def get_opDatas(spo,corpus_cnt=100000):
    root_dir = os.getcwd()
    indexs=list(range(spo.shape[0]))
    shuffle(indexs)
    indexs_shuffled=indexs[:corpus_cnt]
    params1 = ['', '是', '为', '在']
    params2 = ['哪个城市', '哪个市', '哪个省', '那个地方', '哪个高校', '什么机构', '来自哪个', '谁', '啥', '什么', '哪个县', '哪部作品', '什么电影', '啥电影',
               '哪个国家', '来自', '哪些作品', '哪个学校', '哪个公司'
        , '什么公司']
    params3 =  ['的','','的','的']
    params4 = ['', '?', '？']
    with open(os.path.join(root_dir, '../corpus', 'data.txt'), 'a', encoding='utf-8') as fwrite:
        for row in spo.iloc[indexs_shuffled].iterrows():
                #print(row[1][':START_ID(label1)'], row[1][':TYPE'], row[1][':END_ID(label2)'])
                obj = row[1][0]
                param1 = random_select(params1)[0]
                param2 = random_select(params2)[0]
                param3 = random_select(params3)[0]
                prop = row[1][1]
                param4 = random_select(params4)[0]
                s = '{}{}{}{}{}{}\t{}\t{}\t3'.format(obj, param1,param2, param3,prop,
                                                              param4, obj,prop)
                fwrite.write('{}\n'.format(s))



    # params1 = ['','是','为','在']
    # params2 = ['哪个城市','哪个市','哪个省','那个地方','哪个高校','什么机构','来自哪个','谁','啥','什么','哪个县','哪部作品','什么电影','啥电影','哪个国家','来自','哪些作品','哪个学校','哪个公司'
    # ,'什么公司']
    # params3 = ['','的']
    # params4 = ['', '?', '？']
    # with open(os.path.join(root_dir, '../corpus', 'data.txt'), 'a', encoding='utf-8') as fwrite:
    #     for i in tqdm(range(corpus_cnt)):
    #         obj = random_select(objects)[0]
    #         param1 = random_select(params1)[0]
    #         param2 = random_select(params2)[0]
    #         param3 = random_select(params3)[0]
    #         prop = random_select(properties)[0]
    #         param4 = random_select(params4)[0]
    #         s = '{}{}{}{}{}{}\t{}\t{}\t3'.format(obj, param1,param2, param3,prop,
    #                                              param4, obj,prop)
    #         #print(s)
    #         fwrite.write('{}\n'.format(s))

#@check_file
def get_soDatas(spo,corpus_cnt=100000):
    root_dir = os.getcwd()
    indexs = list(range(spo.shape[0]))
    shuffle(indexs)
    indexs_shuffled = indexs[:corpus_cnt]
    params1 = ['','和','与','和','与']
    params2 = ['是啥关系','是什么关系','关系是啥','有啥关系','有什么关系','关系名是啥','的关系','']
    params3 = ['', '?', '？']
    # results=[]
    with open(os.path.join(root_dir, '../corpus', 'data.txt'), 'a', encoding='utf-8') as fwrite:
        for row in spo.iloc[indexs_shuffled].iterrows():
            sub = row[1][0]
            param1=random_select(params1)[0]
            obj = row[1][2]
            param2 = random_select(params2)[0]
            param3=random_select(params3)[0]
            s = '{}{}{}{}{}\t{}\t{}\t4'.format(sub,param1,obj,param2,param3,sub,obj)
            #print(s)
            fwrite.write('{}\n'.format(s))


def gen_sceneCorpus():
    '''
    根据场景产生训练语料
    '''
    try:
        root_dir = os.getcwd()
        spo = pd.read_csv(os.path.join(root_dir, '../corpus', 'spo_20w.csv'), encoding='utf-8')
        print('start gening the  corpus...')
        print('start gening  sp ...')
        get_spDatas(spo)
        print('end gening  sp ...')

        print('start gening  spp ...')
        spp = pd.read_csv(os.path.join(root_dir, '../corpus', 'sppo.csv'), encoding='utf-8')
        get_sppDatas(spp)
        print('end gening  spp ...')

        print('start gening  po ...')
        get_poDatas(spo)
        print('end gening  po ...')

        print('start gening  op ...')
        get_opDatas(spo)
        print('end gening  op ...')
        #
        print('start gening  so ...')
        get_soDatas(spo)
        print('end gening  so ...')

        print('end gening the  corpus...')
        pass
    except Exception as ex:
        traceback.print_exc()



if __name__=='__main__':
    gen_sceneCorpus()


