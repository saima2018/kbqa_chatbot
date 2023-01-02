#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/22 13:58
# @Author  : jellyzhang
# @Email    : zhangguodong_12@126.com
# @File    : select_spo.py
# @Desp     :

import os
import json
from tqdm import tqdm
import pandas as pd
from random import shuffle
from collections import defaultdict
from multiprocessing import Pool




def random_select(arr:list,topN=1):
    '''
    列表中随机返回N条
    '''
    shuffle(arr)
    return arr[:topN]



def  gen_spo(spos_datas,s,index):
    prop_readed = defaultdict()
    root_dir = os.getcwd()
    with open(os.path.join(root_dir,'../corpus','spo_20w_{}.csv'.format(index)),'w',encoding='utf-8') as fwrite:
            for row in tqdm(spos_datas.iterrows()):
                sub = row[1]['实体']
                prop =row[1]['属性']
                obj = row[1]['值']

                if prop in s:
                     #print('匹配的属性：{}'.format(prop))
                     if prop_readed.get(prop,0)==0:
                         #print('记录三元组：{}\t{}\t{}'.format(sub,prop,obj))
                         fwrite.write('{},{},{}\n'.format(sub,prop,obj))



def filter_spo(total=200000):
    '''
    筛选质量比较好的三元组
    '''

    root_dir = os.getcwd()
    #spos
    # with open(os.path.join(root_dir,'../corpus','prop2list.txt'),'r',encoding='utf-8') as fread:
    #     for line in fread:
    #         spos=eval(line.rstrip())
    spos = pd.read_csv(os.path.join(root_dir, '../corpus', 'spo.csv'), encoding='utf-8')


    props_seg=defaultdict(list)

    prop_path=os.path.join(root_dir,'../corpus','sorted_allProp.txt')
    with open(prop_path,'r',encoding='utf-8') as fread:
        for line in fread:
            props=eval(line.rstrip())

    #根据属性频次切分成6段：（1：5%）（2-5：5%）（6-100：20%）（101-500：20%）（501-5000：25%）（5001以上：25%）

    for prop in props:
        if prop[1]<=1:
            props_seg['s1'].append(prop[0])
        elif prop[1]>=2  and prop[1]<=5:
            props_seg['s2'].append(prop[0])
        elif prop[1]>=6  and prop[1]<=100:
            props_seg['s3'].append(prop[0])
        elif prop[1]>=101  and prop[1]<=500:
            props_seg['s4'].append(prop[0])
        elif prop[1]>=501  and prop[1]<=5000:
            props_seg['s5'].append(prop[0])
        else:
            props_seg['s6'].append(prop[0])

    s1=random_select(props_seg['s1'],int(total*0.05))
    s2 = random_select(props_seg['s2'], int(total * 0.05))
    s3 = random_select(props_seg['s3'], int(total * 0.2))
    s4 = random_select(props_seg['s4'], int(total * 0.2))
    s5 = random_select(props_seg['s5'], int(total * 0.25))
    s6 = random_select(props_seg['s6'], int(total * 0.25))

    # print(s1)
    # print(s2)
    # print(s3)
    # print(s4)
    # print(s5)
    # print(s6)
    return spos,s1+s2+s3+s4+s5+s6






if __name__=='__main__':
    spos, s = filter_spo()
    len = spos.shape[0]
    count_process = 6
    count_per_process = int(len / count_process)
    pool = Pool(count_process)
    for i in range(count_process):
        pool.apply_async(gen_spo, spos.iloc[i * count_per_process:(i + 1) * count_per_process], s, i)

    pool.close()
    pool.join()


