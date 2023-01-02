#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 14:40
# @Author  : jellyzhang
# @Email    : zhangguodong_12@126.com
# @File    : select_spo_lastest.py
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
                sub = row[1][':START_ID(label1)']
                prop =row[1][':TYPE']
                obj = row[1][':END_ID(label2)']

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
    indexs = list(range(spos.shape[0]))
    shuffle(indexs)


    props_seg=defaultdict(int)

    prop_path=os.path.join(root_dir,'../corpus','sorted_allProp.txt')
    with open(prop_path,'r',encoding='utf-8') as fread:
        for line in fread:
            props=eval(line.rstrip())
            
    for prop in props:
        props_seg[prop[0]]=prop[1]

    #根据属性频次切分成6段：（1：5%）（2-5：5%）（6-100：20%）（101-500：20%）（501-5000：25%）（5001以上：25%）

    s1_cnt=int(total*0.05)
    s2_cnt = int(total * 0.05)
    s3_cnt = int(total * 0.2)
    s4_cnt = int(total * 0.2)
    s5_cnt = int(total * 0.25)
    s6_cnt = int(total * 0.25)

    s1_gen,s2_gen,s3_gen,s4_gen,s5_gen,s6_gen=0,0,0,0,0,0
    with open(os.path.join(root_dir, '../corpus', 'spo_20w.csv'), 'w', encoding='utf-8') as fwrite:
        for row in spos.iloc[indexs].iterrows():
            sub=row[1][':START_ID(label1)']
            prop = row[1][':TYPE']
            obj = row[1][':END_ID(label2)']

            if props_seg[prop] <= 1:
                s1_gen+=1
                if s1_gen<s1_cnt:
                    fwrite.write('{},{},{}\n'.format(sub, prop, obj))
            elif props_seg[prop] >= 2 and props_seg[prop] <= 5:
                s2_gen += 1
                if s2_gen < s2_cnt:
                    fwrite.write('{},{},{}\n'.format(sub, prop, obj))
            elif props_seg[prop] >= 6 and props_seg[prop] <= 100:
                s3_gen += 1
                if s3_gen < s3_cnt:
                    fwrite.write('{},{},{}\n'.format(sub, prop, obj))
            elif props_seg[prop] >= 101 and props_seg[prop] <= 500:
                s4_gen += 1
                if s4_gen < s4_cnt:
                    fwrite.write('{},{},{}\n'.format(sub, prop, obj))
            elif props_seg[prop] >= 501 and props_seg[prop] <= 5000:
                s5_gen += 1
                if s5_gen < s5_cnt:
                    fwrite.write('{},{},{}\n'.format(sub, prop, obj))
            else:
                s6_gen += 1
                if s6_gen < s6_cnt:
                    fwrite.write('{},{},{}\n'.format(sub, prop, obj))








if __name__=='__main__':
    filter_spo()



