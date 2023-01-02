#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 10:15
# @Author  : jellyzhang
# @Email    : zhangguodong_12@126.com
# @File    : tst2.py
# @Desp     :
import os
root_dir = os.getcwd()
prop_path = os.path.join(root_dir, '../corpus', 'sorted_allProp.txt')
with open(prop_path, 'r', encoding='utf-8') as fread:
    for line in fread:
        props = eval(line.rstrip())