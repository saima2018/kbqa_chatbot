#!/usr/bin/env python
# -*- coding: utf-8 -*-


from ner.task_sequence_labeling_ner_crf import NER
import warnings
warnings.filterwarnings('ignore')

def get_entity(sent:str):
    entities=NER.recognize(sent)
    print('api.py entities:', entities)
    return entities

if __name__=='__main__':
    s='姚明的妻子是谁'
    # print(get_entity(s))

