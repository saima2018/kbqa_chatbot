#!/usr/bin/env python
# -*- coding: utf-8 -*-


from simbert.simbert_baseSave import generateSimSentence

def tst_simbert():
    gss=generateSimSentence()
    print(gss.gen_synonyms('股权分置的简要描述'))



if __name__=='__main__':
    tst_simbert()