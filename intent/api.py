#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from intent.classification import model,tokenizer
import tensorflow as tf
import keras

global graph, sess
graph = tf.get_default_graph()
sess = keras.backend.get_session()

def get_intent(s:str,batch_size=1):

    token_ids, segment_ids = tokenizer.encode(s)
    with sess.as_default():
        with graph.as_default():

            y_pred=model.predict([np.array([token_ids]), np.array([segment_ids])]).argmax(axis=1)
    return  y_pred[0]

if __name__=='__main__':

    #s='北京大学校长是谁？'  #sp
    #s='叶莉和姚明是什么关系？'  #so
    s='姚明的女儿'
    print(get_intent(s))
    #y_pred = model.predict(x_true).argmax(axis=1)

