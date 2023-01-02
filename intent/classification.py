#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import warnings
warnings.filterwarnings('ignore')
set_gelu('tanh')  # 切换gelu版本

num_classes =5
maxlen = 128
batch_size = 32
dir_name=os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(dir_name,'albert_small_zh_google/albert_config.json')
checkpoint_path = os.path.join(dir_name,'albert_small_zh_google/albert_model.ckpt')
dict_path = os.path.join(dir_name,'albert_small_zh_google/vocab.txt')


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if len(l.strip().split('\t'))==2:
                text, label = l.strip().split('\t')
                D.append((text, int(label)))
            else:
                pass

    return D

# if not os.path.exists('datasets/train.txt'): #数据集没有切割，则先进行数据切割
#     from random import shuffle
#     datas=[]
#     with open('datasets/data.txt','r',encoding='utf-8') as fread:
#         for line in fread:
#             tmp=line.rstrip().split('\t')
#             datas.append('{}\t{}'.format(tmp[0],tmp[-1]))
#         shuffle(datas)
#     train_cnt=int(len(datas)*0.8)
#     valid_cnt=int(len(datas)*0.1)
#     with open('datasets/train.txt','w',encoding='utf-8') as f_train:
#         [f_train.write('{}\n'.format(data)) for data in datas[:train_cnt]]
#     with open('datasets/valid.txt', 'w', encoding='utf-8') as f_train:
#         [f_train.write('{}\n'.format(data)) for data in datas[train_cnt:train_cnt+valid_cnt]]
#     with open('datasets/test.txt', 'w', encoding='utf-8') as f_train:
#         [f_train.write('{}\n'.format(data)) for data in datas[train_cnt+valid_cnt:]]



# 加载数据集
train_data = load_data(os.path.join(dir_name,'datasets/train.txt'))
valid_data = load_data(os.path.join(dir_name,'datasets/valid.txt'))
test_data = load_data(os.path.join(dir_name,'datasets/test.txt'))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

# model = keras.models.Model(bert.model.input, output)
# model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
# AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
#
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     # optimizer=Adam(1e-5),  # 用足够小的学习率
#     optimizer=AdamLR(lr=1e-4, lr_schedule={
#         1000: 1,
#         2000: 0.1
#     }),
#     metrics=['accuracy'],
# )

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

def get_intent(s:str,batch_size=1):

    token_ids, segment_ids = tokenizer.encode(s)

    y_pred=model.predict([np.array([token_ids]), np.array([segment_ids])]).argmax(axis=1)
    return  y_pred[0]

if __name__ == '__main__':

    evaluator = Evaluator()

    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=100,
    #     callbacks=[evaluator]
    # )
    #
    # model.load_weights('best_model.weights')
    # print(u'final test acc: %05f\n' % (evaluate(test_generator)))

else:
    model = keras.models.Model(bert.model.input, output)
    AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

    model.compile(
        loss='sparse_categorical_crossentropy',
        # optimizer=Adam(1e-5),  # 用足够小的学习率
        optimizer=AdamLR(lr=1e-4, lr_schedule={
            1000: 1,
            2000: 0.1
        }),
        metrics=['accuracy'],
    )
    model.load_weights(os.path.join(dir_name,'best_model.weights'))