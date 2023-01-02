#! -*- coding: utf-8 -*-

import tensorflow as tf
import os
from random import shuffle
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
maxlen = 64
epochs = 15
batch_size = 1
bert_layers = 6
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

dir_name=os.path.dirname(os.path.abspath(__file__))
# bert配置
config_path = os.path.join(dir_name,'chinese_L-12_H-768_A-12/bert_config.json')
checkpoint_path = os.path.join(dir_name,'chinese_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(dir_name,'chinese_L-12_H-768_A-12/vocab.txt')

def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results

def load_data(filename):
    """加载数据
    单条格式：[(片段1, 标签1), (片段2, 标签2), (片段3, 标签3), ...]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D

#切割数据
# def transfer_data(s:str):
#     arr=s.replace(' ','').rstrip().split('\t')
#     sentence=arr[0]
#     chars=list(sentence)
#     labels=['O']*len(sentence)
#     entities=arr[1:-1]
#     type=arr[-1]
#     positions=[]
#     if type=='0':  # sp
#         if len(entities)==2:
#             [positions.append((sentence.find(ent),sentence.find(ent)+len(ent))) for ent in entities]
#             #s
#             labels[positions[0][0]]='B-SUB'
#             for i in range(positions[0][0]+1,positions[0][1]):
#                 labels[i] = 'I-SUB'
#             #p
#             labels[positions[1][0]] = 'B-PRO'
#             for i in range(positions[1][0] + 1,positions[1][1]):
#                 labels[i] = 'I-PRO'
#         else:
#             print('问题意图为：{}的句子:{}标注有误'.format(type,sentence))
#     elif type=='1': #spp
#         if len(entities) == 3:
#             [positions.append((sentence.find(ent), sentence.find(ent) + len(ent))) for ent in entities]
#             # s
#             labels[positions[0][0]] = 'B-SUB'
#             for i in range(positions[0][0] + 1, positions[0][1]):
#                 labels[i] = 'I-SUB'
#             # p1
#             labels[positions[1][0]] = 'B-PRO'
#             for i in range(positions[1][0] + 1, positions[1][1]):
#                 labels[i] = 'I-PRO'
#             # p2
#             labels[positions[2][0]] = 'B-PRO'
#             for i in range(positions[2][0] + 1, positions[2][1]):
#                 labels[i] = 'I-PRO'
#         else:
#             print('问题意图为：{}的句子:{}标注有误'.format(type, sentence))
#     elif type=='2':
#         if len(entities) == 2: #po
#             [positions.append((sentence.find(ent), sentence.find(ent) + len(ent))) for ent in entities]
#             # p
#             labels[positions[0][0]] = 'B-PRO'
#             for i in range(positions[0][0] + 1, positions[0][1]):
#                 labels[i] = 'I-PRO'
#             # o
#             labels[positions[1][0]] = 'B-OBJ'
#             for i in range(positions[1][0] + 1, positions[1][1]):
#                 labels[i] = 'I-OBJ'
#         else:
#             print('问题意图为：{}的句子:{}标注有误'.format(type, sentence))
#     elif type=='3':#op
#         if len(entities) == 2:
#             [positions.append((sentence.find(ent), sentence.find(ent) + len(ent))) for ent in entities]
#             # o
#             labels[positions[0][0]] = 'B-OBJ'
#             for i in range(positions[0][0] + 1, positions[0][1]):
#                 labels[i] = 'I-OBJ'
#             # p
#             labels[positions[1][0]] = 'B-PRO'
#             for i in range(positions[1][0] + 1, positions[1][1]):
#                 labels[i] = 'I-PRO'
#         else:
#             print('问题意图为：{}的句子:{}标注有误'.format(type, sentence))
#     elif type=='4':#so
#         if len(entities) == 2:
#             [positions.append((sentence.find(ent), sentence.find(ent) + len(ent))) for ent in entities]
#             # s
#             labels[positions[0][0]] = 'B-SUB'
#             for i in range(positions[0][0] + 1, positions[0][1]):
#                 labels[i] = 'I-SUB'
#             # O
#             labels[positions[1][0]] = 'B-OBJ'
#             for i in range(positions[1][0] + 1, positions[1][1]):
#                 labels[i] = 'I-OBJ'
#         else:
#             print('问题意图为：{}的句子:{}标注有误'.format(type, sentence))
#     result='\n'.join([c+' '+l for c,l in zip(chars,labels)])
#     #print(result)
#     return result
#
#
# if not os.path.exists('datasets/example.train'):
#     datas=[]
#     train_rate,dev_rate,test_rate=0.8,0.1,0.1
#     with open('datasets/data.txt','r',encoding='utf-8') as fread:
#         for line in fread:
#             datas.append(line.rstrip())
#     shuffle(datas)
#     cnt=len(datas)
#     with open('datasets/example.train','w',encoding='utf-8') as f_train:
#         for d in datas[:int(cnt*train_rate)]:
#             f_train.write('{}\n\n'.format(transfer_data(d)))
#     with open('datasets/example.dev','w',encoding='utf-8') as f_dev:
#         for d in datas[int(cnt * train_rate):int(cnt * (train_rate+dev_rate))]:
#             f_dev.write('{}\n\n'.format(transfer_data(d)))
#     with open('datasets/example.test','w',encoding='utf-8') as f_test:
#         for d in datas[int(cnt * (train_rate+dev_rate)):]:
#             f_test.write('{}\n\n'.format(transfer_data(d)))





# 标注数据
train_data = load_data(os.path.join(dir_name,'datasets/example.train'))
valid_data = load_data(os.path.join(dir_name,'datasets/example.dev'))
test_data = load_data(os.path.join(dir_name,'datasets/example.test'))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ['SUB', 'PRO', 'OBJ','O','SENT','PROP','B-SENT','I-SENT','B-PROP','I-PROP','B-OENT','I-OENT']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)
output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output


output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        # print('good till now')
        nodes = model.predict([token_ids, segment_ids])[0]
        # print('good till now??????')

        labels = self.decode(nodes)
        # print('labels',labels, text)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]

class NER:
    def __init__(self):
        self.NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        # print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights(os.path.join(dir_name,'best_model.weights'))
    NER.trans = K.eval(CRF.trans)