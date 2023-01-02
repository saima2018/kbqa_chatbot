#! -*- coding: utf-8 -*-
# SimBERT base 基本例子
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from tensorflow.python.keras.backend import set_session
import warnings
warnings.filterwarnings('ignore')
sess1: object = tf.Session()
graph1 = tf.get_default_graph()


maxlen = 32

dir_path = os.getcwd()

# bert配置
config_path = dir_path + \
    '/simbert/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = dir_path + \
    '/simbert/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = dir_path + '/simbert/chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

set_session(sess1)
# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])

ques = ['姚明的女儿', '姚明父亲']
X, S = [], []
for que in ques:
    x, s = tokenizer.encode(que)
    X.append(x)
    S.append(s)
X = sequence_padding(X)
S = sequence_padding(S)
with graph1.as_default():
    Z = encoder.predict([X, S])
    # print(Z)


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        output_ids = self.random_sample(
            [token_ids, segment_ids], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


# synonyms_generator = SynonymsGenerator(start_id=None,
#                                        end_id=tokenizer._token_end_id,
#                                        maxlen=maxlen)


class generateSimSentence(object):
    def __init__(self,):
        # 常见问题意图相似度匹配
        self.intention_dict ={'我想查询这个订单1925379234的状态':0,
                              '我想查这个地址0x6a0382d063637781b314a75284b38a302cb6aeca':1,
                              '我想查这个交易哈希地址':2,
                              '帮我查一下这个合约是不是真的':3,
                              '0x0412a3a3e19434eae709e20afb18dae6d90d9a7e698b77af6075caeb250313ac这个交易到账了吗':4,
                              }
        self.faq_dict = {
            '如何注册成为贵司客户': '轻松三步即可成为我司客户，第一步登陆【商户端】点击注册成为商户；第二步绑定设置【钱包信息】：提币地址&谷歌验证码；第三步设置API接口：API key &回调地址即可轻松安全享用我司服务',
            'Hambit支付交易服务优势和特色': '我司提供全球支付、转账和快捷兑换服务。客户可24小时不间断交易，并享受24小时在线客服。',
            '支持哪些币种': 'Hambit支持TRC-20（USDT、USDC、TUSD）、ERC-20（USDT、USDC、BUSD、TUSD）等等多主链、多币种收银',
            '结算周期': 'T+0',
            '单笔交易限额': '0至无上限',
            '交易到账时间': '10分钟内。如遇异常订单，请以客服反馈时间为准。',
            '手续费如何收取': '手续费是按入金费率进行扣除的。每笔收款到账后系统会扣除手续费，按照扣除手续费后的金额给商户入账，入一笔扣一笔。',
            '什么是正常收款': 'SDK发起生成的订单状态（用户点击进入的是币种选择页面，选择支付币种后进入支付页面）',
            '什么是临时收款': '通过客户端临时收款按钮/SDK临时收款按钮创建订单即为临时收款订单',
            '什么是充值业务': '充值客户订单（商户端自己选择了币种，用户点击链接直接进入支付页面）',
            '如何查询订单详情信息': '您好，我司为你提供多种便捷查询服务，您可在首页看到订单信息统计、订单图表亦可在订单管理中查询您详细单笔订单情况。',
            '为什么我的交易失败了': '您好，一般交易会在10分钟内到账。由于区块链网络的波动，到账时间可能会出现延迟。如您无法查询到订单交易，可联系人工客服帮助您进行查询。',
            '收银台是否有浮动金额，浮动范围': '汇率是跟随市场价进行浮动的，没有固定范围如果商户想要在市场价格基础上进行浮动，系统也可以支持设置浮动系数，浮动值最后可被商户吸收',
            '你推薦哪個比特幣錢包': '您好您可以使用Metamask等钱包，也可尝试使用Hambit自主研发的数字资产钱包HambitPay. HambitPay主要为C端用户提供加密资产存储及衍生服务，旨在构建一个真正自由开放的加密资产流通平台。HambitPay除了支持钱包用户向支付商家免手续费付款外，还搭载传统数字钱包的加密数字资产存储及流通功能、DAPP聚合入口、多链资产一站式交易功能外，同时还将集成NFT数字资产存储及流通、平台官方token及相关服务，为用户提供web3.0时代数字资产身份证。',
            'How to register as your customer?': 'First, log in to the merchant side and click on Register to become a merchant. The second step is to set the wallet information: withdrawal address & Google verification code. The third step is to set the API interface: API key & callback address. Then you can easily and safely enjoy our services.',
            'What\'s the amount limit for a transaction': '0 to No upper limit.', }
        with graph1.as_default():
            set_session(sess1)
            self.me = SynonymsGenerator(start_id=None,
                                        end_id=tokenizer._token_end_id,
                                        maxlen=maxlen)

    def gen_synonyms(self, text, n=10, k=5):
        """"含义： 产生sent的n个相似句，然后返回最相似的k个。
        做法：用seq2seq生成，并用encoder算相似度并排序。
        """
        r = self.me.generate(text, n)
        r = [i for i in set(r) if i != text]
        r = [text] + r
        X, S = [], []
        for t in r:
            x, s = tokenizer.encode(t)
            X.append(x)
            S.append(s)
        X = sequence_padding(X)
        S = sequence_padding(S)
        with graph1.as_default():
            set_session(sess1)
            Z = encoder.predict([X, S])
            Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
            argsort = np.dot(Z[1:], -Z[0]).argsort()
        return [r[i + 1] for i in argsort[:k]]

    def gen_sim_value(self, text_01, text_02):
        X, S = [], []
        x_1, s_1 = tokenizer.encode(text_01)
        x_2, s_2 = tokenizer.encode(text_02)
        X.append(x_1)
        X.append(x_2)
        S.append(s_1)
        S.append(s_2)
        X = sequence_padding(X)
        S = sequence_padding(S)
        with graph1.as_default():
            set_session(sess1)
            Z = encoder.predict([X, S])
            Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        return '%.2f' % (np.dot(Z[1], Z[0]))

    def gen_all_sim_value_faq(self, ques: list, pre=False, Z=None):
        if pre:
            X_pre, S_pre = [],[]
            for que in ques:
                x, s = tokenizer.encode(que)
                # print('x,s',x,s, que)
                X_pre.append(x)
                S_pre.append(s)
            X_pre = sequence_padding(X_pre)
            S_pre = sequence_padding(S_pre)
            with graph1.as_default():
                set_session(sess1)
                t4 = time.time()
                Z = encoder.predict([X_pre, S_pre])
                # print('ttttt启动加载faq至内存时间', time.time()-t4)
                # print('z before',Z,tf.shape(Z))
                Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
            return Z
        else:
            Z_pre = Z.copy()

            q_encoded, s_encoded = tokenizer.encode(ques[0])
            q_padded = sequence_padding([q_encoded])
            s_padded = sequence_padding([s_encoded])
            with graph1.as_default():
                set_session(sess1)
                t5 =time.time()
                Q = encoder.predict([q_padded, s_padded])
                # print('ttttt计算请求问题编码时间', time.time()-t5)

                Q /= (Q**2).sum(axis=1, keepdims=True)**0.5
                # print('ZQQQQQ',Z.shape, Q.shape)

                res = np.dot(Z_pre, Q[0])
                res = list(res)
            return res

    def gen_all_sim_value(self, ques: list):
        X, S = [],[]
        for que in ques:
            x, s = tokenizer.encode(que)
            # print('x,s',x,s, que)
            X.append(x)
            S.append(s)
        q_encoded, s_encoded = tokenizer.encode(ques[0])
        X.append(q_encoded)
        S.append(s_encoded)
        X = sequence_padding(X)
        S = sequence_padding(S)
        with graph1.as_default():
            set_session(sess1)
            t= time.time()
            Z = encoder.predict([X, S])

            Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
            # print('z after',Z)

            res = np.dot(Z[1:], Z[0])
            res = list(res)
            return res

    def faq_preload(self):
        # Encode and load FAQ into memory when starting the server



        faq_ques_list = []
        for q in self.faq_dict.keys():
            faq_ques_list.append(q)
        Z = self.gen_all_sim_value_faq(faq_ques_list, pre=True)
        return Z, faq_ques_list




