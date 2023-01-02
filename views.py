#! -*- coding: utf-8 -*-

import os
from flask import render_template, Flask, request, redirect, url_for
try:
    import simplejson as json
except:
    import json
from QA_init import GiveFlaskWebData
from simbert.simbert_base import generateSimSentence
import time

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dir_path = os.getcwd()

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.allow_growth = True
# session=tf.Session(config=config)
# KTF.set_session(session)

app = Flask(__name__)
# graph = tf.get_default_graph()


@app.route('/')
@app.route('/success/')
@app.route('/ownthinkQA/')
def start():
    return render_template('index.html')


# 系统使用方式提醒
@app.route('/ownthinkQA/<question>')
def ownthinkQA(question):
    ans, data, link = search(question)
    data_01 = json.dumps(data)
    link_01 = json.dumps(link)
    return render_template('true.html', ans=ans, data_list=data_01, link_list=link_01)


@app.route('/search', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        question = request.form['question']
        # print((question))
        return redirect(url_for('ownthinkQA', question=question))
    else:
        question = request.args.get('question')
        return redirect(url_for('ownthinkQA', question=question))

sim = generateSimSentence()
t0 =time.time()
Z_faq_pre, faq_ques_list = sim.faq_preload()
print('ttttt000000', time.time() - t0)


def search(question, Z_faq_pre=Z_faq_pre):
    getData = GiveFlaskWebData()
    start_time = time.time()

    # Try QA pair matching first
    t2 = time.time()
    sim_val_list = sim.gen_all_sim_value_faq([question],pre=False,Z=Z_faq_pre)  # 批量计算
    print('ttttttt2222222', time.time()-t2)

    # print('sssssssssssim', sim_val_list)
    temp_sim_val = max(sim_val_list)
    # print('tttp',temp_sim_val)
    max_sim_val = '%.4f' % (float(temp_sim_val))
    sim_threshold = 0.55
    if float(max_sim_val) > sim_threshold:
        q_index = sim_val_list.index(temp_sim_val)
        # print('qqqqqqxxxxxx', q_index, faq_ques_list, len(faq_ques_list))
        faq_q = faq_ques_list[q_index]

        answer = sim.faq_dict[faq_q]
        # searchPath = 'FAQ'
        sim_val = max_sim_val
        print('相似问题顺序：%s,相似度%s' % (faq_ques_list[q_index], sim_val))
        timeUsed = str(int(float('%.3f' % (time.time() - start_time)) * 1000)) + 'ms'

        answer = {
            '查询路径': 'FAQ',
            '返回答案': answer,
            '答案得分': sim_val,
            '用时': timeUsed
        }
        return answer, [], []
    # No similar questions in FAQ, use KGQA instead
    else:
        answer, data, link = getData.getWebTypeData(question)

    # print(data)
    return answer, data, link


if __name__ == "__main__":
    app.debug = False

    app.run(host='0.0.0.0',port='5000')
