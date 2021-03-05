# -*- coding: utf-8 -*-

from numpy import random, average
from scipy.stats import powerlaw
import math
from matplotlib.backends.backend_pdf import PdfPages

from scipy import *
import numpy as np
import gc
import copy
from scipy.stats import poisson
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

Route = r'D:\dududu\model_related\dblp_data\extracted_data\dblp-rels-retag-1988-2019_1951_hyper_degree_squence_observe_at_2000.txt'

files = open(Route, 'r', encoding='UTF-8')
raw_data_null = []
raw_data_0 = []
# for i in range (10):
while 1:
    lin = files.readline().strip()
    if lin != '':
        data = lin.split()

        temp = []
        for t1 in data:
            temp.append(int(t1))

        # if int(temp[0]) + int(sum(temp[1:9])) > 13:  # ifll=0 这里是2009年之前的论文
        # 1951-2000年间论文数不超过13篇的
        if int(sum(temp[0:13])) > 60 or int(sum(temp[0:-2])) > 180:  # 只考虑历史论文数0篇的researchers
            continue

        data_1 = [int(data[0])]

        for ss in range(1, len(data) - 1):  # 已经去掉最后一列
            data_1.append(data_1[-1] + int(data[ss]))  # data_1 28列 1992-2019

        raw_data_null.append(data_1[1:])  # 27列，1993-2019
        raw_data_0.append(data_1[1:])
    else:
        break
files.close()

# for ll in range(0, 15):
# ss1 = ll + 2005  # 2001+8
test_data = []
test_targets = []

for kk in range(0, len(raw_data_null)):
    temp = []
    for ss in raw_data_null[kk][0: 12]:
        temp.append([ss])
    test_data.append(temp)

    test_targets.append(raw_data_null[kk][12])

print(raw_data_null[40])
print(test_data[40])
print(test_targets[40])
test_data = np.array(test_data, dtype=float)

# test_targets=np.array(test_targets, dtype = float)

# test_data=test_data.reshape( (1, len(test_data),len(test_data[0])   ))
# test_targets=test_targets.reshape( (1, len(test_targets),1   ))
# load_model = load_model('D:\\dududu\\deep_learning_test\\test_dududu.h5')                         ###### 加载模型
model = load_model('D:\dududu\model_related\deep_learning_test\model_test_history_15.h5')
y_predict = model.predict(test_data, batch_size=5)

# 预测篇数为1的调高
# 预测小于序列最后一年累计数时，调为相等
# alpha = 1.138
# beta = 1
count = 0
count_1 = 0
count_2 = 0


# ================================== 幂律函数 =====================================


def test_fun2(last_num):
    alpha = 0.05
    # 0.1
    # obeserved_at_2000 => (0.3, 1.15)

    # 0.33  1.22
    r = powerlaw.rvs(alpha, loc=0, scale=1 * last_num ** 0.88, size=1)  # 0.32 1.2/1.22
    return r


# ================================== 幂律函数 =====================================

# ================================== modify ======================================
for si in range(0, len(y_predict)):
    lam = test_fun2(sum(test_data[si][-1]))
    poi = random.poisson(lam=lam)
    y_predict[si] = [poi + y_predict[si][-1]]
# ================================== modify ======================================

print('=================')
print(y_predict[40])
print('=================')

# y_predict=np.round(y_predict)

ss = 'D://dududu//model_related//dblp_data//predict//LSTM1//dblp-rels-retag-test-predict-' + str(1 + 2000) + '_curr_history_15.txt'
fout = open(ss, 'w')
for ii in range(len(y_predict)):
    fout.write('%f\n' % y_predict[ii][0])

fout.close()

ss = 'D://dududu//model_related//dblp_data//predict//LSTM1//dblp-rels-retag-test-ture-' + str(1 + 2000) + '_curr_history_15.txt'
fout = open(ss, 'w')
for ii in range(len(test_targets)):
    fout.write('%f\n' % test_targets[ii])

fout.close()

for ll in range(2, len(raw_data_null[0]) - 11):  # (2, 14)
    test_data_1 = test_data
    test_data = []
    test_targets = []

    for kk in range(0, len(raw_data_null)):
        temp = []
        for ss in test_data_1[kk][1:]:
            temp.append(ss)
        temp.append(y_predict[kk])

        test_data.append(temp)

        # print(len(raw_data_null[kk]), ll + 11)
        test_targets.append(raw_data_null[kk][ll + 11])
    print(raw_data_null[40])
    print(test_data[40])
    test_data = np.array(test_data, dtype=float)
    # load_model = load_model('D:\\dududu\\deep_learning_test\\test_dududu.h5')                 ###### 加载模型

    y_predict = model.predict(test_data, batch_size=5)

    # ================================== modify ======================================

    for si in range(0, len(y_predict)):
        lam = test_fun2(sum(test_data[si][-1]))
        poi = random.poisson(lam=lam)
        y_predict[si] = [poi + sum(test_data[si][-1])]

    # ================================== modify ======================================

    print('=================')
    print(y_predict[40])
    print('=================')

    ss = 'D://dududu//model_related//dblp_data//predict//LSTM1//dblp-rels-retag-test-predict-' + str(ll + 2000) + '_curr_history_15.txt'
    fout = open(ss, 'w')
    for ii in range(len(y_predict)):
        fout.write('%f\n' % y_predict[ii][0])

    fout.close()

    ss = 'D://dududu//model_related//dblp_data//predict//LSTM1//dblp-rels-retag-test-ture-' + str(ll + 2000) + '_curr_history_15.txt'
    fout = open(ss, 'w')
    for ii in range(len(test_targets)):
        fout.write('%f\n' % test_targets[ii])

    fout.close()
