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

while 1:
    lin = files.readline().strip()
    if lin != '':
        data = lin.split()

        temp = []
        for t1 in data:
            temp.append(int(t1))

        if int(sum(temp[0:13])) > 60 or int(sum(temp[0:-2])) > 180:
            continue

        data_1 = [int(data[0])]

        for ss in range(1, len(data) - 1):
            data_1.append(data_1[-1] + int(data[ss]))

        raw_data_null.append(data_1[1:])
        raw_data_0.append(data_1[1:])
    else:
        break
files.close()


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

model = load_model('D:\dududu\model_related\deep_learning_test\model_test_history_15.h5')
y_predict = model.predict(test_data, batch_size=5)

count = 0
count_1 = 0
count_2 = 0


# ================================== Powerlaw Function ==========================


def test_fun2(last_num):
    alpha = 0.1
    r = powerlaw.rvs(alpha, loc=0, scale=0.33 * last_num ** 1.22, size=1)
    return r


# ================================== Powerlaw Function ===========================


# ================================== modify ======================================

for si in range(0, len(y_predict)):
    lam = test_fun2(sum(test_data[si][-1]))
    poi = random.poisson(lam=lam)
    y_predict[si] = [poi + y_predict[si][-1]]

# ================================== modify ======================================


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

for ll in range(2, len(raw_data_null[0]) - 11):
    test_data_1 = test_data
    test_data = []
    test_targets = []

    for kk in range(0, len(raw_data_null)):
        temp = []
        for ss in test_data_1[kk][1:]:
            temp.append(ss)
        temp.append(y_predict[kk])
        test_data.append(temp)
        test_targets.append(raw_data_null[kk][ll + 11])
    
    print(raw_data_null[40])
    print(test_data[40])
    test_data = np.array(test_data, dtype=float)

    y_predict = model.predict(test_data, batch_size=5)

    # ================================== modify ======================================

    for si in range(0, len(y_predict)):
        lam = test_fun2(sum(test_data[si][-1]))
        poi = random.poisson(lam=lam)
        y_predict[si] = [poi + sum(test_data[si][-1])]

    # ================================== modify ======================================


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
