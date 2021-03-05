# -*- coding: utf-8 -*-
from math import log

import matplotlib.pyplot as plt
import scipy.misc
import scipy
import time
import string
import os
import matplotlib
import pylab as pl
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
from keras import models
from keras import layers
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation, Dropout
from keras.layers import Flatten
from keras.models import load_model


def build_model():
    model = models.Sequential()
    # model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(149, 1)))
    # model.add(layers.MaxPooling1D(5))
    #   model.add(SimpleRNN(32, input_shape=(8, 1)))  # , return_sequences=True)
    model.add(LSTM(32, input_shape=(12, 1)))
    # model.add(Dropout(0.2))  # 被遗忘的比例
    #    model.add(SimpleRNN(32))
    # model.add(layers.Dense(32  ))
    model.add(layers.Dense(1))
    # model.add(Activation('linear'))
    model.add(Activation('relu'))  # 激活因子
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    model.summary()
    return model


time_unit = 1000 * 60

raw_data_null = []
raw_data_0 = []
Route = r'D:\dududu\model_related\dblp_data\extracted_data\dblp-rels-retag-2000-2019_1951_hyper_degree_squence_observe_at_2000.txt'

files = open(Route, 'r', encoding='UTF-8')

# for i in range (10):
while 1:
    lin = files.readline().strip()
    if lin != '':
        data = lin.split()

        if int(data[0]) > 15:
            continue
        data_1 = [int(data[0])]

        for ss in range(1, len(data) - 1):
            data_1.append(data_1[-1] + int(data[ss]))

        raw_data_null.append(data_1[1:])
        raw_data_0.append(data_1[1:14])
    else:
        break
files.close()

raw_data_0 = np.array(raw_data_0, dtype=float)

raw_data_1 = raw_data_0.T
print(len(raw_data_1[0]), len(raw_data_1))

raw_data_2 = raw_data_1

temp = []
for ii in range(0, len(raw_data_1[-1])):
    # if raw_data_1[-1][ii]!=0:
    temp.append(raw_data_1[-1][ii])

print('mean score', average(temp))

raw_data_2 = np.array(raw_data_2, dtype=float)

raw_data_2 = raw_data_2.T

raw_data = []
for ii in range(0, len(raw_data_2)):
    temp = []
    for jj in raw_data_2[ii]:
        temp.append([jj])

    raw_data.append(temp)

random.shuffle(raw_data)  # raw_data是打乱后的raw_data_2

print(len(raw_data))

N_features = len(raw_data[0]) - 1

num_tran_samples = len(raw_data)

train_data = []
train_targets = []
for ii in range(0, num_tran_samples):
    temp = raw_data[ii]
    temp = list(temp)

    temp_1 = temp[:-1]
    train_data.append(temp_1)  # 前12列数据作为输入，输出为第13列的预测值

    train_targets.append(temp[-1])  # 第13列的真实值

train_data = np.array(train_data, dtype=float)

train_targets = np.array(train_targets, dtype=float)

num_epochs = 20
acc_histories = []
val_acc_histories = []

k = 4
num_val_samples = len(train_data) // k
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    print(train_data.shape)
    print(partial_train_data.shape)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=4, verbose=0)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    acc_histories.append(acc)
    val_acc_histories.append(val_acc)

model.save('D:\dududu\model_related\deep_learning_test\model_test_history_15.h5')

average_acc_history = [
    np.mean([x[i] for x in acc_histories]) for i in range(num_epochs)]

average_val_acc_history = [
    np.mean([x[i] for x in val_acc_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_acc_history) + 1), average_acc_history, 'bo', label='Training', linewidth=1.5)
plt.plot(range(1, len(average_val_acc_history) + 1), average_val_acc_history, 'b', label='Validation', linewidth=1.5)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
plt.xlabel('Epochs', font1)
plt.ylabel('Accuracy', font1)
# plt.title('Training and validation accuracy')
plt.legend(loc='best', prop=font1)
pp = PdfPages('D:\dududu\model_related\deep_learning_test\model_acc_history_15.pdf')
pp.savefig()
plt.show()
pp.close()
