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
    model.add(LSTM(32, input_shape=(12, 1)))
    model.add(layers.Dense(1))
    model.add(Activation('relu')) 
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    model.summary()
    return model


time_unit = 1000 * 60

raw_data_null = []
raw_data_0 = []

# A data set that used to train a LSTM model.
Route = r'dblp-rels-retag-2000-2019_1951_hyper_degree_squence_observe_at_2000.txt'

files = open(Route, 'r', encoding='UTF-8')

# Filter the data set
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
    temp.append(raw_data_1[-1][ii])

raw_data_2 = np.array(raw_data_2, dtype=float)
raw_data_2 = raw_data_2.T

raw_data = []
for ii in range(0, len(raw_data_2)):
    temp = []
    for jj in raw_data_2[ii]:
        temp.append([jj])

    raw_data.append(temp)

random.shuffle(raw_data) 

N_features = len(raw_data[0]) - 1

num_tran_samples = len(raw_data)

train_data = []
train_targets = []
for ii in range(0, num_tran_samples):
    temp = raw_data[ii]
    temp = list(temp)
    temp_1 = temp[:-1]
    train_data.append(temp_1)
    train_targets.append(temp[-1])

train_data = np.array(train_data, dtype=float)
train_targets = np.array(train_targets, dtype=float)

num_epochs = 20
acc_histories = []
val_acc_histories = []

# the fourfold cross validation
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

# save the model that have been trained.
model.save('LSTM_model.h5')

# get mean prediction accuracy 
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
plt.legend(loc='best', prop=font1)
pp = PdfPages('model_accuracy.pdf')
pp.savefig()
plt.show()
pp.close()
