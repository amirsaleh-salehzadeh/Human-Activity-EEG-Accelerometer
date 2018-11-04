import pandas as pd
import numpy as np
import os
import pickle
from keras.utils.vis_utils import plot_model
from keras.layers.recurrent import LSTM
import codecs
import csv
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.callbacks import CSVLogger, EarlyStopping
from keras import backend as K
from biosppy.signals import eeg

import tensorflow as tf
import random as rn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(3)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
windowsize = 100
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

tf.set_random_seed(1234)
classes = 2
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

stdvVal = []
varVal = []
meanVal = []
inptVal = []
pwrVal = []
rootFolder = "C:/RecordingFiles/"


def get_data(fp, colsX, colsY):
    dataTmp = np.array(np.genfromtxt(fp, delimiter=',', usecols=colsX))
    if("EEG" in fp):
        eegdata = eeg.get_power_features(signal=dataTmp, sampling_rate=150, overlap=0.777)
    yLabels = norm_data(pd.read_csv(filepath_or_buffer=fp, sep=',', usecols=colsY, dtype=np.int32))
    normed_data = norm_data(eegdata["all_bands_amir"])
    return normed_data, yLabels


def subtract_mean(input_data):
    centered_data = input_data - input_data.mean()
    return centered_data


def norm_data(data):
    c_data = subtract_mean(data)
    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data


def vectorize(normed):
    sequences = []
    for i in range(len(normed) - windowsize):
        sequences.append(normed[i:i + windowsize])
        stdvVal.append(np.std(normed[i:i + windowsize]))
        varVal.append(np.var(normed[i:i + windowsize]))
        meanVal.append(np.mean(normed[i:i + windowsize]))
    sequences = np.array(sequences)
#     showSeq(sequences)
    return sequences


def build_inputs(files_list, colsX, colsY):
    X_seq = []
    y_seq = []
    for path in files_list:
        if(os.path.isfile(path + ".file")):
            with open(path + ".file", "rb") as f:
                dump = pickle.load(f)
                return dump[0], dump[1]
        else:
            stdvVal = []
            varVal = []
            meanVal = []
            normed_data, y_labels = get_data(path, colsX, colsY)
            input_list = vectorize(normed_data)
            rangeSiz = len(input_list)
            if(len(input_list) >= len(y_labels)):
                rangeSiz = len(y_labels)
            else:
                rangeSiz = len(input_list)
            for inputs in range(rangeSiz):
                X_seq.append(input_list[inputs])
                y_seq.append(y_labels[inputs])
            X_ = np.array(X_seq)
            y_ = np.array(y_seq)
            with open(path + ".file", "wb") as f:
                pickle.dump([X_, y_], f, pickle.HIGHEST_PROTOCOL)
    return X_, y_


def build_model(X_train, Y_train, noLSTM):
    with codecs.open(rootFolder + "training.csv", 'a') as logfile:
        fieldnames = ['lstms', 'outpts']
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)
        writer.writerow({'lstms': noLSTM[0], 'outpts': noLSTM[1]})
        print(noLSTM[0], " >> ", noLSTM[1])
#         spamwriter.writerow([noLSTM[0], noLSTM[1]])
#     return
    model = Sequential()
#     model.add(DeepConvNet(classes, 10, windowsize))
    for p in range(noLSTM[0]):
        model.add(LSTM(noLSTM[p], kernel_initializer    ='he_normal', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
                           recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                           kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=True,
                           return_state=False, stateful=False, input_shape=(windowsize, 10)))
#      model.add(ConvLSTM2D(11, kernel_size=(windowsize/2, 10), activation='tanh', kernel_initializer='he_normal', recurrent_initializer='orthogonal', 
#                        bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
#                        recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
#                        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, 
#                        return_state=False, stateful=False, input_shape=(windowsize, 10)))
    model.add(Flatten())
    model.add(Dense(math.ceil(noLSTM[1]/2)))
    model.add(Dropout(0.5))
#     model.add(LSTM(7, activation='relu', go_backwards=True, return_sequences=False))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='sigmoid'))
#     model.add(Dense(11, activation='sigmoid'))
    model.add(Dense(classes))
    model.compile(loss='mse', optimizer='adamax', metrics=['accuracy'])
    fnametmp = "model_plot{}-{}.png".format(noLSTM[0], noLSTM[1])
    plot_model(model, to_file=rootFolder + fnametmp, show_shapes=True, show_layer_names=True)
    csv_logger = CSVLogger(rootFolder + 'training.csv', append=True)
    early_stop = EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode='auto')
    history = model.fit(X_train, Y_train, batch_size=1, epochs=20,
              callbacks=[csv_logger, early_stop], validation_split=0.2)
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model


noLSTMOutputs = [4 , 12, 64, 128]
if __name__ == '__main__':
    training_files = {rootFolder + "labeledEEG.csv"}
    X_train, y_train = build_inputs(training_files, [2, 3], [5, 6])
    for q in range(1, 11):
        for tt in range(len([4, 6, 8])):
            model = build_model(X_train, y_train, np.array([q, noLSTMOutputs[tt]]))
    
