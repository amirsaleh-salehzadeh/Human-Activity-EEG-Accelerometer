import pandas as pd
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import CSVLogger, EarlyStopping
from keras import backend as K
from biosppy.signals import eeg

import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(3)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

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


def get_data(fp, colsX, colsY):
    dataTmp = np.array(np.genfromtxt(fp, delimiter=',', usecols=colsX))
    if("EEG" in fp):
        eegdata = eeg.get_power_features(signal=dataTmp, sampling_rate=150, overlap=0.99)
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
    for i in range(len(normed) - 100):
        sequences.append(normed[i:i + 100])
        stdvVal.append(np.std(normed[i:i + 100]))
        varVal.append(np.var(normed[i:i + 100]))
        meanVal.append(np.mean(normed[i:i + 100]))
    sequences = np.array(sequences)
#     showSeq(sequences)
    return sequences


def build_inputs(files_list, colsX, colsY):
    X_seq = []
    y_seq = []
    for path in files_list:
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
    return X_, y_


def build_model(X_train, Y_train):
    model = Sequential()
    model.add(LSTM(111, activation='tanh', return_sequences=True, input_shape=(100, 10)))
    model.add(Dropout(0.5))
    model.add(LSTM(66, activation='tanh', return_sequences=True, input_shape=(100, 10)))
    model.add(Dropout(0.5))
    model.add(LSTM(33, activation='tanh', return_sequences=False, input_shape=(100, 10)))
    model.add(Dropout(0.5))
    model.add(Activation("sigmoid"))
    model.add(Dense(classes))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     plot_model(model, to_file='C:/RecordingFiles/model.png')
#     SVG(model_to_dot(model).create(prog='dot', format='svg'))
    csv_logger = CSVLogger('C:/RecordingFiles/training.csv', append=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10, verbose=0, mode='auto')
    history = model.fit(X_train, Y_train, batch_size=1, epochs=200,
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


if __name__ == '__main__':
    training_files = {"C:/RecordingFiles/labeledEEGTraining.csv"}
    X_train, y_train = build_inputs(training_files, [2, 3], [5, 6])
    model = build_model(X_train, y_train)
    
