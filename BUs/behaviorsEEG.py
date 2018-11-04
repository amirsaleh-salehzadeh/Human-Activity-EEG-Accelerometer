import pandas as pd
import numpy as np
import os
import pickle
from keras.utils.vis_utils import plot_model
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import CSVLogger, EarlyStopping
from keras import backend as K
import tensorflow as tf
import random as rn
import time
from numpy.random.mtrand import shuffle
from biosppy.signals import eeg
import codecs
import csv
from biosppy import storage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
rootFolder = "C:/RecordingFiles/"
windowsize = 20


def get_filepaths(mainfolder):
    training_filepaths = {}
#     testing_filepaths = {}
    folders = os.listdir(mainfolder)
    for folder in folders:
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "ACC" not in folder:
            filenames = os.listdir(fpath)
            filenames = [x for x in filenames if "csvexperim" not in x]
            for filename in filenames[:len(filenames)]:
                fullpath = fpath + "/" + filename
                training_filepaths[fullpath] = folder
#             for filename1 in filenames[int(round(0.8 * len(filenames))):]:
#                 fullpath1 = fpath + "/" + filename1
#                 testing_filepaths[fullpath1] = folder
    return training_filepaths#, testing_filepaths


def get_labels(mainfolder):
    """ Creates a dictionary of labels for each unique type of motion """
    labels = {}
    label = 0
    for folder in os.listdir(mainfolder):
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "ACC" not in folder:
            labels[folder] = label
            label += 1
    return labels


def get_data(fp, labels, folders):
    file_dir = folders[fp]
    datasignals = pd.read_csv(filepath_or_buffer=fp, sep=',', usecols=[2, 3], dtype='float')
    plt.plot(datasignals)
    plt.title(file_dir)
    plt.ylabel("volts (mV)")
    plt.xlabel("time-steps")
    plt.savefig(rootFolder + file_dir + "-raw.png")
    plt.close()
    low = .2
    high = .8
    quant_df = datasignals.quantile([low, high])
    filt_df = datasignals.apply(lambda x: x[(x > quant_df.loc[low, x.name]) & 
                                    (x < quant_df.loc[high, x.name])], axis=0)
    plt.plot(filt_df)
    plt.title(file_dir + "- Min/Max Scaled")
    plt.ylabel("volts (mV)")
    plt.xlabel("time-steps")
    plt.savefig(rootFolder + file_dir + "-filtered.png")
    plt.close()
    eegdata = eeg.get_power_features(signal=filt_df, sampling_rate=200, size=.1,
                                 overlap=0.999)['all_bands_amir']
    eegdata = np.nan_to_num(eegdata)
    datasignals = norm_data(eegdata)
    one_hot = np.zeros(3)
    label = labels[file_dir]
    one_hot[label] = 1
    return datasignals, one_hot, label


def subtract_mean(input_data):
    centered_data = input_data - input_data.mean()
    return centered_data


def norm_data(data):
    c_data = subtract_mean(data)
    mms = MinMaxScaler()
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data


def standardize(data):
    c_data = subtract_mean(data)
    std_data = c_data / np.std(c_data)
    return std_data


def vectorize(normed):
    sequences = [normed[i:i + windowsize] for i in range(len(normed) - windowsize)]
    shuffle(sequences)
    sequences = np.array(sequences)
    return sequences


def build_inputs(files_list, accel_labels, file_label_dict):
    X_seq = []
    y_seq = []
    labels = []
    if(os.path.isfile(rootFolder + "experim.file")):
            with open(rootFolder + "experim.file", "rb") as f:
                dump = pickle.load(f)
                return dump[0], dump[1], dump[2]
    else:
        for path in files_list:
            if(os.path.isfile(path + "experim.file")):
                with open(path + "experim.file", "rb") as f:
                    dump = pickle.load(f)
                    input_list,  target, target_label =dump[0], dump[1], dump[2]
            else:
                normed_data, target, target_label = get_data(path, accel_labels, file_label_dict)
                input_list = vectorize(normed_data)
                with open(path + "experim.file", "wb") as f:
                    pickle.dump([input_list, target, target_label], f, pickle.HIGHEST_PROTOCOL)
            for inputs in range(len(input_list)):
                X_seq.append(input_list[inputs])
                y_seq.append(list(target))
                labels.append(target_label)
        X_ = np.array(X_seq)
        y_ = np.array(y_seq)
    with open(rootFolder + "experim.file", "wb") as f:
        pickle.dump([X_, y_, labels], f, pickle.HIGHEST_PROTOCOL)
    return X_, y_, labels


def build_model(X_train, Y_train, noLSTM):
    model = Sequential()
    with codecs.open(rootFolder + "training.csv", 'a') as logfile:
        fieldnames = ['lstms', 'outpts']
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)
        writer.writerow({'lstms': noLSTM[0], 'outpts': noLSTM[1]})
        print(noLSTM[0], " >> ", noLSTM[1])
    for p in range(noLSTM[0]):
        model.add(LSTM(noLSTM[p], activation='tanh', recurrent_activation='hard_sigmoid', \
                        use_bias=True, kernel_initializer='glorot_uniform', \
                        recurrent_initializer='orthogonal', \
                        unit_forget_bias=True, kernel_regularizer=None, \
                        recurrent_regularizer=None, \
                        bias_regularizer=None, activity_regularizer=None, \
                        kernel_constraint=None, recurrent_constraint=None, \
                        bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, \
                        implementation=1, return_sequences=True, return_state=False, \
                        go_backwards=False, stateful=False, unroll=False, \
                        input_shape=(windowsize, 26)))
        model.add(Dropout(0.5))
    model.add(Flatten())    
    model.add(Dense(3))
    model.add(Activation('softmax'))

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    print("Compilation time: ", time.time(), '-', start)
    fnametmp = "plot-{}-{}-{}.png".format("Model", noLSTM[0], noLSTM[1])
    plot_model(model, to_file=rootFolder + fnametmp, show_shapes=True, show_layer_names=True)
    csv_logger = CSVLogger(rootFolder + 'training.csv', append=True)
    early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose=2, mode='max')
    history = model.fit(X_train, Y_train, batch_size=1, epochs=100,
              callbacks=[csv_logger, early_stop], validation_split=0.2, shuffle=True, show_accuracy=False)
    with open(rootFolder + "history-" + noLSTM[0] + "-" + noLSTM[1] + ".file", "wb") as f:
        pickle.dump([history], f, pickle.HIGHEST_PROTOCOL)
#   ['acc', 'loss', 'val_acc', 'val_loss']
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fnametmp = "plot-{}-{}-{}.png".format("model-accuracy", noLSTM[0], noLSTM[1])
    plt.savefig(fnametmp)
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fnametmp = "plot-{}-{}-{}.png".format("model-loss", noLSTM[0], noLSTM[1])
    plt.savefig(fnametmp)
    plt.close()


def compute_accuracy(predictions, y_labels):
    predicted_labels = []
    for prediction in predictions:
        prediction_list = list(prediction)
        predicted_labels.append(prediction_list.index(max(prediction_list)))
    correct = 0
    for label in range(len(predicted_labels)):
        print("Predicted label: {}; Actual label: {}".format(predicted_labels[label], y_labels[label]))
        if predicted_labels[label] == y_labels[label]:
            correct += 1
    accuracy = 100 * (correct / len(predicted_labels))
    print("Predicted {} out of {} correctly for an Accuracy of {}%".format(correct, len(predicted_labels), accuracy))
    return


noLSTMOutputs = [4 , 12, 64, 128]
if __name__ == '__main__':
    activity_labels = get_labels(rootFolder)
    training_dict = get_filepaths(rootFolder)
    training_files = list(training_dict.keys())
    X_train, y_train, train_labels = build_inputs(
        training_files,
        activity_labels,
        training_dict)
    for q in range(1, 11):
        for tt in range(len([4, 6, 8])):
            build_model(X_train, y_train, np.array([q, noLSTMOutputs[tt]]))

