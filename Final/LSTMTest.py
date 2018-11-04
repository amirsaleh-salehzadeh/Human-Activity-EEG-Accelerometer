import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import time

import tensorflow as tf
import random as rn
from keras import backend as K

import os
import codecs
import csv
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)
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

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def get_filepaths(mainfolder):
    """
    """
    training_filepaths = {}
    testing_filepaths = {}
    folders = os.listdir(mainfolder)
    for folder in folders:
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            filenames = os.listdir(fpath)
            for filename in filenames[:int(round(0.8 * len(filenames)))]:
                fullpath = fpath + "/" + filename
                training_filepaths[fullpath] = folder
            for filename1 in filenames[int(round(0.8 * len(filenames))):]:
                fullpath1 = fpath + "/" + filename1
                testing_filepaths[fullpath1] = folder
    return training_filepaths, testing_filepaths


def get_labels(mainfolder):
    """ Creates a dictionary of labels for each unique type of motion """
    labels = {}
    label = 0
    for folder in os.listdir(mainfolder):
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            labels[folder] = label
            label += 1
    return labels

def build_inputs(files_list, accel_labels, file_label_dict, norm_bool, std_bool, center_bool):
    X_seq = []
    y_seq = []
    labels = []
    for path in files_list:
        normed_data, target, target_label = get_data(path, accel_labels, file_label_dict, norm_bool, std_bool, center_bool)
        input_list = vectorize(normed_data)
        for inputs in range(len(input_list)):
            X_seq.append(input_list[inputs])
            y_seq.append(list(target))
            labels.append(target_label)
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    return X_, y_, labels

rootFolder = "C:/RecordingFiles/"

def get_data(fp, labels, folders, norm, std, center):
    """
    Creates a dataframe for the data in the filepath and creates a one-hot
    encoding of the file's label
    """
    data = pd.read_csv(filepath_or_buffer=fp, sep=',', dtype='float')
    normed_data = norm_data(data)
    one_hot = np.zeros(3)
    file_dir = folders[fp]
    label = labels[file_dir]
    one_hot[label] = 1
    return normed_data, one_hot, label


def subtract_mean(input_data):
    # Subtract the mean along each column
    centered_data = input_data - input_data.mean()
    return centered_data


def norm_data(data):
    """
    Normalizes the data.
    For normalizing each entry, y = (x - min)/(max - min)
    """
    c_data = subtract_mean(data)
    mms = MinMaxScaler()
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data


def vectorize(normed):
    """
    Uses a sliding window to create a list of (randomly-ordered) 300-timestep
    sublists for each feature.
    """
    sequences = [normed[i:i + 300] for i in range(len(normed) - 300)]
    shuffle(sequences)
    sequences = np.array(sequences, dtype='float')
    return sequences



# Builds the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', \
                    use_bias=True, kernel_initializer='glorot_uniform', \
                    recurrent_initializer='orthogonal', \
                    unit_forget_bias=True, kernel_regularizer=None, \
                    recurrent_regularizer=None, \
                    bias_regularizer=None, activity_regularizer=None, \
                    kernel_constraint=None, recurrent_constraint=None, \
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, \
                    implementation=1, return_sequences=True, return_state=False, \
                    go_backwards=False, stateful=False, unroll=False, \
                    input_shape=(300, 5)))
    model.add(Dropout(0.5))
    model.add(LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', \
                    use_bias=True, kernel_initializer='glorot_uniform', \
                    recurrent_initializer='orthogonal', \
                    unit_forget_bias=True, kernel_regularizer=None, \
                    recurrent_regularizer=None, \
                    bias_regularizer=None, activity_regularizer=None, \
                    kernel_constraint=None, recurrent_constraint=None, \
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, \
                    implementation=1, return_sequences=False, return_state=False, \
                    go_backwards=False, stateful=False, unroll=False,
                    input_shape=(300, 5)))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    print("Compilation time: ", time.time(), '-', start)

    return model


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


if __name__ == '__main__':

    mainpath = "C:\RecordingFiles"

    activity_labels = get_labels(mainpath)
    training_dict, testing_dict = get_filepaths(mainpath)
    training_files = list(training_dict.keys())
    testing_files = list(testing_dict.keys())

    # build training inputs and labels
    X_train, y_train, train_labels = build_inputs(
        training_files,
        activity_labels,
        training_dict,
        True, False, False)
    # build tesing inputs and labels
    X_test, y_test, test_labels = build_inputs(
        training_files,
        activity_labels,
        training_dict,
        True, False, False)

    # build and run model
    epochs = 5  # 200
    for test in range(5):
        model = build_model()
        csv_logger = CSVLogger('training.log', append=True)

        # launch TensorBoard via tensorboard --logdir=/full_path_to_your_logs
        tb_logs = TensorBoard(log_dir='./logs', histogram_freq=10,
        batch_size=32, write_graph=True, write_grads=True, write_images=True,
        embeddings_freq=25, embeddings_layer_names=None, embeddings_metadata=None)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10,
                                        verbose=0, mode='auto')
        csv_logger = CSVLogger(rootFolder + 'training.csv', append=True)
        model.fit(X_train, y_train, epochs=epochs,
            validation_split=0.2, callbacks=[csv_logger, early_stop])  # , tb_logs])

        pred = model.predict(X_test)
        print("Predicted one-hot values: {} \n Actual one-hot values: {}".format(pred, y_test))
        print("Prediction shape: {} \n Actual shape: {}".format(pred.shape, y_test.shape))

        compute_accuracy(pred, test_labels)
