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
from sklearn.metrics.classification import confusion_matrix,\
    classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics.ranking import roc_curve, auc
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


def build_inputs(files_list, accel_labels, file_label_dict, norm_bool, std_bool, center_bool):
    X_seq = []
    y_seq = []
    labels = []
    for path in files_list:
        normed_data, target, target_label = get_data(path, accel_labels, file_label_dict, norm_bool, std_bool, center_bool)
        for inputs in range(len(normed_data)):
            X_seq.append(normed_data[inputs])
            y_seq.append(list(target))
            labels.append(target_label)
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    return X_, y_, labels

rootFolder = "C:/RecordingFiles/"
# Builds the LSTM model
def build_model():
    model = Sequential()
    model.add(Dense(3, input_dim=5))
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
    random_state = np.random.RandomState(0)
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()