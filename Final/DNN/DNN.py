import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt

import time

import tensorflow as tf
import random as rn
from keras import backend as K

from sklearn.metrics.classification import confusion_matrix, \
    precision_recall_fscore_support
import itertools
from sklearn.metrics.ranking import roc_auc_score, roc_curve
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
# https://pdfs.semanticscholar.org/df0b/05d8985846e694cda62d41a04e7c85090fa6.pdf

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


def drawMe(yVal=None, xVal=None, title="title", xlabel="xlabel", ylabel="ylabel", legend=["", ""], save=False,
           fileName="fileName", show=False):
    if(yVal != None):
        plt.plot(yVal)
    plt.plot(xVal)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if(legend != None):
        plt.legend(legend, loc='upper left')
    if(save):
        plt.savefig(fileName)
    if(show):
        plt.show()
    plt.close()
    return


def drawPLT(history):
    plt.figure(1)  
    plt.subplot(211)  
    plt.plot(history.history['acc'])  
    plt.plot(history.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    
    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left') 
    plt.subplots_adjust(hspace=0.5) 
    plt.savefig("acc_loss")
    plt.close()

    
def get_filepaths(mainfolder):
    training_filepaths = {}
    testing_filepaths = {}
    folders = os.listdir(mainfolder)
    for folder in folders:
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            filenames = os.listdir(fpath)
            for filename in filenames:
                fullpath = fpath + "/" + filename
                training_filepaths[fullpath] = folder
#             for filename1 in filenames[int(round(0.8 * len(filenames))):]:
#                 fullpath1 = fpath + "/" + filename1
#                 testing_filepaths[fullpath1] = folder
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
    data = pd.read_csv(filepath_or_buffer=fp, sep=',', dtype='float',
                        names=["EEG1", "EEG2", "Acc_X", "Acc_Y", "Acc_Z"],
                        header=None, skiprows=1)
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
    c_data = subtract_mean(data)    
    mms = MinMaxScaler()
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data


def vectorize(normed):
    sequences = [normed[i:i + 300] for i in range(len(normed) - 300)]
    shuffle(sequences)
    sequences = np.array(sequences, dtype='float')
    return sequences


def build_inputs(files_list, accel_labels, file_label_dict, norm_bool, std_bool, center_bool):
    X_seq = []
    y_seq = []
    XT_seq = []
    yT_seq = []
    labels = []
    labelsT = []
    for path in files_list:
        normed_data, target, target_label = get_data(path, accel_labels, file_label_dict, norm_bool, std_bool, center_bool)
        for inputs in range(len(normed_data)):
            if inputs < int(round(0.8 * len(normed_data))):
                X_seq.append(normed_data[inputs])
                y_seq.append(list(target))
                labels.append(target_label)
            else:
                XT_seq.append(normed_data[inputs])
                yT_seq.append(list(target))
                labelsT.append(target_label)
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    XT_ = np.array(XT_seq)
    yT_ = np.array(yT_seq)
    return X_, y_, XT_, yT_, labels, labelsT


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
    tttt = "Predicted {} out of {} correctly for an Accuracy of {}%".format(correct, len(predicted_labels), accuracy)
    f = open("predAcc.txt", "w+")
    f.write(tttt)
    f.close()
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('OrRd')):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("MTRX")

    
def evalRes(Y_test, pred, test_labels):
    y_pred = np.argmax(pred, axis=1)
    y_test = test_labels
    print('Classification Report')
    target_names = ['Reading', 'Speaking', 'Watching']
    cnf_matrix = confusion_matrix(y_pred, test_labels)
#   CLASSIFICATION REPORT
    df_class_report = pandas_classification_report(y_true=y_test, y_pred=y_pred)
    df_class_report.to_csv('classification_report.csv', sep=',')
#     MATRIX
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                          title='Normalized confusion matrix')
    


def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total

    class_report_df['avg / total'] = avg
    
    return class_report_df.T


def build_model():
    model = Sequential()
    model.add(Dense(20, input_dim=5))
    model.add(Activation('sigmoid'))
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(3))
    model.add(Activation('linear'))
    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation time: ", time.time(), '-', start)
    return model


if __name__ == '__main__':
    mainpath = "data/"
    activity_labels = get_labels(mainpath)
    training_dict, testing_dict = get_filepaths(mainpath)
    training_files = list(training_dict.keys())
    testing_files = list(testing_dict.keys())
    X_train, y_train, X_test, y_test, train_labels, test_labels = build_inputs(
        training_files,
        activity_labels,
        training_dict,
        True, False, False)
    epochs = 21  # 21
    model = build_model()
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=10,
                                    verbose=2, mode='auto')
    csv_logger = CSVLogger('training.csv', append=True, separator=',')
    history_callback = model.fit(X_train, y_train, epochs=epochs, batch_size= 1,
        validation_split=0.2, callbacks=[csv_logger, early_stop])
    
    drawPLT(history_callback)
    
    pred = model.predict(X_test)
    np.savetxt('pred.csv', pred, delimiter=',')
    compute_accuracy(pred, test_labels)
    evalRes(y_test, pred, test_labels)
