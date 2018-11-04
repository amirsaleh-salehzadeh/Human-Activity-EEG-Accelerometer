import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
import matplotlib.dates as md

import matplotlib.pyplot as plt


import random as rn

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)
np.random.seed(3)
rn.seed(12345)

from keras import backend as K



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
    """
    """
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


def get_data(fp, labels, folders):
    eeg = pd.read_csv(filepath_or_buffer=fp, sep=',', dtype='float',
                        names=["EEG1", "EEG2", "Acc_X", "Acc_Y", "Acc_Z"],
                        header=None, skiprows=1)
    eeg = eeg[:2000]
    rng = pd.date_range('00:00:00', periods=len(eeg), freq='5L')
    eeg = eeg.set_index(rng)
    eeg = eeg.resample('5ms')
    eeg = eeg.fillna(method='ffill')
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
    xfmt = md.DateFormatter('%M:%S')
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.xaxis_date()
    ax1.plot(eeg.index, eeg.EEG1); ax1.set_title('Raw EEG')
    ax2.specgram(eeg['EEG1'], Fs=200, NFFT=128, cmap=plt.get_cmap('Spectral_r'), noverlap=127)
    ax2.set_title('Spectrogram')
    ax2.set_ylabel('Freq (Hz)');
    ax2.set_xlabel('Time (s)');
    plt.show()



def build_inputs(files_list, accel_labels, file_label_dict, norm_bool, std_bool, center_bool):
    X_seq = []
    y_seq = []
    XT_seq = []
    yT_seq = []
    labels = []
    labelsT = []
    for path in files_list:
        get_data(path, accel_labels, file_label_dict)
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    XT_ = np.array(XT_seq)
    yT_ = np.array(yT_seq)
    return X_, y_, XT_, yT_, labels, labelsT




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
    
    
    
