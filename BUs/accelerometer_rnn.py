import pandas as pd
import numpy as np
import os
from keras.backend.tensorflow_backend import dropout
from pandas.tests.io.parser import skiprows
from keras.initializers import Initializer, VarianceScaling
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import TimeseriesGenerator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from random import shuffle

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, RemoteMonitor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras import backend as K, optimizers, callbacks, losses

import time

import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

import os
from scipy import signal
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(3)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

tf.set_random_seed(1234)
classes = 2
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def get_data(fp, norm, std):
    data = pd.read_csv(filepath_or_buffer=fp, sep=',', names=["x", "y", "z"], usecols=[1, 2, 3], skiprows=0)
    yLabels = norm_data(pd.read_csv(filepath_or_buffer=fp, sep=',', usecols=[4, 5], dtype=np.int32))
    if norm and not std:
        normed_data = norm_data(data)
    return normed_data, yLabels


def subtract_mean(input_data):
    centered_data = input_data - input_data.mean()
    return centered_data


def norm_data(data):
    c_data = subtract_mean(data)
    mms = MinMaxScaler(copy=True, feature_range=[0, 1])
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data


def vectorize(normed):
    sequences = [normed[i:i + 100] for i in range(len(normed) - 100)]
#     shuffle(sequences)
    sequences = np.array(sequences)
    return sequences


def build_inputs(files_list, norm_bool, std_bool, center_bool):
    X_seq = []
    y_seq = []
    labels = {0, 1}
    for path in files_list:
        normed_data, y_labels = get_data(path, norm_bool, std_bool)
        input_list = vectorize(normed_data)
        # for yvals in range(len(y_labels.as_matrix())):
#         y_seq.append(y_labels)
        rangeSiz = len(input_list)
        if(len(input_list) >= len(y_labels)):
            rangeSiz = len(y_labels)
        else:
            rangeSiz = len(input_list)
        for inputs in range(rangeSiz):
            X_seq.append(input_list[inputs])
#             print("{} >>>> {}".format(inputs, input_list[inputs]))
#             if(inputs<y_labels.__len)
            y_seq.append(y_labels[inputs])
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    return X_, y_, labels


def build_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
#     model.add(LSTM(100, activation='tanh',
#                     use_bias=True, input_shape=(150, 3), return_sequences=False, dropout=0.0, recurrent_dropout=0.5))
#     model.add(Dropout(0.5))
#     model.summary()
#     model.add(LSTM(50, activation='tanh', return_sequences=False,
#                     go_backwards=True, input_shape=(150, 3), dropout=0.0, recurrent_dropout=0.5))
#     model.add(Dropout(0.5))
#     model.summary()
#     model.add(Activation("softmax"))
#     model.summary()
#     model.add(Dense(classes, activation='linear'))
#     model.summary()
#     data_gen = TimeseriesGenerator(X_train, Y_train,
#                                length=10, sampling_rate=10,
#                                batch_size=10)
    model.add(LSTM(100, activation='tanh', use_bias=True, unit_forget_bias=True,
                   kernel_regularizer=None, recurrent_regularizer=None, \
                    bias_regularizer=None, activity_regularizer=None, \
                    kernel_constraint=None, recurrent_constraint=None, \
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, \
                    return_sequences=True, kernel_initializer=VarianceScaling(scale=1.0, \
                    mode='fan_in', distribution='normal', seed=None), return_state=False, \
                    go_backwards=False, stateful=False, unroll=False, input_shape=(100, 3), implementation=1))
    model.add(Dropout(0.5))
    model.add(LSTM(50, activation='tanh',
                    use_bias=True, unit_forget_bias=True, recurrent_regularizer=None, \
                    bias_regularizer=None, activity_regularizer=None, \
                    kernel_constraint=None, recurrent_constraint=None, \
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, \
                    return_sequences=False, return_state=False, \
                    go_backwards=True, stateful=False, unroll=False, input_shape=(100, 3), implementation=2))
    
#      validation_split=0.05
    
    
    model.add(Dropout(0.5))
#     model.add(Dense(22, use_bias=True))
    model.add(Dense(11, use_bias=True))
    model.add(Dense(2))
    model.summary()
    model.add(Activation("softmax"))

#     model.add(LSTM(100, input_shape=(100, 3), return_sequences=True))
#     model.add(Dropout(.5, seed=5))
#     model.add(Dense(50, use_bias=True))
#     model.add(Dense(25, activation="relu", use_bias=True))
#     model.add(Dense(11, activation="relu", use_bias=True))
#     model.add(Dense(7, activation="relu"))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
#     tb_logs = TensorBoard(log_dir='./logs', histogram_freq=10,
#         batch_size=32, write_graph=True, write_grads=True, write_images=True,
#         embeddings_freq=25, embeddings_layer_names=None, embeddings_metadata=None)
    csv_logger = CSVLogger('C:/RecordingFiles/training.log', append=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10,
                                        verbose=0, mode='auto')
#     remote_monitor = RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data',
#                                    headers=None, send_as_json=True)
    history = model.fit(X_train, Y_train, batch_size=100, epochs=200,
              callbacks=[csv_logger, early_stop], validation_data=(X_test, Y_test), validation_split=0.05)
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#     for epoch in range(1000):
#         model.reset_states()
#         train_loss = 0
#         for i in range(10):
#             train_loss += model.train_on_batch(X_train[i:i + 1],
#                              Y_train[i:i + 1])
#         print ('# epoch', epoch, '  loss ', train_loss / float(Y_train.shape[0]))
    return model


def compute_accuracy(predictions, y_labels):
    predicted_labels = {0, 1}
    correct = 0
    for label in range(len(predicted_labels)):
        print("Predicted label: {}; Actual label: {}".format(predicted_labels[label], y_labels[label]))
        if predicted_labels[label] == y_labels[label]:
            correct += 1
    accuracy = 100 * (correct / len(predicted_labels))
    print("Predicted {} out of {} correctly for an Accuracy of {}%".format(correct, len(predicted_labels), accuracy))
    return


if __name__ == '__main__':

    training_files = {"C:/RecordingFiles/1/labeled.csv" , "C:/RecordingFiles/2/labeled.csv"}
    testing_files = {"C:/RecordingFiles/3/labeled.csv"}

    # build training inputs and labels
    X_train, y_train, train_labels = build_inputs(
        training_files,
        True, False, False)
    
    # build tesing inputs and labels
    X_test, y_test, test_labels = build_inputs(
        testing_files,
        True, False, False)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    # build and run model
    epochs = 1000  # 200
#     for s in 6,12,18:
#         model = build_model(s, y_train.size - 2)
    model = build_model(X_train, y_train, X_test, y_test)

    # launch TensorBoard via tensorboard --logdir=/full_path_to_your_logs
#     tb_logs = TensorBoard(log_dir='./logs', histogram_freq=10,
#         batch_size=32, write_graph=True, write_grads=True, write_images=True,
#         embeddings_freq=25, embeddings_layer_names=None, embeddings_metadata=None)
#     csv_logger = CSVLogger('C:/RecordingFiles/training.log', append=True)
#     early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10,
#                                         verbose=0, mode='auto')

#     history = model.fit(X_train, y_train, epochs=epochs, batch_size=1,
#         callbacks=[csv_logger, early_stop], validation_data=(X_test, y_test))  # , tb_logs])
#     print(history.history.keys())
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    pred = model.predict(X_test)
    print("Predicted one-hot values: {} \n Actual one-hot values: {}".format(pred, y_test))
    print("Prediction shape: {} \n Actual shape: {}".format(pred.shape, y_test.shape))

#     compute_accuracy(pred, test_labels)
