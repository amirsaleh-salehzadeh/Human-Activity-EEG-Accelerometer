import argparse
import os
import pickle as pk

import keras
import numpy as np
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import (ELU, GRU, LSTM, BatchNormalization, Bidirectional,
                          Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          Reshape, TimeDistributed)
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, load_model
from keras.utils import to_categorical
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils import shuffle

import models

SampFreq = 256
ChannelNum = 22

def get_parser():
    parser = argparse.ArgumentParser(description='train multiple win LOPO jobs')
    parser.add_argument('-f', '--folder', help='data floder')
    parser.add_argument('-s', '--save_dir', help='save dir')
    parser.add_argument('-se', '--start_epoch', help='start epoch')
    parser.add_argument('-e', '--epoch', help='train epoch')
    parser.add_argument('-c', '--ckpt_file', help='ckpt file')
    parser.add_argument('-m', '--model', help='model name')
    return parser.parse_args()

def load_data(data, label):
    t_data = np.load(open(data, 'rb')) 
    t_label = np.load(open(label, 'rb'))
    return t_data, t_label

def dataset_preprocess(data, label):
   # data shuffle
   s_data, s_label = shuffle(data, label, random_state=2018)
   return s_data, s_label


def main():
    args = get_parser()
    data_path = args.folder
    save_dir = args.save_dir
    start_epoch = int(args.start_epoch)
    epoch = int(args.epoch)
    ckpt_path = args.ckpt_file
    model_name = args.model


    model_options = {
            'raw_cnn':models.raw_cnn,
            'shallowconv': models.shallow_conv_net,
            'deepconv': models.deep_conv_net,
            'orig_EEG': models.origin_EEG_net,
            'hyper_EEG': models.hyper_tune_EEGnet,
            'swwae': models.SWWAE_model}

    train_data, train_label = load_data(
            os.path.join(
                data_path, '{}-data-train.npy'.format(data_path[-5:])), 
            os.path.join(
                data_path, '{}-label-train.npy'.format(data_path[-5:])))

    val_data, val_label = load_data(
            os.path.join(
                data_path, '{}-data-val.npy'.format(data_path[-5:])), 
            os.path.join(
                data_path, '{}-label-val.npy'.format(data_path[-5:])))

    print('train_size:', np.shape(train_data))
    print('val_size:', np.shape(val_data))
    data_shape = np.shape(train_data[0])

    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label)
    train_label = np.asarray(train_label)
    val_label = np.asarray(val_label)
    if not ckpt_path:
        model = model_options[model_name](2, 22, 768)
        model.compile(loss='categorical_crossentropy',
            optimizer = optimizers.Adagrad(), 
            metrics = ['accuracy'])
    else:
        model = load_model(ckpt_path)

    print(model.summary())
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, 'model.{epoch:04d}-{val_loss:.2f}.hdf5'))

    logger = CSVLogger(os.path.join(save_dir, "training-{}-{}.log.csv".format(start_epoch, epoch)))

    model.fit(
        train_data, train_label, batch_size=32, 
        epochs=epoch, verbose=1, 
        validation_data=(val_data, val_label), shuffle=True, 
        initial_epoch=start_epoch, callbacks=[checkpoint, logger])

if __name__ == '__main__':
    main()
