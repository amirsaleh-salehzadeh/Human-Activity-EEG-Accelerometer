from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2019)

import keras
import LOPO_train
import argparse
from keras.models import load_model
import os
from keras import optimizers
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger

def get_parser():
    parser = argparse.ArgumentParser(description='fine tune model')
    parser.add_argument('-f', '--folder', help='data floder for finetune data')
    parser.add_argument('-s', '--save_dir', help='save dir')
    parser.add_argument('-e', '--epoch', help='train epoch')
    parser.add_argument('-c', '--ckpt_file', help='ckpt file')
    return parser.parse_args()


def load_fine_tune_data(data_dir):
    seizure_file_list = []
    normal_data_path = ''
    for file in os.listdir(data_dir):
            if 'seizure' in file:
                seizure_file_list.append(os.path.join(data_dir, file))
            if 'normal' in file:
                normal_data_path = os.path.join(data_dir, file)
    seizure_data = []
    no_seizure_data = []
    raw_no_seizure_data = np.load(normal_data_path)
    start = 0
    for file_path in seizure_file_list:
        data = np.load(file_path)
        seizure_data.append(data)
        no_seizure_data.append(raw_no_seizure_data[start:start + len(data)])
        start += len(data)

    return seizure_data, no_seizure_data

def gen_val(except_index, seizure_data, no_seizure_data):
    val_data = []
    val_label = []
    for i in range(len(seizure_data)):
        if i == except_index:
            continue
        val_data.extend(seizure_data[i])
        val_data.extend(no_seizure_data[i])
        label = [1]*len(seizure_data[i]) + [0]*len(no_seizure_data[i])
        val_label.extend(label)
    return val_data, val_label

def main():
    args = get_parser()
    data_path = args.folder
    save_dir = args.save_dir
    epoch = int(args.epoch)
    base_model = args.ckpt_file
    seizure_data, no_seizure_data = load_fine_tune_data(data_path)
    LORO_num = len(seizure_data)
    for i in range(LORO_num):
        # prepare train/val data, using leave one record out
        train_data = []
        train_data.extend(seizure_data[i])
        train_data.extend(no_seizure_data[i])
        train_label = [1]*len(seizure_data[i]) + [0]*len(no_seizure_data[i])
        val_data, val_label = gen_val(i, seizure_data, no_seizure_data)

        val_label = to_categorical(val_label)
        train_label = to_categorical(train_label)
        train_data = np.asarray(train_data)
        val_data = np.asarray(val_data)
        model = load_model(base_model)
        # freeze all layer except Dense
        # train the Dense layer
        for layer in model.layers:
            layer_config = layer.get_config()
            print(layer_config['name'])
            if not 'dense' in layer_config['name']:
                layer.trainable = False
        print(model.summary())
        print('train_size: {}'.format(np.shape(train_data)))
        print('val_size: {}'.format(np.shape(val_data)))
        model.compile(
                loss='categorical_crossentropy', 
                optimizer = optimizers.Adam(lr=0.0001),
                metrics=['accuracy'])

        cur_dir = os.path.join(save_dir, '{}_LORO'.format(i))
        if not os.path.isdir(cur_dir):
            os.mkdir(cur_dir)

        checkpoint = ModelCheckpoint(
            os.path.join(cur_dir, 'model.{epoch:04d}-{val_loss:.2f}.hdf5'))

        logger = CSVLogger(os.path.join(cur_dir, "training-{}-{}.log.csv".format(0, epoch)))
        history = model.fit(
                train_data, train_label, batch_size=32, 
                epochs=epoch, verbose=1, 
                validation_data=(val_data, val_label), 
                shuffle=True, 
                initial_epoch=0, callbacks=[checkpoint, logger])
        print(history)


if __name__ == '__main__':
    main()
