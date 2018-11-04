import os
import pickle as pk

import keras
import keras.backend as K
import numpy as np
from keras import optimizers
from keras.applications.mobilenet import DepthwiseConv2D
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.constraints import max_norm
from keras.initializers import glorot_normal, he_normal, he_uniform
from keras.layers import (ELU, GRU, BatchNormalization, Bidirectional,
                          Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          Reshape, SpatialDropout2D, TimeDistributed)
from keras.layers.convolutional import (AveragePooling2D, Conv2D, MaxPooling2D,
                                        SeparableConv2D)
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1_l2
from keras.utils import to_categorical
from sklearn.feature_selection import RFECV


def raw_cnn(num_classes, chans=22, samples=768):
    '''
    a simple cnn with ELU as activation & MaxPooling
    '''
    data_shape = (3, chans, samples)
    model = Sequential()
    model.add(Conv2D(64, (1, 30), 
        data_format='channels_first', input_shape=data_shape))
    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())
    model.add(Dropout(0.2, seed=2018))
    model.add(Conv2D(64, (5, 30), data_format='channels_first'))
    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())
    model.add(MaxPooling2D((1, 6), data_format='channels_first'))
    model.add(Dropout(0.2, seed=2018))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

def shallow_conv_net(num_classes, chans = 22, samples = 768):
    """ 
    Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    """

    def square(x):
        return K.square(x)

    def log(x):
        return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

    # start the model
    input_main = Input((3, chans, samples))
    block1 = Conv2D(40, (1, 15), data_format='channels_first',
                           input_shape=(3, chans, samples))(input_main)
    block1 = Conv2D(40, (chans, 1), use_bias=False, data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15), data_format='channels_first')(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(0.5)(block1)
    flatten = Flatten()(block1)
    dense = Dense(num_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)



def deep_conv_net(num_classes, chans = 22, samples = 768):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    """
    
    # start the model
    input_main = Input((3, chans, samples))
    block1 = Conv2D(25, (1, 5), data_format='channels_first',
             input_shape=(3, chans, samples))(input_main)

    block1 = Conv2D(25, (chans, 1), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block1)
    block1 = Dropout(0.5)(block1)
  
    block2 = Conv2D(50, (1, 5), data_format='channels_first')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block2)
    block2 = Dropout(0.5)(block2)
    
    block3 = Conv2D(100, (1, 5), data_format='channels_first')(block2)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block3)
    block3 = Dropout(0.5)(block3)
    
    block4 = Conv2D(200, (1, 5),data_format='channels_first' )(block3)
    block4 = BatchNormalization(axis=1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block4)
    block4 = Dropout(0.5)(block4)
    
    flatten = Flatten()(block4)
    
    dense   = Dense(num_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


def origin_EEG_net(num_classes, chans=22, samples=768):
    '''
    original EEGnet
    '''
    data_shape = (3, chans, samples)
    model = Sequential()
    model.add(Conv2D(4, (1, 64), 
        data_format='channels_first', padding='same', use_bias=False,
        input_shape=data_shape))

    model.add(BatchNormalization(axis=1))

    model.add(DepthwiseConv2D((chans, 1), 
        data_format='channels_first', use_bias=False,
        depth_multiplier=2, depthwise_constraint=max_norm(1.)))

    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(AveragePooling2D((1, 4), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(SeparableConv2D(8, (1, 16), 
        use_bias=False,
        padding='same',
        data_format='channels_first'))

    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(AveragePooling2D((1, 8), data_format='channels_first'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', 
        kernel_constraint=max_norm(0.25)))
    return model

def hyper_tune_EEGnet(num_classes, chans=22, samples=768, pool_method='AVG'):
    data_shape = (3, chans, samples)
    pooling_options = {
            'AVG':AveragePooling2D, 
            'MAX':MaxPooling2D
            }

    model = Sequential()
    model.add(Conv2D(16, (1, 32), 
        data_format='channels_first', padding='same', use_bias=False,
        kernel_initializer= glorot_normal(),
        input_shape=data_shape))

    model.add(BatchNormalization(axis=1, momentum=0.01))

    model.add(DepthwiseConv2D((chans, 1), 
        data_format='channels_first', use_bias=False,
        depth_multiplier=2, depthwise_constraint=max_norm(1.),
        kernel_initializer=glorot_normal()))

    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())
    model.add(pooling_options[pool_method]((1, 4), data_format='channels_first'))
    model.add(Dropout(0.5))

    model.add(SeparableConv2D(32, (1, 16), 
        kernel_initializer= glorot_normal(), use_bias=False,
        padding='same',
        data_format='channels_first'))

    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())
    model.add(pooling_options[pool_method]((1, 8), data_format='channels_first'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', 
        kernel_constraint=max_norm(0.25)))
    return model

def SWWAE_model(num_classes, chans=22, samples=768):
    data_shape = (3, chans, samples)
    input_tensor = Input(shape=data_shape, name='Input')
    x = Conv2D(16, (1, 32), 
            data_format='channels_first', padding='same', use_bias=False,
            kernel_initializer= glorot_normal())(input_tensor)
    

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    
    x = DepthwiseConv2D((22, 1), 
        data_format='channels_first', use_bias=False,
        depth_multiplier=2, depthwise_constraint=max_norm(1.),
        kernel_initializer=glorot_normal())(x)

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    x = ELU()(x)

    orig_1 = x
    x = AveragePooling2D((1, 4), data_format='channels_first')(x)
    encode_1 = Dropout(0.5)(x)

    # decode part
    
    x = UpSampling2D(size=(1, 4))(encode_1)
    the_shape = K.int_shape(orig_1)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_1)
    # print('origReshaped.shape: ' + str(K.int_shape(origReshaped)))
    xReshaped = Reshape(shape)(x)
    # print('xReshaped.shape: ' + str(K.int_shape(xReshaped)))
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    # print('together.shape: ' + str(K.int_shape(together)))
    x = Unpooling()(together)
    x = ELU()(x)
    x = BatchNormalization(axis=1, momentum=0.01)(x)


    x = Conv2DTranspose(32, (22, 1), 
        data_format='channels_first', use_bias=False,
        kernel_initializer=glorot_normal())(x)

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    out_1 = Conv2DTranspose(3, (1, 32), 
            data_format='channels_first', padding='same', use_bias=False,
            kernel_initializer= glorot_normal())(x)

    #---------------------------------------------------------
    x = SeparableConv2D(32, (1, 16), 
        kernel_initializer= glorot_normal(), use_bias=False,
        padding='same',
        data_format='channels_first')(encode_1)

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    x = ELU()(x)
    orig_2 = x
    x = AveragePooling2D((1, 8), data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    out_2 = Dense(num_classes, activation='softmax', 
        kernel_constraint=max_norm(0.25))(x)
    model = Model(inputs=[input_tensor], outputs=[out_1, out_2])
    return model
