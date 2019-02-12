#!/usr/bin/env python

"""
Neural network used in the study.
"""

import keras
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, Input, Dense, Activation
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2


def mycnn(imsize, nf=32, reg = 5e-5, padding='valid'):
    """Return a 512 pixel CNN."""
    # input
    inp = Input(shape=(imsize, imsize, 1))

    # conv block
    x = Conv2D(nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(inp)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(2*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(2*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(2*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(4*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D( 8*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D( 8*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D( 8*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(x)
    
    # final regression
    x = GlobalAveragePooling2D()(x)
    x = Dense(2)(x)
        
    model = Model(inputs=inp, outputs=x)
    return model