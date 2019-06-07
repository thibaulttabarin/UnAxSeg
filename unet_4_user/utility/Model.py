#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:01:34 2019

@author: thibault
Just Model
Architecture of unet
"""

import numpy as np

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.core import Dropout, Lambda
import keras.backend as K


def conv_bn_relu(s, features, kernel):
    
    c = Conv2D(features, kernel, kernel_initializer='he_normal', padding='same')(s)
    c = BatchNormalization(axis=3)(c)
    c = Activation('relu')(c)
    return c

def UnAxSeg(img_size, dropout = True):
    inputs = Input((img_size, img_size, 1))
    s = Lambda(lambda x: x / 255)(inputs)
    s = Lambda(lambda x: (x-K.mean(x)) /K.std(x))(s)

    c1 = conv_bn_relu (s, 16, (3,3))
    if dropout: c1 = Dropout(0.2)(c1)
    c1 = conv_bn_relu (c1, 16, (3,3))
    p1 = Conv2D(16, (2,2), strides = (2,2), kernel_initializer='he_normal')(c1)

    c2 = conv_bn_relu (p1, 32, (3,3))
    if dropout: c2 = Dropout(0.3)(c2)
    c2 = conv_bn_relu (c2, 32, (3,3))
    p2 = Conv2D(32, (2,2), strides =(2,2), kernel_initializer='he_normal')(c2)

    c3 = conv_bn_relu (p2, 64, (3,3))
    if dropout: c3 = Dropout(0.4)(c3)
    c3 = conv_bn_relu (c3, 64, (3,3))
    p3 = Conv2D(64, (2,2), strides =(2,2), kernel_initializer='he_normal')(c3)

    c4 = conv_bn_relu (p3, 128, (3,3))
    if dropout: c4 = Dropout(0.5)(c4)
    c4 = conv_bn_relu (c4, 128, (3,3))
    p4 = Conv2D(128, (2,2), strides =(2,2), kernel_initializer='he_normal')(c4)

    c5 = conv_bn_relu (p4, 256, (3,3))
    if dropout: c5 = Dropout(0.6)(c5)
    c5 = conv_bn_relu (c5, 256, (3,3))
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_bn_relu (u6, 128, (3,3))
    if dropout: c6 = Dropout(0.5)(c6)
    c6 = conv_bn_relu (c6, 128, (3,3))

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])    
    c7 = conv_bn_relu (u7, 64, (3,3))
    if dropout: c7 = Dropout(0.4)(c7)
    c7 = conv_bn_relu (c7, 64, (3,3))
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2]) 
    c8 = conv_bn_relu (u8, 32, (3,3))
    if dropout: c8 = Dropout(0.3)(c8)
    c8 = conv_bn_relu (c8, 32, (3,3))

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)    
    c9 = conv_bn_relu (u9, 16, (3,3))
    if dropout: c9 = Dropout(0.2)(c9)
    c9 = conv_bn_relu (c9, 16, (3,3))

    outputs = Conv2D(3, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model