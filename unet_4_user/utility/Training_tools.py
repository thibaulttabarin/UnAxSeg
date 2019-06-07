#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:20:04 2019

@author: thibault
"""

from time import time
import numpy as np

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.core import Dropout, Lambda
from keras.losses import categorical_crossentropy
from keras import optimizers
import keras.backend as K
import keras
from keras.callbacks import TensorBoard
import utility.losses as losses


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


def train_UnAxSeg (img_size= 256, train_generator=None, val_generator=None,
                   steps_per_epoch=10, epochs=100,
                   model_path=None, tensorboard=None):
    
    '''
    Set the model, optimizer and fit_generator for training phase and start the training
    save the model
    imput :
        img_size: should match the image size from the generator
        train_generator: should be an interator create by Generator_Augmented_Data for instance
        val_generator: same as train_generator by for validation
        steps_per_epoch:
        epochs:
        model_path: where to save the model
        tensorboard: keras.callbacks.Tensorboard(log_dir=log_path) 
    output: train model
    '''
    
    model = UnAxSeg(img_size)
    
    loss_fun =  losses.softmax_dice_loss_2
    #loss_fun =  losses.double_head_loss
    #loss_fun =  losses.mask_contour_mask_loss
    #loss_fun = losses.softmax_dice_loss
    #loss_fun = categorical_crossentropy
    
    # Optimizer
    opti_adam = optimizers.adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=True)
    
    # metrics
    dice_myelin=losses.dice_coef_ch1
    dice_axon =losses.dice_coef_ch2
    
    model.compile(optimizer=opti_adam, loss=loss_fun, metrics=['accuracy', dice_myelin, dice_axon])
        
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_images =True)
    model.fit_generator(train_generator, steps_per_epoch= steps_per_epoch, epochs=epochs,
                        validation_data = val_generator,
                        validation_steps = steps_per_epoch //2,
                        callbacks= [tensorboard])
    
    model.save(model_path)
    
    return model


import Augmentor
import utility.Augmentor_add_on as aug_addon

def Generator_Augmented_Data(Data_train, batch_size=10):
    
    p = Augmentor.DataPipeline(Data_train)
    
    p.rotate(probability=0.9, max_left_rotation=15, max_right_rotation=15)
    p.rotate90(probability=0.8)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.random_distortion(probability=1,grid_width=1, grid_height=1, magnitude=8)
    p.shear(0.9,max_shear_left=10, max_shear_right=10)
    
    mask_3ch = aug_addon.Mask_3ch() # convert the masks in 3 channels with 0 or 1 value (hotshot)
    p.add_operation(mask_3ch)
    
    g =   p.keras_generator_with_mask(batch_size=batch_size)
    return g








