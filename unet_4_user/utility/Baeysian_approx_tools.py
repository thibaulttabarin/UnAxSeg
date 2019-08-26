#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:08:27 2019

@author: thibault
Tools for Bayesian Approximation use

"""
import numpy as np
import sys
import os

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.losses import categorical_crossentropy
from keras import optimizers
import keras.backend as K

import keras
from keras.callbacks import TensorBoard

#path =  os.path.dirname(os.path.abspath(__file__)) 
#if not path in sys.path:
#    sys.path.append(path)

from Apply_model import Unet_by_patches_2


def unet_Bays_Approx(img_size, dropout= True):
    
    #learning_rate = 1e-2
    #nb_initial_epochs = 100
    #decay_rate = learning_rate / nb_initial_epochs
    dropout_proba = 0.5
    #dropout=True
    
    inputs = Input((img_size, img_size, 1))
    s = Lambda(lambda x: x / 255)(inputs)
    s = Lambda(lambda x: (x-K.mean(x)) /K.std(x))(s)
    
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    batch1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
    if dropout: pool1 = Dropout(dropout_proba)(pool1)


    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    batch2 = BatchNormalization(axis=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)
    if dropout: pool2 = Dropout(dropout_proba)(pool2)

   
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)
    batch3 = BatchNormalization(axis=3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch3)
    if dropout: pool3 = Dropout(dropout_proba)(pool3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv4)
    batch4 = BatchNormalization(axis=3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(batch4)
            
    pool4 = Dropout(dropout_proba)(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu")(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv5)
    if dropout:conv5 = Dropout(dropout_proba)(conv5)
    
    up6_interm = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6_interm, conv4], axis=3)

    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv6)
    batch6 = BatchNormalization(axis=3)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    if dropout: up7 = Dropout(dropout_proba)(up7)
    
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv7)
    batch7 = BatchNormalization(axis=3)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(batch7), conv2], axis=3)
    if dropout: up8 = Dropout(dropout_proba)(up8)

    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv8)
    batch8 = BatchNormalization(axis=3)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(batch8), conv1], axis=3)
    if dropout: up9 = Dropout(dropout_proba)(up9)

    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv9)
    batch9 = BatchNormalization(axis=3)(conv9)

    conv10 = Conv2D(3, (1, 1), activation="softmax")(batch9)

    model = Model(outputs=conv10, inputs=inputs)

    return model


def call_BA(self, inputs, training=None):
    """
    Override Dropout. Make it able at test time
    """
    if 0. < self.rate < 1.:
        noise_shape = self._get_noise_shape(inputs)
        def dropped_inputs():
            return K.dropout(inputs, self.rate, noise_shape,
                             seed=self.seed)
        if (training):
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        else:
            return K.in_test_phase(dropped_inputs, inputs, training=None)
    return inputs

def call_default(self, inputs, training=None):
    if 0. < self.rate < 1.:
        noise_shape = self._get_noise_shape(inputs)

        def dropped_inputs():
            return K.dropout(inputs, self.rate, noise_shape,
                            seed=self.seed)
        
        return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
    return inputs


def BA_experiment (img_PIL, model, n_sample = 20, verbose=False):
    '''    
    Run the BA (Baysian Approximation) model several time (n_sample) and create a tensor
    (n_sample,size_1,size_2,n_ch). 
    Warming during the run of one image, the image is process patch by patch therefore each patch of
    the image is not process by the same network. This result has to considered as a whole tensor.
    TODO : find a way of blocking the state of network for each run of Unet_by_patches
    input :
        - img_PIL: image in PIL format
        - model : keras.model (with weight and dropout.call set for BA)
        - n_sample : number predicted images
    output :
        - prediction as tensor (n_sample,size_1,size_2,n_ch)
    '''
    prediction =[]
    for i in range(n_sample):
        
        pred_i = Unet_by_patches_2(img_PIL, model, patch_size=256, overlap=64, RGB = True, verbose=verbose)
        prediction.append(pred_i)
    
    prediction =np.array(prediction)
    return prediction

def coords_per_region(regions, i):
    '''
    return the coordinate of the pixels from region i
    input : regionprops and i (index of the region of interest)
    output : tuple (x, y)
    '''  
    region_i = regions[i]
    coord =region_i.coords
    x = coord[:,0]
    y = coord[:,1]
    return x, y

def Entropy_fun(prediction_tensor):
    '''
    Calculate entropy using prediction tensor that encapuslate of teh BA experiment
    input : 
        - prediction_tensor : output of the function BA_experiment
    output :
        - Entropy: -p*log(p)
    '''
    
    Entropy = -prediction_tensor * np.log(prediction_tensor)
    
    return np.sum(Entropy,axis=0)

def coef_variation_fun (prediction_tensor):
    '''
    Calculate the coefficient of variation.
    input : prediction tensor from 
    outpu : coefficient of variation std/mean
    '''
    
    return np.std(prediction_tensor, axis=0)/np.mean(prediction_tensor, axis=0)

