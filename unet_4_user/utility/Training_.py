#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:40:33 2019

@author: thibault
"""
import numpy as np
import matplotlib.pyplot as plt
import os

import json
from keras.models import model_from_json
import datetime

import Augmentor

from Utility import *
from Utility_Create_Dataset import *
import Augmentor_add_on as aug_addon

os.chdir('/home/thibault/Documents/Thibault_Python_dev/Unet_TT')
from Utility import *
from Unet_tt_v2 import *



###############################################################
#prepare the data and generator
# Path to training data

train_path = 'Dataset_demo/Patch_for_Training/train'
    
Data_train = get_data(train_path)
# create the generator
img, mask = Data_train[0]
fig, ax =plt.subplots(nrows=1,ncols=3)
ax[0].imshow(img)
ax[1].imshow(mask)
ax[2].imshow(mask)


p = Augmentor.DataPipeline(Data_train)

# =============================================================================
# =============================================================================
#p.shear(0.75,max_shear_left=10, max_shear_right=10)

p.rotate(probability=0.9, max_left_rotation=15, max_right_rotation=15)
p.rotate90(probability=0.8)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.random_distortion(probability=1,grid_width=1, grid_height=1, magnitude=8)
p.shear(0.9,max_shear_left=10, max_shear_right=10)
#p.histogram_equalisation(probability =1.0)

mask_3ch = aug_addon.Mask_3ch() # transform the masks in 3 channels with 0 or 1 value (hotshot)
p.add_operation(mask_3ch)

# generate a batch in keras format (channel last) : tuple (image, mask) (batch_size, width, height, channel)
g = p.keras_generator_with_mask(batch_size=10)

a = next(g)
imgs = a[0]
masks = a[1]
fig, ax =plt.subplots(nrows=1,ncols=3)
ax[0].imshow(imgs[0,:,:,0])
ax[1].imshow(masks[0,:,:,1])
ax[2].imshow(masks[0,:,:,2])

# Validation
validation_path = '/home/thibault/Documents/Data/ADS/Training_256_v6a/dataset/Validation'

Data_val = get_data(validation_path)
# create the generator
v = Augmentor.DataPipeline(Data_val)

# =============================================================================
# =============================================================================
#p.shear(0.75,max_shear_left=10, max_shear_right=10)

v.rotate(probability=0.9, max_left_rotation=15, max_right_rotation=15)
v.rotate90(probability=0.8)
v.flip_left_right(probability=0.5)
v.flip_top_bottom(probability=0.5)
v.random_distortion(probability=1,grid_width=1, grid_height=1, magnitude=8)
v.shear(0.9, max_shear_left=10, max_shear_right=10)
#p.histogram_equalisation(probability =1.0)

mask_3ch = tb_add.Mask_3ch() # transform the masks in 3 channels with 0 or 1 value (hotshot)
v.add_operation(mask_3ch)

# generate a batch in keras format (channel last) : tuple (image, mask) (batch_size, width, height, channel)
val = p.keras_generator_with_mask(batch_size=10)


def Generator_Augmented_Data(Data_train):
    
    p = Augmentor.DataPipeline(Data_train)
    
    p.rotate(probability=0.9, max_left_rotation=15, max_right_rotation=15)
    p.rotate90(probability=0.8)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.random_distortion(probability=1,grid_width=1, grid_height=1, magnitude=8)
    p.shear(0.9,max_shear_left=10, max_shear_right=10)
    
    mask_3ch = aug_addon.Mask_3ch() # transform the masks in 3 channels with 0 or 1 value (hotshot)
    p.add_operation(mask_3ch)
    
    g =   p.keras_generator_with_mask(batch_size=10)
    return g
    

g = Generator_Augmented_Data(Data_train)

a = next(g)
imgs = a[0]
masks = a[1]
fig, ax =plt.subplots(nrows=1,ncols=3)
ax[0].imshow(imgs[0,:,:,0])
ax[1].imshow(masks[0,:,:,1])
ax[2].imshow(masks[0,:,:,2])



###################################################################
# Training phase
# Create a name for the model witht the date and time of the training
currentDT = datetime.datetime.now()
date_time = currentDT.strftime("%Y-%m-%d_%H%M")
model_path = '/home/thibault/Documents/Data/ADS/Model_Unet_tt/'+ date_time +'.h5'

#train_Unet_v2 (img_size= 256, generator=g, steps_per_epoch=10, epochs=400, model_path=model_path)

#train_Unet_v3 (img_size= 256, generator=g, steps_per_epoch=20, epochs=400, model_path=model_path)

train_Unet_v4 (img_size= 256, train_generator=g, val_generator=val,
                   steps_per_epoch=10, epochs=600,
                   model_path=model_path)

