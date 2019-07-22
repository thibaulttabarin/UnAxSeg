#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:23:27 2019

@author: thibault
Advance version for Baeysian approximation experiment
1- Calculate prediction without BA
2- Calculate prediction with BA for n_sample
3- Calculate the Uncertainty metric (Entropy, coefficient of variation) using multiple prediction approximation
4- Use the axon mask from prediction without BA 
Identify axon label and regions properties per axon
Extract the uncertainty metric per axon 
Compare the score of uncertainty with groundtruth
"""

import os 
from keras.layers import Dropout
import matplotlib.pylab as plt
import numpy as np

os.chdir('/home/thibault/Documents/Thibault_Python_dev/UnAxSeg/unet_4_user/utility')
from Baeysian_approx_tools import call_BA, call_default, BA_experiment, coords_per_region, Entropy_fun, coef_variation_fun
from keras.models import load_model
from Apply_model import losses
from Apply_model import Unet_by_patches_2, Create_mask
from Utility import Import_image

# import models with and without BA
# With BA
Dropout.call=call_BA
model_path = '/home/thibault/Documents/Thibault_Python_dev/UnAxSeg/unet_4_user/Models/2019-06-18_1018.h5'
model_BA = load_model(model_path, custom_objects={'softmax_dice_loss_2': losses.softmax_dice_loss_2,
                                               'dice_coef_ch1':losses.dice_coef_ch1,
                                              'dice_coef_ch2':losses.dice_coef_ch2})
# Without BA
Dropout.call=call_default
model_path = '/home/thibault/Documents/Thibault_Python_dev/UnAxSeg/unet_4_user/Models/2019-06-14_1451.h5'
model_no_BA = load_model(model_path, custom_objects={'softmax_dice_loss_2': losses.softmax_dice_loss_2,
                                               'dice_coef_ch1':losses.dice_coef_ch1,
                                              'dice_coef_ch2':losses.dice_coef_ch2})


# Import imag and mask
test_path = '/home/thibault/Documents/Thibault_Python_dev/UnAxSeg_Data/Data/\
dataset_demo/test/sample5/image.png'
img_test, test_path = Import_image(filename = test_path)

mask_path = '/home/thibault/Documents/Thibault_Python_dev/UnAxSeg_Data/Data/\
dataset_demo/test/sample5/mask.png'
mask_test, _ = Import_image(filename = mask_path)

# Create the prediction and BA prediction
BA_prediction = BA_experiment (img_test, model_BA, n_sample = 20)
prediction = Unet_by_patches_2(img_test, model_no_BA, patch_size=256, overlap=64, RGB = True, verbose = False)

axon_mask = Create_mask(prediction, ch=2)


##############################################################################
# identify the axons regions/labels
##############################################################################

from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import measure
from scipy import ndimage as ndi

# Prepare the axon label and regionprops
axon= remove_small_objects(axon_mask, min_size=3)
axon = ndi.binary_fill_holes(axon)
axon_label = measure.label(axon)
axon_regions = measure.regionprops(axon_label)


# Calculate the uncertainty metric Entropy or coefficient off variation
uncertainty = Entropy_fun(BA_prediction) 
uncertainty = coef_variation_fun(BA_prediction) 
uncertainty_axon = uncertainty[:,:,2] # uncertainty for channel 2 

# Calculate the mean uncertainty value per axon
def mean_uncertain_per_region(uncertainty, regions):
    '''
    mean of uncertainty for each individual region 
    '''

    list_uncertainty=[]
    for i, region_i in enumerate(regions):
        
        x, y = coords_per_region(regions, i)
        
        list_uncertainty.append(np.mean(uncertainty[x,y]))
        
    return list_uncertainty

list_uncertainty = mean_uncertain_per_region(uncertainty,axon_regions)


# uncertainty per axon pixel and median
uncertainty_pixel = uncertainty_axon[axon_mask]
median = np.median(uncertainty_pixel)
freq , bin_, _ = plt.hist(uncertainty_pixel, bins=100)
plt.vlines(median, 0, max(freq) ,colors='r')
plt.vlines(median*1.48, 0, max(freq) ,colors='g')










