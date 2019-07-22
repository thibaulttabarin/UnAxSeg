#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:33:10 2019

@author: thibault
"""

import os 
import sys
import math
from keras.layers import Dropout
from keras.models import load_model
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import measure
from skimage.measure import regionprops
from scipy import ndimage as ndi


from utility.Baeysian_approx_tools import call_BA, call_default, BA_experiment, Entropy_fun, coef_variation_fun
from utility.Apply_model import losses,  Unet_by_patches_2
from utility.Utility import Import_image, Image_2_np

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


# Import image and ground truth mask
img_test, test_path = Import_image('')
gt_mask, _ = Image_2_np(title='select ground truth mask')


# Create prediction and BA_prediction
prediction_tensor = BA_experiment (img_test, model_BA, n_sample = 20, verbose=True)
pred_no_BA = Unet_by_patches_2(img_test, model_no_BA, patch_size=256, overlap=64, RGB = True, verbose = True)
mask = np.argmax(pred_no_BA, axis=-1)
axon_mask = mask==2

# Calculate the uncertainty metric Entropy or coefficient off variation
uncertainty = Entropy_fun(prediction_tensor) 
uncertainty = coef_variation_fun(prediction_tensor) 
uncertainty_axon = uncertainty[:,:,2] # uncertainty for channel 2 


# uncertainty per axon pixel and median
uncertainty_pixel = uncertainty_axon[axon_mask]
median = np.median(uncertainty_pixel)
freq , bin_, _ = plt.hist(uncertainty_pixel, bins=100)
plt.vlines(median, 0, max(freq) ,colors='r')


def Axon_uncertainty(uncertain_axon, regions, i):
    
    region_i = regions[i]
    coord =region_i.coords
    x = coord[:,0]
    y = coord[:,1]

    uncertainty_i = uncertainty_axon.copy()
    uncertainty_i= np.mean(uncertainty_i[x,y])
    
    return uncertainty_i, x, y

# Prepare the axon label and regionprops
axon= remove_small_objects(axon_mask, min_size=9)
axon = ndi.binary_fill_holes(axon)
axon_label = measure.label(axon)
axon_regions = measure.regionprops(axon_label)


Uncertainty_List =[Axon_uncertainty(uncertainty_axon,axon_regions,i)[0] for i,_ in enumerate(axon_regions) ]  


#############################################################################################################
# Predicted axon
# Prepare label and regionprops 

# MASK axon
# Prepare label and regionprops 
gt_a_mask = gt_mask > 200

gt_axon= remove_small_objects(gt_a_mask, min_size=9)
gt_axon = ndi.binary_fill_holes(gt_axon)
gt_axon_label = measure.label(gt_axon)
gt_axon_regions = measure.regionprops(gt_axon_label)


gt_center_ = np.array([list(x.centroid) for x in gt_axon_regions], dtype=np.int )
#gt_center_2 = np.array([list(x.centroid) for x in gt_axon_regions])
gt_center = set([tuple(row) for row in gt_center_])

#areas = np.array([x.area for x in  axon_regions])
#centroids = centroids[areas > 16]

dict_index = {(int(x.centroid[0]), int(x.centroid[1])):x.label for x in gt_axon_regions}
n_extra = 0

features = ['pred_label', 'true_positive', 'gt_label', 'pred_area', 'gt_area', 'dice', 'uncertainty']
axon_df = pd.DataFrame(columns=features)#, dtype = 'float')

for i, axon in enumerate(axon_regions):
    
    # for the fiber
    diameter_a = 2*np.sqrt(axon.area/math.pi) # diameter in pixel
    axon_df.loc[i] = [axon.label, 0, 0, axon.area, 0, 0, Uncertainty_List[i]]
    
    axon_coords = set([tuple(row) for row in axon.coords])
    axon_center = np.array(axon.centroid).astype(int)
    center_match = axon_coords & gt_center
    gt_center = gt_center.difference(center_match)
    center_match = list(center_match)
    
    if len(center_match) !=0:
        
        diff = np.sum((center_match - axon_center) ** 2, axis=1)
        ind = np.argmin(diff)
        center = center_match[ind]
        axon_df.loc[i,'true_positive'] = 1
        
        # identify the predicted fiber correponding to the gt_fiber
        label  = dict_index[center]
        gt_axon_ = gt_axon_regions[label-1]
        axon_df.loc[i,'gt_label'] = label
        axon_df.loc[i,'gt_area'] = gt_axon_.area
        
        # for metric of the fiber
        # Calculate the dice
        gt_coords = set([tuple(row) for row in gt_axon_.coords])
        intersection = len(gt_coords & axon_coords)
        sum_ =len(axon_coords) + len(gt_coords)
        axon_df.loc[i,'dice'] =2*intersection/sum_



dice = np.array(axon_df['dice'])
Uncertainty = np.array(axon_df['uncertainty'])
plt.scatter(dice,Uncertainty)




























