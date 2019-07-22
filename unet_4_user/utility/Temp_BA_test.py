#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:53:05 2019

@author: thibault
BA play around
test 
uncertainty by axon
"""

import os 
import sys
from keras.layers import Dropout
import matplotlib.pylab as plt
import numpy as np
    
from Baeysian_approx_tools import call_BA, call_default, BA_experiment, Entropy_fun, coef_variation_fun
from keras.models import load_model
from Apply_model import losses
from Apply_model import Unet_by_patches_2
from Utility import Import_image, Image_2_np

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

# import image
# option 1 Choose the test image of the dataset manually
#test_path = '/home/thibault/Documents/Thibault_Python_dev/UnAxSeg_Data/Data/dataset_demo/test/sample3/image.png'
test_path = '/home/thibault/Documents/Thibault_Python_dev/UnAxSeg_Data/Data/\
dataset_demo//sample5/image.png'
img_test, test_path = Import_image(filename = test_path)


# Sanity check
# function to run twice and compare the image... SANITY CHECK!!!
def test_model(model, img_test):
    fig, ax = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True, figsize=(15,10))
    ax[0].imshow(np.array(img_test))
    axon_mask_list=[]
    for i in range(2):
        
        prediction = Unet_by_patches_2(img_test, model, patch_size=256, overlap=64, RGB = True)
        mask = np.argmax(prediction, axis=-1)
        myelin_mask = np.array(mask==1, dtype= np.uint8)
        axon_mask = np.array(mask==2, dtype= np.uint8)
        axon_mask_list.append(axon_mask)
        
        ax[i+1].imshow(axon_mask)
        
    ax[3].imshow(axon_mask_list[0]-axon_mask_list[1])


# test the 22 imported model
test_model(model_BA, img_test)
    
test_model(model_no_BA, img_test)    

##############################################################################
# Perform the experiment
##############################################################################

prediction_tensor = BA_experiment (img_test, model_BA, n_sample = 20)
pred_no_BA = Unet_by_patches_2(img_test, model_no_BA, patch_size=256, overlap=64, RGB = True, verbose = False)

fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(15,10))
ax[0].imshow(pred_no_BA[:,:,0])
ax[1].imshow(pred_no_BA[:,:,1])
ax[2].imshow(pred_no_BA[:,:,2])


# Get the axon mask from the prediction
mask = np.argmax(pred_no_BA, axis=-1)
axon_mask = mask==2
plt.imshow(axon_mask)

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

Area_list= [region_i.area for region_i in axon_regions]
biggest = np.argmax(Area_list)
index = [region_i.label for index, region_i in enumerate(axon_regions)]

def mask_one_region(size, region, i):
    
    A  =  np.zeros(size, dtype=np.bool)
    region_i = regions_axon[i]
    coord =region_i.coords
    
    x = coord[:,0]
    y = coord[:,1]

    A[x, y] = 1
    
    return A

def display_uncertainty_i (uncertainty_axon, regions, i):
    
    size = uncertainty_axon.shape
    mask = mask_one_region(size, regions, 15)
    map_i = uncertainty_axon.copy()
    map_i[np.bitwise_not(mask)] = 0
    plt.imshow(map_i)

# Calculate the uncertainty metric Entropy or coefficient off variation
uncertainty = Entropy_fun(prediction_tensor) 
uncertainty = coef_variation_fun(prediction_tensor) 
uncertainty_axon = uncertainty[:,:,2] # uncertainty for channel 2 

uncertainty_i = uncertainty_axon.copy()
uncertainty_i_1 = np.mean(uncertainty_i[mask])

def coord_per_region(regions, i):
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
    
def Axon_uncertainty(uncertain_axon, regions, i):
    
    region_i = regions_axon[i]
    coord =region_i.coords
    x = coord[:,0]
    y = coord[:,1]

    uncertainty_i = uncertainty_axon.copy()
    uncertainty_i= np.mean(uncertainty_i[x,y])
    
    return uncertainty_i, x, y

uncertainty_i = Axon_uncertainty(uncertainty_axon,regions_axon, 15)
print(uncertainty_i)

Uncertainty_List =[Axon_uncertainty(uncertainty_axon,regions_axon,i)[0] for i,_ in enumerate(regions_axon) ]  

axon_uncertainy_map = np.zeros_like(axon_label, dtype=np.float)

for i, _ in enumerate(regions_axon):
    
    uncertainty_i, xi, yi = Axon_uncertainty(uncertainty_axon,regions_axon,i)
    axon_uncertainy_map[xi,yi] = uncertainty_i
    
fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize= (15,10))
ax[0].imshow(np.array(img_test))
ax[1].imshow(axon_mask)
ax[2].imshow(uncertainty_axon)
im= ax[3].imshow(axon_uncertainy_map, cmap=mycmap)
cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.5)

# uncertainty per axon pixel and median
uncertainty_pixel = uncertainty_axon[axon_mask]
median = np.median(uncertainty_pixel)
freq , bin_, _ = plt.hist(uncertainty_pixel, bins=50)
plt.vlines(median, 0, max(freq) ,colors='r')

##########################################################
# random color coding with black or white background
def ordered_cmap(N, base_cmap='nipy_spectral'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    color_list[0,:]=[0., 0., 0., 0.]
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

mycmap = ordered_cmap(310, base_cmap= 'cool')

A = np

base = plt.cm.get_cmap('cool')
color_list = base(np.linspace(0, 1, 310))

color_list = color_list[index_uncertain]
color_list[0,:]=[0., 0., 0., 0.]
cmap_name = base.name + str(310)
cmy_cmap = base.from_list(cmap_name, color_list, 310)


plt.imshow(axon_label, cmap=cmy_cmap)
plt.colorbar()


from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import measure
from scipy import ndimage as ndi

# Predicted axon
# Prepare label and regionprops 

img_test, test_path = Import_image()
pred_no_BA = Unet_by_patches_2(img_test, model_no_BA, patch_size=256, overlap=64, RGB = True, verbose = False)
mask = np.argmax(pred_no_BA, axis=-1)
a_mask = mask==2

axon= remove_small_objects(a_mask, min_size=9)
axon = ndi.binary_fill_holes(axon)
axon_label = measure.label(axon)
axon_regions = measure.regionprops(axon_label)


# MASK axon
# Prepare label and regionprops 
gt_mask, _ = Image_2_np(title='select ground truth mask')
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

features = ['pred_label', 'true_positive', 'gt_label', 'pred_area', 'gt_area', 'dice']
axon_df = pd.DataFrame(columns=features)#, dtype = 'float')

for i, axon in enumerate(axon_regions):
    
    # for the fiber
    diameter_a = 2*np.sqrt(axon.area/math.pi) # diameter in pixel
    axon_df.loc[i] = [axon.label, 0, 0, axon.area, 0, 0]
    
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
        

def Fiber_analysis(gt_fiber, pred_fiber, gt_axon, pred_axon, min_area=16, pixel_size = 0.11):
    """
    Compare fiber (axon+mylein) from ground truth and the prediction result. 
    Collect the morphologicall features for the axon and fiber and label if the a truth is detected
    
    features = ['coords', 'center','area', 'diameter', 'detected', 'dice', 'pred_centroid', 'pred_diameter', 'pred_area']
    
    input: gt_fiber: label image of groundtruth fiber
           pred_fiber: label image of predicted fiber
           gt_axon: label image of groundtruth axon
           gt_fiber: label image of predicted axon
           min_area: threshold for fiber minimun size in pixel
           
    """

    regions_fiber_pred = measure.regionprops(pred_fiber)
    
    regions_axon_pred = measure.regionprops(pred_axon)
    
    centroids = np.array([list(x.centroid) for x in regions_axon_pred])
    centroids = centroids.astype(int)
    areas = np.array([x.area for x in regions_axon_pred])
    centroids = centroids[areas > min_area]

    centroid_candidates = set([tuple(row) for row in centroids])
    dict_index = {(int(x.centroid[0]), int(x.centroid[1])):x.label for x in regions_fiber_pred}

    regions_fiber_gt = measure.regionprops(gt_fiber)
    regions_axon_gt = measure.regionprops(gt_axon)
    
    n_extra = 0

    #features = ['coords', 'area', 'center', 'diameter', 'detected']
    #true_axon_df = pd.DataFrame(columns=features, dtype = 'float')
    
    features = ['gt_label', 'true_positive', 'pred_label', 'gt_diameter','pred_diameter', 'dice', 'gt_area', 'pred_area']
    
    fiber_df = pd.DataFrame(columns=features)#, dtype = 'float')
    axon_df = pd.DataFrame(columns=features)#, dtype = 'float')
    
    for i, fiber in enumerate(regions_fiber_gt):
        
        # for the fiber
        diameter_f = 2*np.sqrt(fiber.area/math.pi) # diameter in pixel
        fiber_df.loc[i] = [fiber.label, 0, 0, diameter_f, 0, 0, fiber.area, 0]
        
        true_coords = set([tuple(row) for row in fiber.coords])
        fiber_center = np.array(fiber.centroid).astype(int)
        centroid_match = true_coords & centroid_candidates
        centroid_candidates = centroid_candidates.difference(centroid_match)
        centroid_match = list(centroid_match)
        
        # for the axon
        axon = regions_axon_gt[i]
        diameter_a = 2*np.sqrt(axon.area/math.pi)
        axon_df.loc[i] = [axon.label, 0, 0, diameter_a, 0, 0, axon.area, 0]
        
        true_coords_axon = set([tuple(row) for row in axon.coords])
        
        
        if len(centroid_match) != 0:
            diff = np.sum((centroid_match - fiber_center) ** 2, axis=1)
            ind = np.argmin(diff)
            center = centroid_match[ind]
            fiber_df.loc[i,'true_positive'] = 1
            axon_df.loc[i,'true_positive'] = 1
            
            # identify the predicted fiber correponding to the gt_fiber
            label  = dict_index[center]
            pred_fiber = regions_fiber_pred[label-1]
            fiber_df.loc[i,'pred_label'] = label
            axon_df.loc[i,'pred_label'] = label
            
            # for metric of the fiber
            # Calculate the dice
            pred_coords = set([tuple(row) for row in pred_fiber.coords])
            intersection = len(pred_coords & true_coords)
            sum_ =len(pred_coords) + len(true_coords)
            fiber_df.loc[i,'dice'] =2*intersection/sum_
            # other metric
            fiber_df.loc[i,'pred_diameter'] = 2*np.sqrt(pred_fiber.area/math.pi)
            fiber_df.loc[i,'pred_area'] =pred_fiber.area
            
            # for metric of the axon
            pred_axon = regions_axon_pred[label-1]
            # Calculate the dice
            pred_coords = set([tuple(row) for row in pred_axon.coords])
            intersection = len(pred_coords & true_coords_axon)
            sum_ =len(pred_coords) + len(true_coords_axon)
            axon_df.loc[i,'dice'] =2*intersection/sum_
            # other metric
            axon_df.loc[i,'pred_diameter'] = 2*np.sqrt(pred_axon.area/math.pi)
            axon_df.loc[i,'pred_area'] =pred_axon.area
            
            
            n_extra += len(centroid_match) - 1
            

    centroids_F = list(centroid_candidates)
    
    P = len(regions_fiber_gt)
    TP = sum(fiber_df['true_positive']==1)

    FP = len(centroids_F)

    #sensitivity = round(float(TP) / P, 3)
    # errors = round(float(FP) / P, 3)
    #diffusion = float(n_extra) / (TP + FP)
    #precision = round(float(TP) / (TP + FP), 3)
    
    return P, TP, FP, axon_df, fiber_df












