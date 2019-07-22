#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:19:34 2019

@author: thibault
Development compare ground truth segmentation and prediction for axon only
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

#############################################################################################################
####### Function for gt vs pred comparison and performance metrics
#############################################################################################################
def region_axon_fun(axon_mask,min_size=9):
    '''
    Function that: 
        clean the axon mask (remove small object and fill hole)
        find individual object and label
        calculate the regionprops for each label object
    input : mask (boolen array)
            min_size object allow
    output : regionprop (list) and labels map (2d array)
        
    '''
    axon= remove_small_objects(axon_mask, min_size=9)
    axon = ndi.binary_fill_holes(axon)
    axon_label = measure.label(axon)
    axon_regions = measure.regionprops(axon_label)
    
    return axon_regions, axon_label
    
def center_set(regions):
    '''
    Create a set with the center
    '''
    center_ = np.array([np.round(x.centroid) for x in regions], dtype=np.int )
    center = set([tuple(row) for row in center_])
    
    return center

def compare_axon_pred_gt(pred_axon_regions, gt_axon_regions):
    
    '''
    identify the axons matching between ground truth and predicted
    '''
    
    gt_center_set = center_set(gt_axon_regions)
    dict_index = {(int(np.round(x.centroid[0])), int(np.round(x.centroid[1]))):x.label for x in gt_axon_regions}
    missing=0

    features = ['pred_label', 'true_positive', 'gt_label', 'pred_center', 'gt_center', 'pred_area', 'gt_area', 'dice']
    axon_df = pd.DataFrame(columns=features)#, dtype = 'float')

    for i, axon in enumerate(pred_axon_regions):
    
        diameter_a = 2*np.sqrt(axon.area/math.pi) # diameter in pixel
        pred_center = list(np.array(np.round(axon.centroid), dtype=np.int))
        axon_df.loc[i] = [axon.label, 0, 0,pred_center,[0,0], axon.area, 0, 0]
    
        axon_coords = set([tuple(row) for row in axon.coords])
        axon_center = np.array(axon.centroid).astype(int)
        center_match = axon_coords & gt_center_set
        gt_center_set = gt_center_set.difference(center_match)
        center_match = list(center_match)
    
        if len(center_match) !=0:
        
            # get the closest gt_center from pred_center
            diff = np.sum((center_match - axon_center) ** 2, axis=1)
            ind = np.argmin(diff)
            center = center_match[ind]
            
            axon_df.loc[i,'true_positive'] = 1
        
            # identify the predicted axon correponding to the gt_axon
            label  = dict_index[center]
            gt_axon_ = gt_axon_regions[label-1]
            
            axon_df.loc[i,'gt_label'] = label
            axon_df.loc[i,'gt_area'] = gt_axon_.area
            axon_df.loc[i,'gt_center'] = list(center)
        
            # for metric of the fiber
            # Calculate the dice
            gt_coords = set([tuple(row) for row in gt_axon_.coords])
            intersection = len(gt_coords & axon_coords)
            sum_ =len(axon_coords) + len(gt_coords)
            axon_df.loc[i,'dice'] =2*intersection/sum_
        
    return axon_df

def Performance_metric(axon_df, gt_axon_regions):
    '''
    Calculate the performance metric:
        True positive, False Positive, False Negative, Recall, Precision, DICE
    '''
    gt_center_set = center_set(gt_axon_regions)

    gt_center_detected = np.array(axon_df['gt_center'][axon_df['true_positive']==1].tolist())    
    gt_center_detected = set([tuple(row) for row in gt_center_detected]) 

    TP_set = np.array(axon_df['pred_center'][axon_df['true_positive']==1].tolist())    
    TP_set = set([tuple(row) for row in TP_set]) 

    FP_set = np.array(axon_df['pred_center'][axon_df['true_positive']==0].tolist())    
    FP_set = set([tuple(row) for row in FP_set]) 

    FN_set = gt_center_set.difference(gt_center_detected)

    TP_arr = np.array([[x[0],x[1]] for x in TP_set])
    FP_arr = np.array([[x[0],x[1]] for x in FP_set])
    FN_arr = np.array([[x[0],x[1]] for x in FN_set])
    
    return TP_arr, FP_arr, FN_arr

def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    '''
    Calculate DICE value for the whole image
    '''
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


###################################################################
# Class to crop a big image
class Correct_Segmentation(object):
    '''
    Class object used to create mask for big image
    When the object is created from an image, the image is resize (smaller size) to create and manipulate the mask faster
    the function  get_nask create a figure with cursor to draw a polygone region of interest
    the function cropping 
    
    '''

    def __init__(self, img_test, prediction, gt_mask):
        
        self.img = img_test
        self.prediction = prediction
        self.gt = gt_mask   
        self.point=[]
        self.get_axons()
        self.get_perf_metric()
        self.corrected_gt = np.array(gt_mask, dtype=float)
        
    def get_axons(self):
        
        prediction = self.prediction
        gt_mask = self.gt
        
        pred_regions, pred_labels =   region_axon_fun(prediction,min_size=1)
    
        # Prepare label and regionprops 
        gt_regions, gt_labels =   region_axon_fun(gt_mask,min_size=1)
    
        # Compare the prediction and groundtruth
        axon_df = compare_axon_pred_gt(pred_regions, gt_regions)
    
        # Calculate performance metric
        
        
        self.dataframe=axon_df
        
        self.pred_regions= pred_regions
        self.gt_regions= gt_regions
        self.pred_labels= pred_labels
        self.gt_labels= gt_labels
    
    class _Metric:
        
        def __init__(self, TP_arr, FP_arr, FN_arr, Recall, Precision, Image_Dice):
            
            self.TP_arr = TP_arr
            self.FP_arr = FP_arr
            self.FN_arr = FN_arr
            self.Recall = Recall
            self.Precision = Precision
            self.Image_Dice = Image_Dice
        
        def print_metric(self):
            
            print ('Recall:{0}\nPrecision:{1}\nImage Dice:{2}'.format(self.Recall, self.Precision, self.Image_Dice))
        
    def get_perf_metric(self):
        
        axon_df = self.dataframe
        gt_regions= self.gt_regions
        prediction = self.prediction
        gt_mask = self.gt
        
        TP_arr, FP_arr, FN_arr = Performance_metric(axon_df, gt_regions)
        self.perf_metric= (TP_arr, FP_arr, FN_arr)
        
        Recall = len(TP_arr)/(len(TP_arr)+len(FN_arr))
        Precision = len(TP_arr)/(len(TP_arr)+len(FP_arr))
        Image_Dice = hard_dice_coef(gt_mask, axon_mask)
        
        self.Metric= self._Metric(TP_arr, FP_arr, FN_arr, Recall, Precision, Image_Dice)
                
    def Create_plot(self):
        
        img_test = self.img
        data_prediction = np.array(self.prediction, dtype= np.float)
        data_gt = np.array(self.gt, dtype= np.float)
        TP_arr = self.Metric.TP_arr
        FP_arr = self.Metric.FP_arr
        FN_arr = self.Metric.FN_arr

        
        fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(15,10))
        
        self.img_0= ax[0].imshow(np.array(img_test))
        ax[0].scatter(TP_arr[:,1],TP_arr[:,0], s=3, color='r')
        ax[0].scatter(FP_arr[:,1],FP_arr[:,0], s=3, color='g')
        ax[0].scatter(FN_arr[:,1],FN_arr[:,0], s=3, color='k')

        self.img_1 = ax[1].imshow(data_prediction)
        ax[1].scatter(TP_arr[:,1],TP_arr[:,0], s=3, color='r')
        ax[1].scatter(FP_arr[:,1],FP_arr[:,0], s=3, color='g')
    
        ax[2].imshow(data_gt)
        ax[2].scatter(TP_arr[:,1],TP_arr[:,0], s=3, color='r')
        ax[2].scatter(FN_arr[:,1],FN_arr[:,0], s=3, color='k')
        

        self.fig=fig
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        
    def onclick(self,event):
        
        p=self.point
        pred_regions = self.pred_regions
        gt_regions = self.gt_regions       
        
        if event.dblclick :
            point = list(map (int,np.round([event.xdata, event.ydata])))
            print('xdata=%f, ydata=%f' % ( event.xdata, event.ydata))
            print('x=%f, y=%f' % ( point[0], point[1]))
            p.append(point)
            
            pixels = self.find_corresponding_axon(point)
            
            self.add_new_axon_to_gt(pixels)
            
            self.display_new_axon_in_gt()
                       
    def find_corresponding_axon(self, point):
        
        pred_labels = self.pred_labels
        gt_labels = self.gt_labels
        pred_regions = self.pred_regions
        gt_regions = self.gt_regions   
        
        pred_label = pred_labels[point[1],point[0]]
        gt_label = gt_labels[point[0],point[1]]
        
        pred_object = pred_regions[pred_label-1]
        
        pixels = pred_object.coords
        
        return pixels
    
    def add_new_axon_to_gt(self, pixels):
        
        correct_gt = self.corrected_gt
        correct_gt[pixels[:,0], pixels[:,1]] = 2
        self.corrected_gt = correct_gt
        
    def display_new_axon_in_gt(self):
        
        correct_gt = self.corrected_gt
        myobj = self.fig.axes[2]
        myobj.imshow(correct_gt)


#############################################################################################################
model_path = '/home/thibault/Documents/Thibault_Python_dev/UnAxSeg/unet_4_user/Models/2019-06-18_1018.h5'
model_ = load_model(model_path, custom_objects={'softmax_dice_loss_2': losses.softmax_dice_loss_2,
                                               'dice_coef_ch1':losses.dice_coef_ch1,
                                              'dice_coef_ch2':losses.dice_coef_ch2})

path_img_test = '/home/thibault/Documents/Data/ADS/18_06_CC_S5_ADS_Training/manual_segmentation/20190626__18_06_CC_S5_good_2_croped.jpg'
path_gt_mask = '/home/thibault/Documents/Data/ADS/18_06_CC_S5_ADS_Training/manual_segmentation/20190626__18_06_CC_S5_good_2_croped_full_Seg.png'
# Import image and ground truth mask
img_test, test_path = Import_image(filename = path_img_test ,title='select image to predict')
gt_mask, _ = Image_2_np(filename = path_gt_mask, title='select ground truth mask')

# Predict
pred_ = Unet_by_patches_2(img_test, model_, patch_size=256, overlap=64, RGB = True, verbose = True)
mask = np.argmax(pred_, axis=-1)
axon_mask = mask==2
gt_mask = gt_mask > 200


#############################################################################################################
# interactive correction
#############################################################################################################

correction_session = Correct_Segmentation(img_test, axon_mask, gt_mask)
correction_session.Metric.print_metric()
correction_session.Create_plot()

#############################################################################################################
# When you are satisfied with the correction
corrected_gt = correction_session.corrected_gt
new_gt_mask = corrected_gt > 0
correction_session = Correct_Segmentation(img_test, axon_mask, new_gt_mask)
correction_session.Metric.print_metric()
correction_session.Create_plot()








#############################################################################################################
# Predicted axon
# Prepare label and regionprops 
axon_regions, _ =   region_axon_fun(axon_mask,min_size=1)

# MASK axon
# Prepare label and regionprops 
gt_a_mask = gt_mask > 200
gt_axon_regions, _ =   region_axon_fun(gt_a_mask,min_size=1)

# Compare the prediction and groundtruth
axon_df = compare_axon_pred_gt(axon_regions, gt_axon_regions)

# Calculate performance metric
TP_arr, FP_arr, FN_arr = Performance_metric(axon_df, gt_axon_regions)

# Print performance metric
Recall = len(TP_arr)/(len(TP_arr)+len(FN_arr))
Precision = len(TP_arr)/(len(TP_arr)+len(FP_arr))
print('Recall: {}'.format(Recall))
print('Precision: {}'.format(Precision))

Image_Dice = hard_dice_coef(gt_a_mask, axon_mask)
print('Dice per image: {}'.format(Image_Dice) )

pred_center_val = np.array(axon_df['pred_center'][axon_df['true_positive']==1].tolist())
gt_center_val = np.array([list(x) for x in center_set(gt_axon_regions)])

fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(15,10))
ax[0].imshow(np.array(img_test))
ax[0].scatter(TP_arr[:,1],TP_arr[:,0], s=3, color='r')
ax[0].scatter(FP_arr[:,1],FP_arr[:,0], s=3, color='g')
ax[0].scatter(FN_arr[:,1],FN_arr[:,0], s=3, color='k')

ax[1].imshow(pred_[:,:,2])
ax[1].scatter(TP_arr[:,1],TP_arr[:,0], s=3, color='r')
ax[1].scatter(FP_arr[:,1],FP_arr[:,0], s=3, color='g')

ax[2].imshow(gt_a_mask)
ax[2].scatter(TP_arr[:,1],TP_arr[:,0], s=3, color='r')
ax[2].scatter(FN_arr[:,1],FN_arr[:,0], s=3, color='k')



#############################################################################################################
# interactive correction
#############################################################################################################






        








