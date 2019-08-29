#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:01:15 2019

@author: thibault
Correct the ground truth using the prediction.
In some case the prediction detects axons that have been forgotten by the expert during the manual segmentation.
using this pipeline you can correct semi automatically the ground truth during a second step using gimp you will able to fine tune
the correction. this pipeline will work better using spyder.
run this code line by line

"""
import os
from unet_4_user.utility.Axon_Metrics_Correction_utility import *
from unet_4_user.utility.Utility import Import_image, Image_2_np
from unet_4_user.utility.ImageIO import *

#############################################################################################################
#  import the prediction 
experience_path = '/test_share/unet_experiments/evaluation_experiments/dice_axon_vs_myelin_experiments/\
loss_dice_50_ce_50_axon_10_myelin_90_aug_thibault_expand_model_unet_normalized'

original_image = '00007_image_original.png'
original_path = os.path.join(experience_path, original_image)
original_img, _ = Import_image(filename = original_path ,title='select image to predict')
original_img = read_input_image(original_path, output_mode='gray')

pred_mask_image = '00007_image_prediction_all.png'
pred_mask_path = os.path.join(experience_path, pred_mask_image)

gt_mask_image = '00007_image_ground_truth_axon_only.png'
gt_mask_path =os.path.join(experience_path, gt_mask_image)

##########################################################################################################
#
original_img = read_input_image(original_path, output_mode='gray')

pred_mask =read_mask_3c(pred_mask_path)
pred_mask_axon = pred_mask ==2

gt_mask = read_mask_3c(gt_mask_path)
gt_mask_axon = gt_mask ==2

#############################################################################################################
# correct the gt_mask
# Instruction 
# white dots true positive
# red dots Faslse positive (detected by the network not by expert)
# black dots False negative (detected by expert not by the network)
# tool in the upper left corner to nagivate zoom the image
# left : original image, middle prediction, right ground truth
# to add a false positive to th ground truth double click on the axon

correction_session = Correct_Segmentation(original_img, pred_mask_axon, gt_mask_axon)
correction_session.Create_plot()

#############################################################################################################
# save the corrected mask
path_to_save =""
name = 'mask_axon_correct'
mask_save_path = os.path.join(path_to_save,name)
correction_session.save_corrected_gt(mask_save_path)

#############################################################################################################
# 
'''
Open the newly create corrected mask in gimp
the old axon are in gray
the newly added axons are in white 
and then you can edit the mask in gimp
'''

################

read_input_image()
read_input_image

