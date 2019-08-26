#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:01:15 2019

@author: thibault
Test for Metrics and correction utility
"""


from Axon_Metrics_Correction_utility import *
from Utility import Import_image, Image_2_np

from Apply_model import losses,  Unet_by_patches_2
from keras.models import load_model


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
correction_session = Correct_Segmentation(img_test, axon_mask, gt_mask)


pred_mask = correction_session.prediction
gt_mask = correction_session.gt

Recall, Precision, Image_Dice = calculate_metrics(pred_mask, gt_mask)


##############################################################################################################
evatuation_sess = Evaluation(axon_mask,gt_mask)
evatuation_sess.Metric.print_metric()

#########################################################################
#########################################################################
from unet_4_user.utility.utility_plot import *


# Choose the feature you want to plot
feature = 'Axon area' # from 'Axon diameter', 'Gratio' or 'Myelin Thickness'

#measured_option = {'Axon diameter':'diam_a', 'Gratio':'gratio', 'Myelin Thickness':'myelin_thick'}
measured_option = {'Axon area':'area'}

pred = 'pred_'+ measured_option[feature]
gt = 'gt_'+ measured_option[feature]
measured = feature
# define unit: '${\mu}m$' for Axon diameter and Myelin Thickness
#                '' for g ratio
unit = 'pixel'

# plot 1
title_1 = 'Predicted {0} distribution \n TP and FP'. format(measured)
x_label_1 = 'Predicted {0} {1}'. format(measured, unit)
y_label_1 = 'Percentage of fiber'

pred_xx_TP =df.loc[df['true_positive']==1, pred]
pred_xx_FP =df.loc[df['true_positive']==0, pred]
pred_xx = df[pred]
bins,fig = bar_plot_dist_2_categ(pred_xx, pred_xx_TP, pred_xx_FP, 
                                 legend=('TP','FP'), bins=50, 
                                 display_max=True, figsize=(12,9))

plt.xlabel(x_label_1, fontsize=16); plt.ylabel(y_label_1, fontsize=16)
plt.title(title_1, fontsize=16)
plt.show()





