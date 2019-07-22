#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:45:18 2019

@author: thibault
"""
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def Fiber_morpho(fiber_label, axon_label, pixel_size=0.11):
    
    regions_fiber = measure.regionprops(fiber_label)
    regions_axon = measure.regionprops(axon_label)
    
    features = ['diam_f', 'diam_a', 'myelin_thick', 'gratio', 'area_f', 'area_a']
    fiber_df = pd.DataFrame(columns=features)
    
    for i, fiber in enumerate(regions_fiber):
        
        axon = regions_axon[i]
        diameter_f = pixel_size*2*np.sqrt(fiber.area/math.pi) # diameter in pixel
        diameter_a = pixel_size*2*np.sqrt(axon.area/math.pi) # diameter in pixel
        myelin_thick = (diameter_f-diameter_a)/2
        gratio = diameter_a/diameter_f
        fiber_df.loc[i] = [diameter_f, diameter_a, myelin_thick, gratio, fiber.area, axon.area]
        
    return fiber_df


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
    
    centroids = np.array([list(x.centroid) for x in regions_fiber_pred])
    centroids = centroids.astype(int)
    areas = np.array([x.area for x in regions_fiber_pred])
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

def pred_fiber_df(label_fiber_pred, df_fiber):
    
    '''
    Create a dataframe containing feature propterties of the fiber detected.
    Analysis case when comparing predicted fiber and ground truth
    Refer to the function Fiber_analysis and module utility.Myelin_segmentation_utility
    input : 
        - label_fiber_pred: after quick_myelin and clean_up
        - df_fiber: result from Fiber_analysis
    '''
    
    regions_fiber_pred = measure.regionprops(label_fiber_pred)
    features = ['gt_label', 'true_positive', 'pred_label', 'gt_diameter','pred_diameter', 'dice', 'gt_area', 'pred_area']
    df_fiber_pred = pd.DataFrame(columns=features)#, dtype = 'float')

    for i, fiber in enumerate(regions_fiber_pred):
        
        # for the fiber
        label = fiber.label
        A = df_fiber.loc[df_fiber['pred_label']==label]
        if not A.empty: 
            df_fiber_pred.loc[i]=list(A.iloc[0])
        else:
            diameter_f = 2*np.sqrt(fiber.area/math.pi)
            df_fiber_pred.loc[i]=[0, 0, fiber.label, 0, diameter_f, 0, fiber.area, 0]

    return df_fiber_pred


def Create_df(df_fiber_pred, df_axon_pred, pixel_size=0.11):
    
    '''
    Create a dataframe for predicted fiber with specific features decribes in pred_fiber_df
    
    '''
    df = df_fiber_pred[['gt_label','pred_label', 'true_positive','gt_diameter', 'pred_diameter', 'dice']]
    df=df.rename(index=str, columns={'gt_diameter':'gt_diam_f','pred_diameter':'pred_diam_f', 'dice': 'dice_f'})
    
    df ['gt_diam_f'] = pixel_size*df ['gt_diam_f']
    df ['pred_diam_f'] = pixel_size*df ['pred_diam_f']
    
    df_a = df_axon_pred[['gt_label','pred_label', 'true_positive','gt_diameter', 'pred_diameter', 'dice']]
    df_a=df_a.rename(index=str, columns={'gt_label':'gt_label_a','pred_label':'pred_label_a',
                                'true_positive':'true_positive_a','gt_diameter':'gt_diam_a',
                                'pred_diameter':'pred_diam_a', 'dice': 'dice_a'})
    df_a ['gt_diam_a'] = pixel_size*df_a ['gt_diam_a']
    df_a ['pred_diam_a'] = pixel_size*df_a ['pred_diam_a']    
    
    df= pd.concat([df, df_a],axis=1,sort=False)
    df = df[['gt_label','pred_label', 'true_positive','gt_diam_f', 'pred_diam_f', 'dice_f','gt_diam_a', 'pred_diam_a', 'dice_a']]
    
    df['pred_gratio'] = df['pred_diam_a']/df['pred_diam_f']
    df['pred_myelin_thick'] = (df['pred_diam_f']-df['pred_diam_a'])/2
    
    df['gt_gratio'] = df['gt_diam_a']/df['gt_diam_f']
    df['gt_myelin_thick'] = (df['gt_diam_f']-df['gt_diam_a'])/2
    
    return df