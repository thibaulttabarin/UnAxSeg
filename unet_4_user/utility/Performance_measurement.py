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
    
    features = ['diam_f', 'diam_a', 'myelin_thick', 'gratio']
    fiber_df = pd.DataFrame(columns=features)
    
    for i, fiber in enumerate(regions_fiber):
        
        axon = regions_axon[i]
        diameter_f = pixel_size*2*np.sqrt(fiber.area/math.pi) # diameter in pixel
        diameter_a = pixel_size*2*np.sqrt(axon.area/math.pi) # diameter in pixel
        myelin_thick = (diameter_f-diameter_a)/2
        gratio = diameter_a/diameter_f
        fiber_df.loc[i] = [diameter_f, diameter_a, myelin_thick, gratio]
        
    return fiber_df
