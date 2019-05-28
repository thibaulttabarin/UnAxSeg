#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:26:27 2019

@author: thibault
"""

import os
from PIL import Image
from PIL import ImageOps
import numpy as np
import skimage.io as io
from tqdm import tqdm
from skimage import exposure
import matplotlib.pyplot as plt

from tkinter import filedialog
from tkinter import Tk

from keras.models import load_model

from Utility import list_patches, Import_image
import losses
#img_size = 256
#model = Unet_tt_v2(img_size)
#save_weight_path = '/home/thibault/Documents/Data/ADS/Training_v3_256/unet_v3_6.h5'
#model.load_weights(save_weight_path)

def Import_model(model_path = ''):
    '''
    import and load the model return a ready to use model 
    input :
        + model_path : location of the model or anything else (default : '') will open an askfile window ''
    return :
        + 'model' as keras model
    '''
    
    model = None
    
    if model_path is None or not os.path.isfile(model_path):
        root = Tk()
        root.withdraw()
        model_path =  filedialog.askopenfilename(initialdir = './',filetypes = [("h5 model",".h5")],\
                                              title = 'select a model .h5')
    elif os.path.isfile(model_path):
        model = load_model(model_path, custom_objects={'softmax_dice_loss_2': losses.softmax_dice_loss_2}) 
        
    return model


###################################################
# Unet_tt by patches

def Unet_by_patches(img, model, patch_size=256, overlap=64, RGB = True):
    
    '''
    routine to process large image patch by patch.
    input: img = image input format is Image PIL
            model = keras model loaded from keras.load_model function and model_xxx.h5
            patch_size = size of input image in pixel (correspond to the model input/output)
            overlap = overlap between patches to reduce boundary effect during prediction
    '''
    if RGB:
        img =img.convert(mode ='L')
        img = ImageOps.invert(img)
    
    img_shape = img.size[::-1] # inverse image size because difference convention between numpy and PIL
    L_pos = list_patches(img_shape, overlap_value=overlap, scw=patch_size)
    pred_image=np.zeros(img_shape + (3,))
        
    ind_0= lambda x : 0 if x==0 else 64

    pbar = tqdm(L_pos)
    for i, e in enumerate (pbar):
        
        box_ = (e[1],e[0],e[1]+patch_size,e[0]+patch_size)
        patch = img.crop(box_)
        # convert PIL image to np.array
        A = np.array(patch)
        
        # don't process patch with only background
        if sum(A.flatten()) > 10*len(A.flatten()):
            
            A = exposure.equalize_adapthist(A,clip_limit=0.01)        
            A = A[np.newaxis,:,:,np.newaxis]
            pred_2 = model.predict(A)[0]
        
            # Reconstruct the image on the fly
            h, w= map(ind_0,e)
            pred_image[e[0]+h:e[0] + patch_size , e[1]+w:e[1] + patch_size,:]=pred_2[h:, w:, :]
    
    return pred_image


def main():
    
    model = Import_model()
    img, filename = Import_image()
    
    pred_image = Unet_by_patches(img, model, patch_size = 256, overlap=64)
        
    mask = np.argmax(pred_image, axis=-1)
    
    return pred_image, mask

def save_prediction(pred_image, save_name = 'pred.png'):    

    io.imsave(save_name, pred_image)

def save_mask(mask, save_name = 'mask.png'):    
    # save the mask as an image
    mask=127*mask.astype(np.uint8)
    io.imsave(save_name, mask)