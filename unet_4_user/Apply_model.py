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

from Utility import list_patches
import losses
#img_size = 256
#model = Unet_tt_v2(img_size)
#save_weight_path = '/home/thibault/Documents/Data/ADS/Training_v3_256/unet_v3_6.h5'
#model.load_weights(save_weight_path)

def prepare_(model_path = 'choose', test_image_path = 'choose'):
    '''
    Prepare the model and the image to apply the model to the image
    input :
        + model_path : location of the model or 'choose' defualt will open an askfile window
        + test_image_path : same
    return :
        + 'model' as keras model
        + 'img' as PIL image
        + Folder_path : location of the test image (to save the result in the same location)
    '''
    #from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    
    img = None
    model = None
    folder_path = None
    
    if model_path == 'choose':
        root = Tk()
        root.withdraw()
        model_path =  filedialog.askopenfilename(initialdir = './',\
                                              title = 'select a model .h5')
    
    if os.path.isfile(model_path):
        model = load_model(model_path, custom_objects={'softmax_dice_loss_2': losses.softmax_dice_loss_2})
    #model_path = '/home/thibault/Documents/Data/ADS/Model_Unet_tt/2019-02-04_1604.h5'
    #model = load_model(model_path)    
        
    
    if test_image_path == 'choose':
        root = Tk()
        root.withdraw()
        test_image_path =  filedialog.askopenfilename(initialdir = './',\
                                                      title = 'select image RGB to test .png or .jpg.')
        
    if not test_image_path == None and os.path.isfile(test_image_path):
        img = Image.open(test_image_path)  
    #path_to_test_image ='/home/thibault/Documents/Data/ADS/Training_256_v5/Gray/Raw_data/Test/sample3/' 
    # import the image as PIL image
        folder_path, image_name = os.path.split(test_image_path)
    
    #img = ImageOps.equalize(img)

    return img, model, folder_path


###################################################
# Unet_tt by patches

def Unet_by_patches(img, model, patch_size=256, overlap=64, RGB = True):
    
    '''
    routine to process large image patch by patch.
    input: img = image input format is Image PIL
            model = keras model loaded from keras.load_model function and model_xxx.h5
            patch_size = size of input image in pixel
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
        A = np.array(patch)
        
        # don't process patch with only background
        if sum(A.flatten()) > 10*len(A.flatten()):
            
            A = exposure.equalize_adapthist(A,clip_limit=0.01)
        
            A = A[np.newaxis,:,:,np.newaxis]
            pred_2 = model.predict(A)[0]
        
            h, w= map(ind_0,e)
            pred_image[e[0]+h:e[0] + patch_size , e[1]+w:e[1] + patch_size,:]=pred_2[h:, w:, :]
    
    return pred_image



def main():
    
    img, model, path_to_test_image = prepare_()
    
    pred_image = Unet_by_patches(img,model, patch_size = 256, overlap=64)
        
    mask = np.argmax(pred_image, axis=-1)
     
    return pred_image, mask

def save_prediction(pred_image,save_name = '/home/thibault/Documents/mask_pred.png'):    
    # save the axon mask as image
    import skimage.io as io
    
    #mask_ = 255*np.array(mask==2, dtype= np.uint8)
    io.imsave(save_name, pred_image)
