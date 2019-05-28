#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:24:43 2019

@author: thibault
Utility for for unet_tt
and patcha and big image
"""
import numpy as np
from PIL import Image
from PIL import ImageOps
import skimage
import cv2

import matplotlib.pyplot  as plt
from matplotlib.widgets import LassoSelector, PolygonSelector
from matplotlib.path import Path

import glob
from natsort import natsorted
from tkinter import filedialog
from tkinter import Tk

import sys 
import os


##TODO
# Refactore list of patch it is very bad
def list_patches(img_shape, overlap_value=25, scw=512):
    
    '''
    Create a list of patch according to patch size and the overlap
    '''
    crop_size = tuple(np.subtract(img_shape, (overlap_value*2,)*2))
    
    # Then we create patches using the prediction window
    spw = scw - 2*overlap_value  # size prediction windows

    qh, rh = divmod(crop_size[0], spw)
    qw, rw = divmod(crop_size[1], spw)

    # Creating positions of prediction windows
    L_h = [spw * e for e in range(qh)]
    L_w = [spw * e for e in range(qw)]

    # Then if there is a remainder we take the last positions (overlap on the last predictions)
    if rh != 0:
        L_h.append(crop_size[0] - spw)
    if rw != 0:
        L_w.append(crop_size[1] - spw)

    xx, yy = np.meshgrid(L_h, L_w)
    P = [np.ravel(xx), np.ravel(yy)]
    L_pos = [[P[0][i], P[1][i]] for i in range(len(P[0]))]
    
    return L_pos

#################################################################
# Import the image from standard image format and convert it to numpy.array
def Image_2_np(filename = None, mode ='L', title = 'select image to test .png.'):
    '''
    Open an image directly into an array. 
    input: filename = local of the image to open, default None then you will be ask to select the file
            mode = correspond to PIL mode default 'L' for grayscale without alpha channel
                to turn off put None
    Output : return img as np.array and the file location.
    
    '''
    # select the location
    if filename is None: 
        root = Tk()
        root.withdraw()# we don't want a full GUI, so keep the root window from appearing
        filename =  filedialog.askopenfilename(initialdir = './',\
                                                      title = title) 
    # Open and convert the image
    Image.MAX_IMAGE_PIXELS = None
    im =Image.open(filename)
    if not mode==None: im =im.convert(mode=mode)
    
    img=np.array(im)
    folder_path, image_name = os.path.split(filename)
    
    return img, folder_path

def Import_image(filename = None):
    '''
    Open an image directly into an Image PIL format:
        
        + input: filename = location of the image to open if nothing or not file open dialogue window 
        + Output: img as PIL.Image, file location.
    
    '''    
    if filename is None or not os.path.isfile(filename): 
        
        root = Tk()
        root.withdraw()# we don't want a full GUI, so keep the root window from appearing
        filename= filedialog.askopenfilename(initialdir = "./",filetypes = [("image files",(".jpg",".png"))],\
                                               title='Select image from histology  .jpg or .png')
    Image.MAX_IMAGE_PIXELS = None
    img =Image.open(filename)

    return img, filename


def get_list_img(ext = '.jpg'):
    '''
    get the list of the image to analyse
    input  : Choose the extension of the images to analyse
    output : return the list of the absolute path of image to analyse
    
    '''
    root = Tk()
    root.withdraw()# we don't want a full GUI, so keep the root window from appearing
    dir_name =  filedialog.askdirectory(initialdir = './',\
                                                      title = 'choose directory containing image to analyse') 

    img_list = natsorted(glob.glob( dir_name + "/*"+ext)) # sort the file_name with the extension "ext" in alphanumeric order 
    
    return img_list, dir_name

###################################################################
# Class to crop a big image
class Mask_Poly_FromImage(object):
    '''
    Class object used to create mask for big image
    When the object is created from an image, the image is resize (smaller size) to create and manipulate the mask faster
    the function  get_nask create a figure with cursor to draw a polygone region of interest
    the function cropping 
    
    '''

    def __init__(self, image):
        
        self.image = image
        self.resize()        
        x, y = np.meshgrid(np.arange(self.resized.shape[1], dtype=int),
                           np.arange(self.resized.shape[0], dtype=int))
        self.pix = np.vstack((x.flatten(), y.flatten())).T
        
    def resize(self):
        
        image= self.image
        if len (image.shape)>2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.bitwise_not(image) # inverse color
        image = np.ubyte(image)    
        self.resized = cv2.resize(image, None, fx = 0.1  , fy = 0.1)
    
    def get_mask(self):
        
        data = self.resized
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(data)
        ax2 = fig.add_subplot(122)
        ax2.imshow(np.zeros_like(data))
        plt.subplots_adjust()
        self.fig=fig
        
        self.poly = PolygonSelector(ax1, self.onselect)
                
    def onselect(self, verts):
        
        p = Path(verts)
        data = self.resized
        ind = p.contains_points(self.pix)
        mask = ind.reshape(data.shape[0:2])
        self.mask_downsize = mask
        self.mask = skimage.img_as_bool(cv2.resize(skimage.img_as_uint(mask), self.image.shape[1::-1]))
        
        selected = np.zeros_like(data)
        selected[mask] = data[mask]
        self.selected = selected
        ax2=self.fig.axes[1]
        ax2.imshow(selected)
        self.fig.canvas.draw_idle()
                    
    def cropping(self):
        
        # resize the mask and tile it to 3rd axis to apply to a 3 channels image
        mask_3ch = np.tile(self.mask[:,:,None], [1,1,3])
        self.mask_ch3 = mask_3ch
        
        crop = np.zeros_like(self.image)
        crop[mask] = self.image[mask]
        self.crop = crop




