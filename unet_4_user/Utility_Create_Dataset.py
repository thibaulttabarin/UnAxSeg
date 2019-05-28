#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:31:34 2019

@author: thibault
pipeline for dataset creatation
"""

#  1- convert RGB to inverse

import os
import PIL
from PIL import Image
from PIL import ImageOps
import numpy as np
import random
from Utility import Import_image, list_patches

from tkinter import filedialog
from tkinter import Tk

from shutil import copyfile, copy2 , move, rmtree
from natsort import natsorted
import glob


def list_directory(path):
    
    list_dir=[]   
    for i, folder in enumerate(os.listdir(path)):
        
        folder_abspath = os.path.join(path, folder)
        list_dir.append(folder_abspath)
    
    return list_dir

def list_Img_Mask(path):
    '''
    Sort filename of masks and images by corresponding pair 
    return : list of tuple with image and mask absolute filename
    [(../image_0,../mask_0),(../image_1,../mask_1),....]
    '''
    
    list_image = []
    list_mask = []
    
    img_paths = natsorted(glob.glob( path + "/image*")) # sort the file_name in alphanumeric order
    mask_paths = natsorted(glob.glob( path + "/mask*"))
    list_image.extend(img_paths)
    list_mask.extend( mask_paths) 
     
    return list(zip(list_image,list_mask))

def move_img_to(path_list, dst):
    '''
    copy images and masks from a list of path path_list to a dsetination dst
    '''
    
    if os.path.isdir(dst): rmtree(dst)
    os.makedirs(dst)
    
    for i, (img_path, mask_path) in enumerate(path_list):
            
        new_img_name = 'image' + '_' + str(i) + '.png'
        move(img_path, os.path.join(dst,new_img_name) )
            
        new_mask_name = 'mask' + '_' + str(i) + '.png'
        move(mask_path, os.path.join(dst,new_mask_name) )

def gray_invert(img):
    
    img =img.convert(mode ='L')
    img = ImageOps.invert(img)
    
    return img

def get_resampling_coef(filename, orig_pix_size=0.11):
    
    # open pixel size file and read it
    file = open(filename, 'r')
    pixel_size = float(file.read())
    resample_coeff = float(pixel_size) / orig_pix_size # Used to set the resolution to the general_pixel_size
    
    return resample_coeff

def create_patches(src, dst, patch_size=256, overlap = 25, index_start =0):
    '''
    Create the patches from an image grayscale
    input:
        + src : source folder containing the image, mask and pixel size
        + dst : destination folder for the patches (images and mask)
        + patch_size
        + overlap : overlap between patches
    ouput:
        + last index
        + list of the coordinate of the left corner of the patch
    
    '''
    src_image = os.path.join(src, "image.png")
    src_mask = os.path.join(src, "mask.png")
    src_pix_size = os.path.join(src, "pixel_size_in_micrometer.txt")
     
    img, _ = Import_image(src_image)
    mask, _ = Import_image(src_mask)
    
    ## TODO : add a resampling method, I am not it is necessary
    # resample = get_resampling_coef(src_pix_size, orig_pix_size=0.11)
    # new_size = tuple([int(e*resample) for e in img.size])
    # img_resize = img.resize(new_size)
    # mask_resize = mask.resize(new_size, resample=PIL.Image.NEAREST)
    
    img =gray_invert(img)
    img_shape = img.size[::-1] # inverse image size because difference convention between numpy and PIL
    
    L_pos = list_patches(img_shape, overlap_value=overlap, scw=patch_size)
    for i, e in enumerate(L_pos):
        
        box_ = (e[1],e[0],e[1]+patch_size,e[0]+patch_size)
        patch_img = img.crop(box_)
        patch_mask = mask.crop(box_)
        
        dst_patch_img = os.path.join(dst,"image_{}".format(index_start + i))
        dst_patch_mask = os.path.join(dst,"mask_{}".format(index_start + i))
        patch_img.save (dst_patch_img, format='png')
        patch_mask.save (dst_patch_mask, format='png')
    
    last_index = index_start + i
    return last_index

def Create_patches_for_dataset(Parent_folder, patch_size=256, overlap = 25 ):
    
    '''
    Create the patches (image and mask) for each sample for Train and validation"
    Store the patches in dst_folder 'Patch_for_Training' automatic creation
    the structure should be :
            +sample_1 :
                + image.png
                + mask.png 
                + pixel_size_micrometer.txt
            +smaple_2:
                + image.png
                + mask.png
                ....
    input: 
        + dataset_folder containing sample_1, sample_2 .... folder
        
    output: dst_folder destination folder where we save all patches
    '''
    
    src_folder = os.path.join(Parent_folder, 'Train_Validation')
    dst_folder = os.path.join(Parent_folder, 'Patch_for_Training')
    

    if os.path.isdir(dst_folder): rmtree(dst_folder)
    os.makedirs(dst_folder)
    
    index = 0
    if os.path.isdir(src_folder):
        
        list_dir = list_directory(src_folder)
                
        for src in list_dir:
            
            last_index = create_patches(src, dst_folder, patch_size=patch_size, overlap = overlap, index_start =index)
            index = last_index+1
            
    return dst_folder
            
 #########################
# Create train, validation, test       
def split_train_validation(path, split=(0.8,0.2), seed = 10):
    '''
    Create a dataset using patches (image, mask) generated by Create_patches_for_dataset
    split the data into 2 folder train and validation using the percentatge in split parameter
    input : 
        path: folder containing the images and masks as patches, image_0, mask_0, image_1, mask_1......
        split : percentage for the split (train, val, test) =(0.8,0.2)
        seed: seed for the random shuffle of the images. to reproduce dataset
    
    usage example:
        
        path='/home/thibault/Documents/Data/ADS/Training_256_v6/Patched_data/Train'
        train_per = 0.8
        validation_per = 0.1
        test_per = 0.1
        split= (train_per,validation_per,test_per)

        dst_train = '/home/thibault/Documents/Data/ADS/Training_256_v6/dataset/Train'
        dst_val = '/home/thibault/Documents/Data/ADS/Training_256_v6/dataset/Validation'
        dst_test = '/home/thibault/Documents/Data/ADS/Training_256_v6/dataset/Test'
        Create_dataset(path, dst_train, dst_val, dst_test, split=split, seed = 3)
            
    '''   

    list_paths = list_Img_Mask(path)
    
    random.seed(seed)
    random.shuffle(list_paths)
    
    train_per, validation_per = split
    # split the list 
    last_train_index = int(len(list_paths)*train_per)+1
    
    list_train = list_paths[0:last_train_index+1]
    list_val = list_paths[last_train_index+1:]

    # Create train folder
    dst_train = os.path.join(path,'train')
    move_img_to(list_train, dst_train)
    # Create validation folder
    dst_validation = os.path.join(path,'validation')
    move_img_to(list_val, dst_validation)

    
def Create_training_dataset(patch_size=256, overlap = 25, split=(0.8,0.2), seed =None):
    '''
    Main function to create the dataset from the following folder structure
    
    /Dataset:
        + /test:
            +/sample1:
                + image
                + mask
                + pixel_size_in_micrometer
                
        + /Train_Validation:
            +/sample_1 :
                + image.png
                + mask.png 
                + pixel_size_micrometer.txt
            +/sample_2:
                + image.png
                + mask.png
    Return:
        + a folder /Dataset/Patch_for_Training:
            +/Train:
                +/image_0
                +/image_1
                +/....
                +/mask_0
                +/mask_1
            +/Validation:
                +/image_0
                +/image_1
                +/....
                +/mask_0
                +/mask_1
    '''
    
    
    root = Tk()
    root.withdraw()# we don't want a full GUI, so keep the root window from appearing
    Parent_folder =  filedialog.askdirectory(initialdir = './',\
                                                      title = 'choose directory containing "/Test and /Train_validation folder"') 
    
    dst_folder = Create_patches_for_dataset(Parent_folder, patch_size = patch_size, overlap = overlap)
    split_train_validation(dst_folder, split = split, seed = seed)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    