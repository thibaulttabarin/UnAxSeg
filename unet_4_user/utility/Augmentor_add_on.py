

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:20:59 2019

@author: thibault
Utilities for Augmentor
add functionality to Augmentor necessary for 3 labels mask

"""

# Create tst image
import numpy as np
import tempfile

def Create_a_Checker():
    
    check =50
    A = np.ones((check, check))
    B = np.zeros((check, check))
    block = np.block([[A,B],[B,A]])
    
    Nb_rep = 5
    image = np.matlib.repmat(block, Nb_rep, Nb_rep)
    
    
    image = np.pad (image, (check*2, check*2), 'constant', constant_values=(0,0))
    
    return image

def create_check_temp_image( file_format='JPEG'):
    tmpdir = tempfile.mkdtemp()
    tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.JPEG')

    image  = Create_a_Checker()
    im = Image.fromarray(np.uint8(image * 255))
    im.save(tmp.name, file_format)

    return tmp, tmpdir

#tmp, tmpdir =create_check_temp_image()
#r_d = Operations.Distort(probability=1, grid_width=8, grid_height=8, magnitude=10)
#tmp_im = []
#tmp_im.append(Image.open(tmp))
#tmp_im = r_d.perform_operation(tmp_im)
#
#I = tmp_im[0]
#I.show()

from Augmentor.Operations import Operation
from PIL import Image

class Mask_3ch(Operation):
    """
    Author: Thibault Tabarin Jan. 2019
    This class is used to re-threshold the mask and convert into 3 binary channels after transformations
    that usually introduce an interpolation (not binary result). This version only works with Datapipeline
    """
    def __init__(self, probability = 1):
        """
        required :attr:`probability` parameter
        :func:`~Augmentor.Pipeline.Pipeline.random_color` function.
        :param probability: probability is force to 1, when operation is invoked in the pipeline,
        it is always performed.
        """
        Operation.__init__(self, probability)
 

    def perform_operation(self, images):
        """
        Threshold the mask which is supposed to contain in images[1].
        :param images: The image to convert into 3 channels.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        def do(image):
            
                im_0 = image.point(lambda p: p >=0 and p < 65 and 255)
                im_1 = image.point(lambda p: p >=65 and p < 190 and 255)
                im_2 = image.point(lambda p: p >=190 and 255)
                im = Image.merge("RGB", (im_0, im_1, im_2))
                            
                return im

        augmented_images = []
        augmented_images.append(images[0])
        augmented_images.append(do(images[1]))
        augmented_images.append(images[1])

        return augmented_images

class Mask_3ch(Operation):
    """
    Author: Thibault Tabarin Jan. 2019
    This class is used to re-threshold the mask and convert into 3 binary channels (each channel binary) after transformations
    that usually introduce an interpolation (not binary result). This version only works with Datapipeline.
    
    """
    def __init__(self, probability = 1):
        """
        required :attr:`probability` parameter
        :func:`~Augmentor.Pipeline.Pipeline.random_color` function.
        :param probability: probability is force to 1, when operation is invoked in the pipeline,
        it is always performed.
        """
        Operation.__init__(self, probability)
 

    def perform_operation(self, images):
        """
        Threshold the mask which is supposed to contain in images[1].
        :param images: The image to convert into 3 channels.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        def do(image):
            
                im_0 = image.point(lambda p: p >=0 and p < 65 and 1)
                im_1 = image.point(lambda p: p >=65 and p < 190 and 1)
                im_2 = image.point(lambda p: p >=190 and 1)
                im = Image.merge("RGB", (im_0, im_1, im_2))
                            
                return im

        augmented_images = []
        augmented_images.append(images[0]) # transformed image
        augmented_images.append(do(images[1])) # binary transformed mask 
        augmented_images.append(images[1]) # interpolated transformed mask

        return augmented_images
    
class Mask_2ch(Operation):
    """
    Author: Thibault Tabarin Jan. 2019
    This class is used to rethreshold the mask and convert into 3 binary channels after transformations
    that usually introduce an interpolation (not binary result). This version only works with Datapipeline
    """
    def __init__(self, probability = 1):
        """
        required :attr:`probability` parameter
        :func:`~Augmentor.Pipeline.Pipeline.random_color` function.
        :param probability: probability is force to 1, when operation is invoked in the pipeline,
        it is always performed.
        """
        Operation.__init__(self, probability)
 

    def perform_operation(self, images):
        """
        Threshold the mask which is supposed to contain in images[1].
        :param images: The image to convert into 3 channels.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        def do(image):
            
                im = image.point(lambda p: p >=128 and 255)
                            
                return im

        augmented_images = []
        augmented_images.append(images[0])
        augmented_images.append(do(images[1]))

        return augmented_images
    
    
class Mask_1ch(Operation):
    """
    Author: Thibault Tabarin March. 2019
    This class is used to rethreshold the mask and convert into 1 binary channel after transformations taking only
    the middle channel (correponding to the myelin channel at the concept time)
    that usually introduce an interpolation (not binary result). This version only works with Datapipeline
    """
    def __init__(self, probability = 1):
        """
        required :attr:`probability` parameter
        :func:`~Augmentor.Pipeline.Pipeline.random_color` function.
        :param probability: probability is force to 1, when operation is invoked in the pipeline,
        it is always performed.
        """
        Operation.__init__(self, probability)
 

    def perform_operation(self, images):
        """
        Threshold the mask which is supposed to contain in images[1].
        :param images: The image to convert into 3 channels.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        def do(image):
            
                im = image.point(lambda p: p >=65 and p < 190 and 1)
                            
                return im

        augmented_images = []
        augmented_images.append(images[0])
        augmented_images.append(do(images[1]))

        return augmented_images