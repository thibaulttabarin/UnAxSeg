from PIL import Image
from PIL import ImageOps
import numpy as np
from AxSeg.Preprocessing.DataConversion import convert_grayscale_mask_to_binary_masks
Image.MAX_IMAGE_PIXELS = None

def read_image_to_gray(input_image_path):
    
    input_image = Image.open(input_image_path)
    
    input_image = input_image.convert(mode='L')
    input_image = ImageOps.invert(input_image)
    return np.expand_dims(np.array(input_image, dtype=np.uint8), axis=-1)

def read_image_to_rgb(input_image_path):
    
    input_image = Image.open(input_image_path)
    rgb_image = np.array(input_image, dtype=np.uint8)[:,:,0:3]
    return np.array(input_image, dtype=np.uint8)[:,:,0:3]

def read_input_image(input_image_path, output_mode='RGB'):
    
    input_image = Image.open(input_image_path)
    if input_image.mode == 'RGB':
        if output_mode=='RGB':            
            output = np.array(input_image, dtype=np.uint8)[:,:,0:3]
            
        else:
            input_image = input_image.convert(mode='L')
            input_image = ImageOps.invert(input_image)
            output = np.array(input_image, dtype=np.uint8)
    else:
            input_image = input_image.convert(mode='L')
            input_image = ImageOps.invert(input_image)
            output = np.array(input_image, dtype=np.uint8)
    return output

def read_mask_image(mask_image_path):
    
    mask_image = Image.open(mask_image_path)
    mask_image = mask_image.convert(mode='L')
    mask_image = np.array(mask_image, dtype=np.uint8)
    masks = convert_grayscale_mask_to_binary_masks(mask_image, [0, 65, 190, 256])
    return masks

def write_mask_1c(arr, path):
    '''
    Convert the binary mask (0,1 or bool) into grayscale (0:0, 1:255)
    input : 
        arr :  mask for one channel, np.array ,bool, float or
        path: location/name to save the image
    '''
    
    
    A = 255*arr.astype(np.uint8)
    A = Image.fromarray(A)
    A = A.convert(mode='L')
    path= path+ ".png"
    A.save(path)
    
def write_mask_3c(arr, path):
    '''
    Convert the mask 3 classes (0,1,2) into grayscale coded classes (0:0, 1:128, 2:255)
    input : 
        arr :  mask for 3 channel, np.array int
        path: location/name to save the image
    '''
    
    A = arr.astype(np.uint8)
    A [A == 0]= 0
    A [A == 1]= 128
    A [A == 2]= 255
    A = Image.fromarray(A)
    A = A.convert(mode='L')
    path= path+ ".png"
    A.save(path)
    
def read_mask_3c(mask_image_path):
    '''
    Read the mask grayscale 3 color and convert to np.array 3 classes (0,1,3)
    input : mask_image_path mask location
    '''
    
    mask_image = Image.open(mask_image_path)
    mask_image = mask_image.convert(mode='L')
    mask_image = np.array(mask_image, dtype=np.uint8)
    
    mask = np.zeros_like(mask_image, dtype=np.int)
    mask[mask_image == 0]=0
    mask[np.bitwise_and(mask_image > 120, mask_image<130)]=1
    mask[mask_image > 230]=2
    
    return mask
    
def read_mask_1c(mask_image_path):
    '''
    Read the binary mask and convert to np.array 2 classes (0,1)
    input : mask_image_path mask location
    '''
    mask_image = Image.open(mask_image_path)
    mask_image = mask_image.convert(mode='L')
    mask_image = np.array(mask_image, dtype=np.uint8)
    
    mask = np.zeros_like(mask_image, dtype=np.int)
    mask[mask_image > 230]=1
    
    return mask
    
