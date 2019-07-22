#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:57:24 2019

@author: thibault
"""

################
# Segmentation of the big image

# Create axon, myelin mask
import numpy as np
import matplotlib.pyplot as plt

# Watershed segmentation for myelin
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import square, remove_small_objects, skeletonize, \
 binary_erosion, label, watershed, skeletonize, remove_small_objects, remove_small_holes

from skimage.measure import find_contours, approximate_polygon, regionprops
from skimage import morphology
from  skimage.filters import rank
from skimage.transform import rescale
import matplotlib.patches as patches


def pad_for_seg (img, overlap_value=25, scw=512):
    
    '''
    Create a padding around the image to make it integer number of time 
    the patching with overlap
    '''
    crop_size = tuple(np.subtract(img.shape, (overlap_value*2,)*2))

    spw = scw - 2*overlap_value 
    
    qh = np.ceil(float(crop_size[0])/spw)*spw-crop_size[0]
    qw = np.ceil(float(crop_size[1])/spw)*spw-crop_size[1]
    img = np.pad(img, ((0, qh.astype('int')), (0, qw.astype('int'))), 'constant',constant_values=((0,0),(0,0)))
    return img


####################################################
def list_patches(img_shape, overlap_value=25, scw=512):
    
    '''
    Create a list of patch for a iamge size and a patching strategy (patch size + overlap)
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

#########################_, _, label_fiber_pred, label_axon_pred  = ms_uti.clean_up (fiber_pred_seg, mask)#############################
def quick_myelin_seg(prediction):

    '''
    Segmentation of the myelin using prediction map from ADS or other unet architecture
    input : prediction = mask in gray scale  0 for bkg, 127 for myelin, 255 for axon
    output : fiber_label = image with labels for each fiber
    '''
    axon = prediction > 200
    myelin = np.logical_and(prediction >= 50, prediction <= 200)
    
    # Step 1
    # Clean up : remove small objects and holes
    axon= remove_small_objects(axon, min_size=3)
    myelin = remove_small_holes(myelin, area_threshold=3)
    fiber = np.logical_xor(axon, myelin)
        
    ###############
    # Step 2 amd 3 : prepare the markers for watershed
    # Create the skeleton of the myelin
    skeleton = skeletonize(myelin)
    
    # fill the skeleton to increase the size of the markers
    fill_skeleton = ndi.binary_fill_holes(skeleton)
    # remove the skeleton from the filled skeleton  create distinguishable region
    seed_mask=np.logical_xor(fill_skeleton, skeleton)
    
    # only keep the region superimposed with axon
    markers = ndi.morphology.binary_propagation(axon, mask=seed_mask)
    
    ###############
    # Step 4
     # watershed segmentation
    distance = ndi.distance_transform_edt(fiber)
    markers_labels = ndi.label(markers)[0]
        
    # use compactness to remove inconstistant extensions
    # additionally, use watershed_line create a visual effect and contour finding 
    fiber_labels = watershed(-distance, markers_labels, mask=fiber, compactness=0.1, watershed_line=True)
        
    return fiber_labels  

############################################################
# Myelin segmentationby patch
def MyelinSeg_by_patch (image, L_pos, overlap_value, scw):
    
    '''
    Myelin segmentation by patch :
        Use quick_myelin_seg to segment patch by patch and reconstitute the image into recons_image
        input :
            image : prediction image
        
    '''
    
    ind_1= lambda x : 0 if x==0 else overlap_value # first patch rule
    max_ind = np.array(L_pos).max(axis =0)
    ind_last_0 = lambda x : overlap_value if x==max_ind[0] else 0 # last patch rule first index
    ind_last_1 = lambda x : overlap_value if x==max_ind[1] else 0 # last patch rule second index
    
   
    label = 0
    recons_img= np.zeros_like(image, dtype = np.int32)
    
    for i, e_1 in enumerate (L_pos):
        
        if i%10 ==0:  print ('......' +  ' Processed ' + str(i) +  ' pactched out of ' + str(len(L_pos)) )
                
        patch = image[e_1[0]:e_1[0] + scw, e_1[1]:e_1[1] + scw]
        
        # Check patch is no empty
        if np.sum(patch) != 0:    
            
            # Calculate segmentation of Myelin
            fiber_labels = quick_myelin_seg(patch)
            # Measure region properties and create centroid vector
            props_fiber = regionprops(fiber_labels)
            np_centroid= np.array([k.centroid for i, k in enumerate(props_fiber)])
            
            # Check np_centroid is no empty
            if np_centroid.size > 0:
                
                # Create the inside patch
                l,u =e_1
                h, w = map(ind_1,(l,u))
                h_l = ind_last_0 (l)
                w_l = ind_last_1 (u)
                ur  =  w
                lr  =  h
                w1 = scw - overlap_value - w + w_l
                h1 = scw - overlap_value - h + h_l
                
                selection_lr = np.logical_and(np_centroid[:,1]>ur, np_centroid[:,1]<ur+w1)
                selection_ud = np.logical_and(np_centroid[:,0]>lr, np_centroid[:,0]<lr+h1)
                selection = np.logical_and (selection_lr, selection_ud)
    
                # identify index corresponding to the selection
                id_ = np.where(selection ==True)[0]
                for j in id_:       
                    coords = props_fiber[j].coords
                    coords += np.array ([l,u])
                    # label =  props_myelin[i].label + last_label
                    label += 1
                    recons_img [coords[:,0] , coords[:,1]] = label

    return recons_img


#############################################################################################
#############################################################################################
def clean_up(recons_img, mask):
    
    '''
    select fiber that contain myelin + axon
    
    input : recons_img = image of labeled fiber
            mask = mask (bkg=0, myelin=127, axon=255)
    output : regionprops list for axons and fibers
    '''
    recons_img = remove_small_objects(recons_img, min_size=16)
    List_fiber_2 = regionprops (recons_img)
    
    # exclude fiber withou axon
    axon = mask > 200
    recons_img_f =  np.zeros_like(recons_img, dtype = np.int32)
    recons_img_a =  np.zeros_like(recons_img, dtype = np.int32)
    i=1
    for f in List_fiber_2:
        axon_mask_id = axon [f.coords[:,0] , f.coords[:,1]]
        if sum(axon_mask_id) > 4:
            recons_img_f [f.coords[:,0] , f.coords[:,1]] = i
            recons_img_a [f.coords[axon_mask_id,0] , f.coords[axon_mask_id,1]] = i
            i +=1
    
    List_fiber_2 = regionprops (recons_img_f)
    List_axon_2 = regionprops (recons_img_a)
    
    return List_fiber_2, List_axon_2, recons_img_f, recons_img_a


##########################################################
# random color coding with black or white background
def random_cmap(N, base_cmap='nipy_spectral'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    np.random.shuffle(color_list)
    color_list[0,:]=[0., 0., 0., 0.]
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

###########################################################
def display_region_colorcoded(image, ax =None):
    
    if ax ==None: fig,ax = plt.subplots(1)
    mycolor=random_cmap(image.max(),'cool')
    ax.imshow(image,cmap =mycolor)
    return ax

def Fiber_Myelin_Axon_colorcoded(fiber, axon, **kwargs):
    
    myelin = fiber-axon
    mycolor=random_cmap(fiber.max(),'cool')

    fig, ax =  plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, **kwargs)
    ax[0].imshow(fiber, cmap =mycolor)
    ax[1].imshow(myelin, cmap =mycolor)
    ax[2].imshow(axon, cmap =mycolor)
    
    return ax











