#!/usr/bin/env python

import numpy as np
from skimage.data import imread
from skimage.io import imsave
import argparse
import sys
import os
def patchmaker(img, imsize=(255,255), percent_overlap=10):
    """
    Split an image into overlapping patches
       
    Parameters
    ----------
    img : ndarray
        Image from which to extract patches
    imsize : tuple of ints
        size of patches
    percent_overlap : int
        Percent as int of overlap desired between overlapping images
        
    Returns
    -------
    patches : list of imsize overlapping segments of the image
        
    """
    # store the patches here
    patches = []
    patch_rows = imsize[0]
    patch_cols = imsize[1]
    if 0 < percent_overlap < 100:
        # determine how many pixels to overlap
        non_overlap_rows = int(patch_rows*.01*(100-percent_overlap))
        non_overlap_cols = int(patch_cols*.01*(100-percent_overlap))
    else:
        non_overlap_rows = patch_rows
        non_overlap_cols = patch_cols
        
    # row indexes into the original image 
    r1, c1 = 0,0
    # column indexes into the original image
    r2, c2 = imsize
    # while the last index of the patch image is less than the size of the original, keep going
    while r2 < img.shape[0]:
        c1 = 0
        c2 = c1 + patch_cols
        while c2 < img.shape[1]:
            patch = img[r1:r2, c1:c2]
            patches.append(patch)
            c1 += non_overlap_rows
            c2 = c1 + patch_cols 
        r1 += non_overlap_rows
        r2 = r1 + patch_rows
    return patches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Options for patchmaker")
    parser.add_argument('input_img', type=str, help="image to be split")
    parser.add_argument('-o', default='patches', help="directory to store patches", dest='out_dir')
    parser.add_argument('-s', '--size', dest='patch_size', default=(100,100), type=tuple, help='tuple in the form of (rows, columns) to designate size of the patches')
    parser.add_argument('-p', type=int, default=10, help='percent overlap between patches as int', dest='perc_overlap')
    parser.add_argument('-f', type=str, default=None, choices=['jpg', 'png'], help='filetype to save patches as', dest='ftype')
 
    # parse command line
    try:
        args = parser.parse_args()
    except :
        parser.print_help()
        sys.exit()
     
    # load input image if possible
    try:
        img = imread(args.input_img)
    except Exception, e:
        print(e)
        sys.exit()
        
    # get patches
    patches = patchmaker(img, args.patch_size, args.perc_overlap)
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    input_base = os.path.join(args.out_dir, args.input_img.split('.')[0])
    if args.ftype is None:
        args.ftype = args.input_img.split('.')[1]
    for xx, patch in enumerate(patches):
        # write patch to file
        patch_name = input_base + '_%03d.'%xx + args.ftype
        imsave(patch_name, patch)
        
    
        
    
