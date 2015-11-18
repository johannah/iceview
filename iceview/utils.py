#!/usr/bin/env python
import numpy as np
from skimage.data import imread
from skimage.io import imsave
import argparse
import sys
import os
from glob import glob
def hsv_imread(img_path):
    return rgb2hsv(imread(img_path))

def gray_imread(img_path):
    return rgb2gray(imread(img_path))

def load_image_names(search_dir, ftype):
    search_path = os.path.join(search_dir, '*'+ftype)
    files = glob(search_path)
    imgs = []
    for ff in files:
        imgs.append({'path':ff, 'name':os.path.split(ff)[1]})
    return imgs

def develop_metadata_mosaic():
    # set max pitch and roll angles as qulity
    pass

def plot_compare(*images, **kwargs):
    """
    Function to display images side by side (from skimage example)

    Parameters
    ----------
    image0, image1, ....: ndarray
        Images to display
    labels: list
        Labels for the different images
    """
    f, ax = plt.subplots(1, len(images), **kwargs)
    ax = np.array(ax, ndmin=1)

    labels = kwargs.pop('labels', None)
    labels = [''] * len(images)
    for n, (image, label) in enumerate(zip(images, labels)):
        ax[n].imshow(image, interpolation='nearest', cmap=plt.gray())
        ax[n].set_title(label)
        ax[n].axis('off')
    plt.tight_layout()
    
def plot_two_matches(img1, img2, k1, k2, matches):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.gray()
    #plot_matches(ax, img1, img2, k1, k2, matches)
    ax[0].imshow(img1)
    ax[0].axis('off')
    ax[0].scatter(k1[:, 1], k1[:, 0],  facecolors='none', edgecolors='r')

    ax[1].imshow(img2)
    ax[1].axis('off')
    ax[1].scatter(k2[:, 1], k2[:, 0], facecolors='none', edgecolors='r')
    plt.show()
    
    plt.show()
    
def plot_two_keypoints(img1, img2, k1, k2, s1=1, s2=1):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.gray()

    ax[0].imshow(img1)
    ax[0].axis('off')
    ax[0].scatter(k1[:, 1], k1[:, 0], facecolors='none', edgecolors='r')

    ax[1].imshow(img2)
    ax[1].axis('off')
    ax[1].scatter(k2[:, 1], k2[:, 0], facecolors='none', edgecolors='r')
    plt.show()

def create_output_file(outpath, overwrite=False):
    """
    Create a .mos file to store image features
    :param outpath: .json  file to store image features in'
    """
    if not overwrite:
        fp = open(outpath, 'a+')
    else:
        if os.path.exists(outpath):
            print("Overwriting previous file: %s" %outpath)
        fp = open(outpath, 'w')
    fp.close()
   
    
def write_output_file(outpath, img_name, img_dict, clear_features=True):
    """
    Write image features to an open json file
    :param fp: json file to write to
    :param img_name: name of the image
    :param img_dict: dictionary comtaining image features, descriptors, etc
    :param descriptors: descriptors
    :param clear_features: erase features for this image if they already exist
    """
    fp = open(outpath)
    jdata = json.load(fp)
    
    # add a new img and its features to the dict
    if img_name not in jdata.keys():
        jdata[img_name] = img_dict
    else:
        # if we were instructed to overrite the features, don't bother to check if 
        # the image is there
        if clear_features:
            jdata[img_name] = img_dict
        else:
            # for each feature in the images dict
            for feat in img_dict.keys():
                if feat in jdata[img_name].keys():
                    if type(feat) == list:
                        jdata[img_name][feat].extend(img_dict[feat])
                    else:
                        jdata[img_name][feat] = img_dict[feat]
                else:
                    jdata[img_name][feat] = img_dict[feat]
                    
    json.dump(jdata, fp)
    fp.close()
    
def read_output_file(outpath):
    if os.path.exists(outpath):
        fp = open(outpath)
        jdata = json.load(fp)
        return jdata
    else:
        create_output_file(outpath)
        return {}
        

                
            
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creating mosaic")
    parser.add_argument('input_dir', default='raw', type=str, help="directory with images to be mosaiced")
    parser.add_argument('output_dir', default='mosaic', help="directory to store mosaic/s")

 
    # parse command line
    try:
        args = parser.parse_args()
    except :
        parser.print_help()
        sys.exit()
        
    if not os.path.exists(args.input_dir):
        print("Error: input directory, %s does not exist" %args.input_dir)
        sys.exit() 
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)