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

def get_keypoint_filename(putdir, imgname):
    kpname = os.path.join(putdir, imgname)
    kpname = ''.join(kpname.split('.')[:-1])
    return kpname

def get_match_filename(putdir, base_name):
    bn = ''.join(base_name.split('.')[:-1])
    bn = os.path.join(putdir, bn + '.match')
    return bn

def create_match_file(putdir, base_name):
    match_name = get_match_filename(putdir, base_name)
    bfp = open(match_name, 'w')
    bfp.write(base_name + '\n')
    return bfp

def add_match(match_fp, putdir, img_name):
    match_name = get_match_filename(putdir, img_name)
    match_fp.write(img_name + '\n')
    # see if this file has previous matches
    if os.path.exists(match_name):
        img_matches = []
        imfp = open(match_name, 'r')
        for line in imfp.readlines():
            print(line)
            match_fp.write(line.strip())
        print("Matched %s match_name, removing" %match_name)
        os.remove(match_name)
    return match_fp

def save_keypoints_and_descriptors(kppath, img_k, img_d):
    kfp = open(kppath+'.kpt', 'w')
    dfp = open(kppath+'.des', 'w')
    np.savetxt(kppath+'.kpt', img_k, delimiter=',')
    np.savetxt(kppath+'.des', img_d, delimiter=',')
    kfp.close()

def load_keypoints_and_descriptors(kppath):
    kfp = open(kppath+'.kpt', 'r')
    dfp = open(kppath+'.des', 'r')
    img_k = np.loadtxt(kfp, delimiter=',')
    img_d = np.loadtxt(dfp, delimiter=',')
    kfp.close()
    dfp.close()
    return img_k, img_d

def get_keypoints_and_descriptors(ikdname, img_gray, extractor):
    try:
        img_k, img_d = load_keypoints_and_descriptors(ikdname)
        print("loaded image %s from file" %ikdname)
    except IOError:
        img_k, img_d = detect_and_extract(extractor, img_gray)
        save_keypoints_and_descriptors(ikdname, img_k, img_d)
        print("derived %s keypoints" %ikdname)
    return img_k, img_d


def get_basename(run_num, base_num, matches):
    txt = ['run_', '_base_', '_matches_']
    name = [run_num, base_num, matches]
    fill = []
    out = ''
    for xx, n in enumerate(name):
        out += txt[xx]
        if n == '*':
            out+= n
        else:
            out+= '%05d' %n
    out += '.%s' %output_image_type
    return out

def get_unmatched(run_num):
    bn = get_basename(run_num, '*', '*')
    bpath = os.path.join(outdir, bn)
    unmatched = glob(bpath)
    return unmatched

def get_matches(base_path):
    try:
        bn = os.path.split(base_path)[1]
        match = int(bn.split('_')[-1].split('.')[0])
    except:
        match = 0
    return match

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







