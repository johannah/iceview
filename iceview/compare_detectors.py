from shutil import copyfile
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import argparse
import sys

import cv2
from copy import copy, deepcopy
from scipy.ndimage import rotate

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import config
import math
import pickle
import time
from skimage.transform import rescale
from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
from skimage.transform import warp, warp_coords, SimilarityTransform, AffineTransform, ProjectiveTransform
from skimage import img_as_float, img_as_ubyte
from scipy.ndimage import rotate
import mosaic as m

detectors = {'ORB':4000, "BRISK":20, "SURF":4000, "SIFT":4000, "zernike":400, "FREAK":400}

#detectors = {'ORB':400}#, "BRISK":20, "SIFT":400, "FREAK":400}

def make_dict(transformations):
    ddict = {}
    for dname in detectors.keys():
        ddict[dname] = {}
        for r in transformations:
            ddict[dname][r] = {'num_inliers':[], 'precision':[],
                                'num_matches':[], 'recall':[], 'num_features':[],
                               'putative_match_ratio':[], 'time':[]}
    return ddict

def get_vals(dname, ddict, k, d, kr, dr, a, r):
    model_robust, ransac_matches, matches, inliers, precision = m.match_from_to(kr, dr, k, d, 8)
    b = time.time()
    ddict[dname][r]['precision'].append(precision)
    ddict[dname][r]['num_features'].append(kr.shape[0])
    print("NUMBER OF FEATURES", kr.shape[0])
    ddict[dname][r]['time'].append(b-a)
    ddict[dname][r]['num_matches'].append(matches.shape[0])
    ddict[dname][r]['num_inliers'].append(inliers.shape[0])
    putative_match_ratio = matches.shape[0]/float(kr.shape[0]+.0001)
    recall = inliers.shape[0]/float(kr.shape[0]+.0001)
    ddict[dname][r]['recall'].append(recall)
    ddict[dname][r]['putative_match_ratio'].append(putative_match_ratio)
    return ddict, ransac_matches


def compare_transformations(save_name, imgs, do_plot, change_type):
    if change_type == 'rotate':
       changes = [5, 15, 30, 45 ]
       title = 'Rotation of %s Degrees'
    elif change_type == 'scale':
       changes = [0.5, 0.75, 1.25, 2.0]
       title = 'Scale of %s'
    else:
        return
    ddict = make_dict(changes)
    for img_path in imgs:
        if not os.path.exists(img_path):
            logging.error("Image does not exist: %s" %img_path)
        img = m.add_alpha(cv2.imread(img_path))
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for dname in detectors.keys():
            num_kpts = detectors[dname]
            if do_plot:
                fig, ax = plt.subplots(nrows=len(changes), ncols=1, figsize=(3, 10))
                fig.suptitle("%s with %s" %(change_type, dname), fontsize=14)
            k, d = m.getKeypointandDescriptors(gimg, dname, num_kpts)
            for xx, r  in enumerate(changes):
                print(img.shape, img.dtype)
                if change_type == 'rotate':
                    rimg = rotate(img, r)
                elif change_type == 'scale':
                    #rimg = rescale(img, r)
                    w, h, c = img.shape
                    rimg = cv2.resize(img,(int(h*r), int(w*r)))

                print(rimg.shape, rimg.dtype)
                ii = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
                mask = img[:,:,3]
                #rows, cols = gimg.shape
                #M = cv2.getRotationMatrix2D((cols/2,rows/2),r,1)
                #ii = cv2.warpAffine(gimg,M,(cols,rows))
                a = time.time()
                kr, dr = m.getKeypointandDescriptors(ii, dname, num_kpts, rimg[:,:,3])
                ddict, ransac_matches = get_vals(dname, ddict, k, d, kr, dr, a, r)
                if do_plot:
                    m.plot_matches(ax[xx], img, rimg, k, kr ,
                                   ransac_matches[:,::-1],
                                   title %r)
                    ax[xx].axis('off')
        if do_plot:
            plt.show()
    fo = open(save_name, 'wb')
    pickle.dump(ddict, fo)


if True:
    #imgs = ['dsc02821s.jpg']
    imgs = ['dji_0029s.jpg']
    do_plot = True
    save_name = 'test_%s.pkl'
    detectors = {'ORB':4000, "BRISK":20, "SIFT":4000, "FREAK":4000}
if True:
    save_name = 'good_%s.pkl'
    imgs = glob('/Volumes/johannah_external 1/good_features_small/*.jpg')
    do_plot = False
    detectors = {'ORB':400, "BRISK":20, "SIFT":400, "FREAK":400}
    #detectors = {'ORB':400, "BRISK":20, "SIFT":400, "FREAK":400, 'SURF':400}
if True:
    imgs = glob('/Volumes/johannah_external 1/bad_features_small/*.jpg')
    save_name = 'bad_%s.pkl'
    do_plot = False
    detectors = {'ORB':400, "BRISK":20, "SIFT":400, "FREAK":400}

print("Found %s images" %len(imgs))
do_change = 'scale'
save_name = save_name %do_change
compare_transformations(save_name, imgs, do_plot, do_change)
print("WROTE", save_name)
