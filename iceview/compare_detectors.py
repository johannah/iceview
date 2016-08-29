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
from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
from skimage.transform import warp, SimilarityTransform, AffineTransform, ProjectiveTransform
from skimage import img_as_float, img_as_ubyte
from scipy.ndimage import rotate
import mosaic as m






def add_params(ddict):
    for i in ddict.keys():
        ddict[i]['num_keypoints'] = []
        ddict[i]['precision'] = []
        ddict[i]['num_matches'] = []
    return ddict

num_kpts = 400
sift = cv2.SIFT(num_kpts)
orb = cv2.ORB(num_kpts)
surf = cv2.SURF(num_kpts)

detectors = ['ORB', "SIFT", "SURF"]# 'freak':freak}
rotations = {20:{}, 40:{}, 80:{}, 160:{}}
ddict = add_params(rotations)
#imgs = ['dsc02821s.jpg']
imgs = ['dji_0029s.jpg']
for dname in detectors:
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(6, 10))
    fig.suptitle("Testing Rotation with %s" %dname, fontsize=14)
    for img_path in imgs:
        if not os.path.exists(img_path):
            logging.error("Image does not exist: %s" %img_path)
        img = m.add_alpha(cv2.imread(img_path))
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k, d = m.getKeypointandDescriptors(gimg, dname, num_kpts)

        for xx, r  in enumerate(sorted(rotations.keys())):
            ii = rotate(gimg, r)
            kr, dr = m.getKeypointandDescriptors(ii, dname, num_kpts)
            model_robust, ransac_matches, precision = m.match_from_to(kr, dr, k, d, 4)

            #precision = 10
            if precision > 0:
                ddict[r]['num_keypoints'].append(kr.shape[0])
                ddict[r]['num_matches'].append(ransac_matches.shape[0])
                m.plot_matches(ax[xx], gimg, ii, k, kr ,
                               ransac_matches[:,::-1],
                               'Rotation: %d Degrees' %r)
                ax[xx].axis('off')

            else:
                ddict[r]['num_keypoints'].append(0)
                ddict[r]['num_matches'].append(0)

        plt.show()
