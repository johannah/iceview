from shutil import copyfile
import subprocess
from subprocess import Popen, PIPE
from multiprocessing import Pool, freeze_support, cpu_count
import matplotlib.pyplot as plt
import itertools
import os
from glob import glob
import numpy as np
import argparse
import sys

from itertools import islice
import cv2
from copy import copy, deepcopy
from scipy.ndimage import rotate

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import config
import math

from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import warp, SimilarityTransform, AffineTransform, ProjectiveTransform
from skimage import img_as_float, img_as_ubyte


def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches, title='',
                 keypoints_color='b', matches_color=None, only_matches=False):
    """Plot matched features.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    """

    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    image = np.concatenate([image1, image2], axis=1)

    offset = image1.shape

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0],
                   facecolors='none', edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset[1], keypoints2[:, 0],
                   facecolors='none', edgecolors=keypoints_color)

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, 2 * offset[1], offset[0], 0))
    ax.set_title(title)

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]

        if matches_color is None:
            color = np.random.rand(3, 1)
        else:
            color = matches_color

        ax.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
                (keypoints1[idx1, 0], keypoints2[idx2, 0]),
                '-', color=color)




def compare(*images, **kwargs):
    """
    Utility function to display images side by side.

    Parameters
    ----------
    image0, image1, image2, ... : ndarrray
        Images to display.
    labels : list
        Labels for the different images.
    """
    f, axes = plt.subplots(1, len(images), **kwargs)
    axes = np.array(axes, ndmin=1)

    labels = kwargs.pop('labels', None)
    if labels is None:
        labels = [''] * len(images)

    for n, (image, label) in enumerate(zip(images, labels)):
        axes[n].imshow(image, interpolation='nearest', cmap='gray')
        axes[n].set_title(label)
        axes[n].axis('off')

    plt.tight_layout()

def add_alpha(img, mask=None):
    """
    Adds a masked alpha channel to an image.

    Parameters
    ----------
    img : (M, N[, 3]) ndarray
        Image data, should be rank-2 or rank-3 with RGB channels. If img already has alpha,
        nothing will be done.
    mask : (M, N[, 3]) ndarray, optional
        Mask to be applied. If None, the alpha channel is added
        with full opacity assumed (1) at all locations.
    """
    # don't do anything if there is already an alpha channel
    #return img

    if img.shape[2] > 3:
        return img
    # make sure the image is 3 channels
    if img.ndim == 2:
        img = gray2rgb(img)
    if mask is None:
        # create transparent mask
        # 1 should be fully transparent
        mask = np.ones(img.shape[:2], np.uint8)*255
    return np.dstack((img, mask))

def find_corners(all_corners):


    # The overally output shape will be max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)

    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape[::-1]).astype(int)

    # This in-plane offset is the only necessary transformation for the base image
    offset = SimilarityTransform(translation= -corner_min)
    return offset, output_shape


def match_from_toa(fk, fd, tk, td, matcher):
    min_matches = 20
    # get matching keypoints between images (from to) or (previous, base) or (next, base)
    try:
        matches = matcher.knnMatch(fd, td, k=2)
        matches_subset = filter_matches(matches)

        src = [fk[match.queryIdx] for match in matches_subset]
        # target image is base image
        dst = [tk[match.trainIdx] for match in matches_subset]

        src = np.asarray(src)
        dst = np.asarray(dst)

        if src.shape[0] > min_matches:
            # TODO - select which transform to use based on sensor data?
            model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=8,
                                           residual_threshold=1)
            accuracy = float(inliers.shape[0])/float(src.shape[0])
            ransac_matches = matches_subset[inliers]
            return model_robust, ransac_matches, accuracy
    except Exception, e:
        logging.error(e)
    return None, None, 0



def getKeypointandDescriptors(img, detector_name, num_kpts=400):
    if detector_name in ['SIFT', 'SURF', 'ORB']:
        detector = eval("cv2.%s(%d)" %(detector_name,num_kpts))
        if detector_name == "SURF":
            detector.upright = False
            detector.nOctaves=6
            detector.nOctaveLayers=8
        kps, des = detector.detectAndCompute(img, None)
        kp = np.asarray([[k.pt[1], k.pt[0]] for k in kps])
        return kp, des

def loadImage(img_path, detector):
    rgb = add_alpha(cv2.imread(img_path))
    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Find key points in base image
    k, d = getKeypointandDescriptors(img, detector)
    return rgb, k, d


def make_chunks(it, size):
    return [it[x:x+size] for x in range(0, len(it), size)]

def make_list_of_matches(matches):
    return [m[0] for m in matches]

def filter_matches(matches, ratio = 0.75):
    filtered_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])
    return filtered_matches




def match_from_to(fk, fd, tk, td, min_matches=10):
    # get matching keypoints between images (from to) or (previous, base) or (next, base)
    try:
        # skimage way
        # may need to reverse
        matches = match_descriptors(fd, td, cross_check=True)
        src = tk[matches[:,1]][::-1]
        dst = fk[matches[:,0]][::-1]

        if src.shape[0] > min_matches:
            # TODO - select which transform to use based on sensor data?
            model_robust, inliers = ransac((src, dst),
                                           AffineTransform,
                                            min_samples=8,
                                           residual_threshold=1)
            accuracy = float(inliers.shape[0])/float(src.shape[0])
            ransac_matches = matches[inliers]
            return model_robust, ransac_matches, accuracy
        else:
            logging.info("Not enough matches: %d < min_matches: %d" %(src.shape[0], min_matches))
    except Exception, e:
        print("Exception", e)
        logging.error(e)
    return None, None, 0

def warp_img(img, transform, output_shape):
    try:
        warped = warp(img, transform, order=1, mode='constant',
                   output_shape=output_shape, clip=True, cval=0)
        return warped
    except Exception, e:
        logging.error("Error warping image %s img shape %s, output shape %s" %(e, img.shape, output_shape))
        return None

def copy_new_files(input_dir, output_dir, in_ftype, out_ftype, wsize, do_clear, limit):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    if do_clear:

        to_clear_mosaics = sorted(glob(os.path.join(output_dir, '*RUN*MATCH*%s'%out_ftype)))
        if len(to_clear_mosaics):
                logging.warning("Clearing RUN files from output_dir: %s" %output_dir)
                for f in to_clear_mosaics:
                    os.remove(f)

        #to_clear = sorted(glob(os.path.join(output_dir, '*%s'%out_ftype)))
        #if len(to_clear):
        #    logging.warning("Clearing files from output_dir: %s" %output_dir)
        #    for f in to_clear:
        #        os.remove(f)

    logging.info("Using convert to transfer and scan input images")
    in_files = sorted(glob(os.path.join(input_dir, '*%s'%in_ftype)))
    if limit is not None:
        try:
            in_files = in_files[:limit]
        except:
            pass

    for iimg in sorted(in_files):
        oname = os.path.basename(iimg).split('.')[0] + '.%s' %out_ftype

        ofile = os.path.join(output_dir, oname)
        if not os.path.exists(ofile):
            cmd = ["convert", iimg, "-resize", "%dx%d" %(wsize[0], wsize[1]), ofile]
            subprocess.call(cmd)
            logging.info("Calling %s" %' '.join(cmd))
        else:
            pass
            logging.debug("The file %s already exists" %ofile)
