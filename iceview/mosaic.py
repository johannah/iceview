import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from skimage.io import ImageCollection
from skimage.data import imread
from skimage.color import rgb2hsv
from skimage.transform import warp, SimilarityTransform, ProjectiveTransform
from skimage.feature import (ORB, match_descriptors, corner_harris,
                             plot_matches)
from skimage.measure import ransac

def compare(*images, **kwargs):
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
        ax[n].imshow(image, interpolation='nearest', cmap='gray')
        ax[n].set_title(label)
        ax[n].axis('off')
    plt.tight_layout()

def hsv_imread(img_path):
    return rgb2hsv(imread(img_path))

def load_images(search_dir, ftype):
    search_path = os.path.join(search_dir, '*'+ftype)
    imgs = ImageCollection(search_path, conserve_memory=False,
                           load_func=hsv_imread)
    return imgs

def detect_and_extract(img, n_keypoints):
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=.05)
    orb.detect_and_extract(img)
    keypoints = orb.keypoints
    descriptors = orb.descriptors
    return keypoints, descriptors

def plot_two_matches(img1, img2, k1, k2, matches):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    plt.gray()
    plot_matches(ax, img_col[0], img_col[1], k1, k2, matches)
    plt.show()

def get_best_matches(img1, img2, k1, k2, matches):
    print(matches.shape)
    src = k1[matches[:,0]][:,::-1]
    dst = k2[matches[:,1]][:,::-1]
    print(src.shape, dst.shape)

    # if there are not enough matches, this fails
    model_robust, inliers = ransac((src, dst), ProjectiveTransform,
                                   min_samples=4, residual_threshold=1,
                                   max_trials=300)

    ## shape of the registration target
    r, c = img1.shape[:2]
    # transforms are in (x, y) format instead of (row, col)
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])
    # warp image corners to new position in mosaic
    warped_corners = model_robust(corners)
    # find extents of reference image and warped image
    all_corners = np.vstack((warped_corners, corners))

    ## determine overall output shape
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)
    # ensure integer shape
    output_shape = np.ceil(output_shape[::-1]).astype(int)
    # in-plane offset transforamtion
    offset = SimilarityTransform(translation=-corner_min)
    # warp img1 to img2 (warp takes inverse mapping)
    transform = (model_robust + offset).inverse

    img1_warped = warp(img1, transform, order=3,
                       output_shape=output_shape, cval=-1)
    # mask == 1 inside the image
    img1_mask = img1_warped != -1
    # get background values
    img1_warped[~img1_mask] = 0

    img2_warped = warp(img2, transform, order=3,
                       output_shape=output_shape, cval=-1)
    # mask == 1 inside the image
    img2_mask = img2_warped != -1
    # get background values
    img2_warped[~img2_mask] = 0

    compare(img1_warped, img2_warped)
    plt.show()

img_col = load_images('../data/jpg/', 'jpg')[:10]
img_feat = {}
num_imgs = len(img_col)
for xx, img in enumerate(img_col):
    k, d = detect_and_extract(img[:,:,2], n_keypoints=200)
    img_feat[xx] = {'keypoints':k, 'descriptors':d}

for x1 in range(num_imgs):
    #compare this image with each image after it
    for x2 in range(x1+1, num_imgs):
        matches = match_descriptors(img_feat[x1]['descriptors'],
                                    img_feat[x2]['descriptors'],
                                    max_distance=.2)

        print(x1, x2, matches.shape)
    #get_best_matches(img1, img2, k1, k2, matches)
