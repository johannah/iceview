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
        ax[n].imshow(image, interpolation='nearest', cmap=plt.gray())
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

def get_best_matches(k1, k2, matches):
    src = k1[matches[:,0]][:,::-1]
    dst = k2[matches[:,1]][:,::-1]
    # if there are not enough matches, this fails
    model_robust, inliers = ransac((src, dst), ProjectiveTransform,
                                   min_samples=20, residual_threshold=1,
                                   max_trials=40)

    return model_robust, inliers

def develop_metadata_mosaic():
    # set max pitch and roll angles as qulity
    pass

def find_mask(img1, img2, model_robust):
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
    # warp img1 according to coordinate transform
    """
    order 0 nearest neighbor
    order 1 bi-linear
    order 2 bi-quadratic
    order 3 bi-cubic
    order 4 bi-quartic
    order 5 bi-quintic
    output_shape (rows, cols)
    mode('constant', 'edge', 'symmetric', 'reflect', 'wrap')
    cval is value outside image boundary
    """

    img1_warped = warp(img1, transform, order=3,
                       output_shape=output_shape, cval=-1)
    # mask == 1 inside the image
    img1_mask = img1_warped != -1
    # get background values
    img1_warped[~img1_mask] = 0

    img2_warped = warp(img2, offset.inverse, order=3,
                      output_shape=output_shape, cval=-1)
    # mask == 1 inside the image
    img2_mask = img2_warped != -1
    # get background values
    img2_warped[~img2_mask] = 0

    merged = (img1_warped + img2_warped)
    overlap = (img1_mask * 1.0 ) + img2_mask
    norm = merged/np.maximum(overlap, 1)
    #plt.imshow(norm, cmap=plt.gray())
    #compare(img1_warped, img2_warped)
    return norm

img_col = load_images('../data/jpg/', 'jpg')[:50]
img_feat = {}
num_imgs = len(img_col)
min_matches = 40
num_keypoints = 400

for x in range(num_imgs):
    k, d = detect_and_extract(img_col[x][:,:,2],
                                    n_keypoints=num_keypoints)
    img_feat[x] = {'keypoints':k, 'descriptors':d}

def create_base(x):
    base_img = img_col[x]
    # use this image as the base image from now on
    in_base = [x]
    return base_img, in_base

def get_matches(base_x, base_img, base_k, base_d):
    for x2 in range(base_x+1, num_imgs):
        print("trying to match", base_x, x2)
        print(base_d.shape, img_feat[x2]['descriptors'].shape)
        matches = match_descriptors(base_d,
                                    img_feat[x2]['descriptors'],
                                    max_distance=.2)
        print("%s %s FOUND THESE MATCHES %s" % (base_x, x2, matches.shape,))
        # if enough matches between the two images are found, perform ransac
        if matches.shape[0] > min_matches:
            model_robust, inliers = get_best_matches(base_k,
                             img_feat[x2]['keypoints'], matches)
            print("RANSAC INLIERS " , base_x, x2, np.count_nonzero(inliers))
            ax, fig = plt.subplots(1, 1, figsize=(14, 9))
            plt.title("%s %s" %(base_x, x2))
            plot_matches(ax, base_img, img_col[x2], base_k, img_feat[x2]['keypoints'],
                         matches[inliers])
            base_img = find_mask(base_img, img_col[x2], model_robust)
            base_d += img_feat[x2]['descriptors']
            base_k += img_feat[x2]['keypoints']
            in_base.append(x2)
        else:
            print('EXITING BASE found %s in_base\n\n' %len(in_base), in_base)
            #if len(in_base) > 1:
            #    plt.imshow(base_img)
            #    plt.show()
            return x2


# compare the base image with each image after it
base_x = 0
while base_x < num_imgs:
    base_img, in_base = create_base(base_x)
    base_k = img_feat[base_x]['keypoints']
    base_d = img_feat[base_x]['descriptors']
    base_x = get_matches(base_x, base_img, base_k, base_d)


#notes:
# perhaps do something special if there is not at least some # of
# matches between images that are adjacent in time/space

# look at the distribution of pixels to determine which method of keypoints
# should be used?

# determine neighbors according to the meta-data, match features only among
# neighbors

# start with smaller scale image, then work up if needed, how to tell which
# size is necessary??? - texture changes maybe

# should we stack onto each image as we go... or when a new line is started

# how to determine if matches are bad.... need to master ransac or force
# matching to do better

# structure from motion, CMVS flightriot.com

#overlapping area, then matched keypoints as ordering

# only select keypoints from the same elevation level

# different resolutions
# 204x153
# 408x306
# 816x612

# quality estimates
# correlation error

# comparison -
# autopano, pix4d (3D), ptoptimizer, ptmender
# look at spatial relation
