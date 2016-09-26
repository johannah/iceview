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
from copy import copy, deepcopy
from scipy.ndimage import rotate

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import config
import math

from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import match_descriptors, corner_peaks, corner_harris
from skimage.feature.util import _mask_border_keypoints, DescriptorExtractor
from skimage.measure import ransac
from skimage.transform import warp, SimilarityTransform, AffineTransform, ProjectiveTransform
from skimage import img_as_float, img_as_ubyte

from mahotas.features import zernike_moments

import cv2
# Parameters for nearest-neighbor matching
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
    trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

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
        if keypoints1.shape[0]:
            if keypoints2.shape[0]:
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

        if keypoints1.shape[0] and keypoints2.shape[0]:
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


class zernike(DescriptorExtractor):

    def __init__(self, descriptor_size=256, patch_size=49,
                  sigma=1, sample_seed=1, mask=None):
        self.descriptor_size = descriptor_size
        self.patch_size = patch_size
        self.sigma = sigma
        self.sample_seed = sample_seed

        self.descriptors = None
        self.mask = mask

    def extract(self, image, keypoints):
        # patch size to build descriptor from
        patch_size = self.patch_size
        desc_size = self.descriptor_size
        #random = np.random.RandomState()
        #random.seed(self.sample_seed)
        ## why 8?
        #samples = np.array((patch_size / 5.0) * random.randn(desc_size * 8)).astype(np.int32)
        #hps2 = - (patch_size-2) // 2
        #samples = samples[(samples < hps) & (samples > hps2)]
        #d2 = desc_size*2
        #pos0 = samples[:d2].reshape(desc_size, 2)
        #pos1 = samples[d2:d2*2].reshape(desc_size, 2)

        #pos0 = np.ascontiguousarray(pos0)
        #pos1 = np.ascontiguousarray(pos1)
        hps = patch_size // 2
        self.mask = _mask_border_keypoints(image.shape, keypoints, hps)

        self.keypoints =  np.array(keypoints[self.mask, :], dtype=np.intp,
                                               order='C', copy=False)

        self.descriptors = []
        for nn in range(self.keypoints.shape[0]):
            kx, ky = self.keypoints[nn]
            patch = image[kx-hps:kx+hps, ky-hps:ky+hps]
            self.descriptors.append(zernike_moments(patch, 8., 12))
        self.descriptors = np.array(self.descriptors)
        # set up descriptors of size
        #self.descriptors = np.zeros((self.keypoints.shape[0], desc_size),
        #                            dtype=bool, order='C')


        #print("Before loop", self.keypoints.shape, self.descriptors.shape)
        #_zern_loop(image, self.descriptors.view(np.uint8), self.mask_keypoints,
        #            pos1, pos2)
        #print("aftere loop", self.keypoints.shape, self.descriptors.shape)



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
    # an alpha channel stores the transparency
    # value for each pixel. Zero means the pixel is
    # transparent and does not contribute to the image
    #return img

    print("ADD ALPHA", img.shape)
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


def match_from_toa(fk, fd, tk, td, min_matches=10 ):
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


def match_from_to_cv(fk, fd, tk, td, min_matches):
    # get matching keypoints between images (from to) or (previous, base) or (next, base)
    try:
        print("FROM TO", fd.shape, td.shape)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #matches = matcher.knnMatch(fd, td, k=2)
        matches = matcher.match(fd, td)
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

def getKeypointandDescriptors(img, detector_name, num_kpts=400, mask=None):
    detector_name = detector_name.upper()
    kps = np.array((0,2))
    des = np.array((0,2))
    if detector_name in ['SIFT', "BRIEF", "BRISK", 'SURF', 'ORB']:
        # most work well with num_kpts at 400, but BRISK needs a lower num (20)
        call ="cv2.%s(%d)" %(detector_name,num_kpts)
        detector = eval(call)
        if detector_name == "SURF":
            detector.upright = 0
            detector.nOctaves = 6
            detector.nOctaveLayers = 8
        kps, des = detector.detectAndCompute(img, mask)
        #skimage way
        #kp = np.asarray([[k.pt[1], k.pt[0]] for k in kps])
        # opencv way
        kp = np.asarray([k.pt for k in kps])
        return kp, des
    elif detector_name == "FREAK":
        # use an orb oriented brief keypoint detector
        feat_detector = cv2.ORB(num_kpts)
        keypoints = feat_detector.detect(img, mask)

        freakExtractor = cv2.DescriptorExtractor_create('FREAK')
        kps,des= freakExtractor.compute(img, keypoints)
        #skimage way
        kp = np.asarray([[k.pt[1], k.pt[0]] for k in kps])
        # opencv way
        #kp = np.asarray([k.pt for k in kps])
    elif detector_name == "ZERNIKE":
        # find keypoint detector
        #kp = corner_peaks(corner_harris(img,
        #                  method='eps', eps=.001, sigma=3), min_distance=5)
        d = cv2.ORB(num_kpts)
        print("HERE 1")
        kps = d.detect(img, mask)
        kp = np.asarray([[k.pt[1], k.pt[0]] for k in kps])
        print("Found keypoints", len(kp))
        descriptor = zernike(mask=mask)
        descriptor.extract(img, kp)
        des = descriptor.descriptors
        kp = descriptor.keypoints
        #kp = kk[descriptor.mask]
    return kp, des


def loadImage(img_path, detector, addedge=False):
    i = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print("LOADING", img_path, i.shape)
    rgb = add_alpha(i)
    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Find key points in base image
    k, d = getKeypointandDescriptors(img, detector)
    if addedge:
        if "RUN" not in img_path:
            print("Adding edge", os.path.split(img_path)[1])
            # used for creating display
            e = 3
            wc = 2
            wv = 240
            r, c, cc = rgb.shape
            rgb[:e,:,wc] = wv
            rgb[r-e:,:,wc] = wv
            rgb[:,:e,wc] = wv
            rgb[:,c-e:,wc] = wv
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


FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

def match_from_to(fk, fd, tk, td, min_matches=10):
    # get matching keypoints between images (from to) or (previous, base) or (next, base)
    ransac_matches = np.zeros((0,2))
    matches = np.zeros((0,2))
    inliers = np.zeros((0,2))
    #try:
    if 1:
        # opencv way
        matches = matcher.knnMatch(fd, td, k=2)
        matches_subset = filter_matches(matches)
        matches_subset =  np.array([[match.trainIdx,match.queryIdx] for match in matches_subset])

        src = np.asarray(fk[matches_subset[:,1]])
        dst = np.asarray(tk[matches_subset[:,0]])
        logging.info("STARTING MATCH src: %d dst %d" %(src.shape[0], dst.shape[0]))
        if src.shape[0] > min_matches:
            # TODO - select which transform to use based on sensor data?
            try:
                model_robust, inliers = ransac((src, dst),
                                               AffineTransform,
                                               min_samples=min_matches,
                                               stop_sample_num=100,
                                               max_trials=2000,
                                               stop_probability=.995,
                                               residual_threshold=2)
            except Exception, e:
                logging.error(e)

            logging.info("FOUND inliers %d" %inliers.shape[0])
            if inliers.shape[0]:
                num_correct = inliers.shape[0]
                num_matches = src.shape[0]
                num_false = num_matches-num_correct
                ransac_matches = matches_subset[inliers]
                perc_correct = 1-float(num_false)/float(num_matches)
                return model_robust, ransac_matches, matches, inliers, perc_correct
        else:
            logging.info("Not enough matches: %d < min_matches: %d" %(src.shape[0], min_matches))
    #except Exception, e:
    #    print("EXCEPTION JO",e)
    #    logging.error(e)
    return np.zeros((3,3)), ransac_matches, matches, inliers, 0


def match_from_to_compare(fk, fd, tk, td, min_matches=10):
    # get matching keypoints between images (from to) or (previous, base) or (next, base)
    ransac_matches = np.zeros((0,2))
    matches = np.zeros((0,2))
    inliers = np.zeros((0,2))
    try:
        # skimage way
        # may need to reverse
        matches = match_descriptors(fd, td, cross_check=True)
        src = tk[matches[:,1]][::-1]
        dst = fk[matches[:,0]][::-1]
        logging.info("STARTING MATCH src: %d dst %d" %(src.shape[0], dst.shape[0]))
        if src.shape[0] > min_matches:
            # TODO - select which transform to use based on sensor data?
            try:
                model_robust, inliers = ransac((src, dst),
                                               AffineTransform,
                                               min_samples=min_matches,
                                               stop_sample_num=100,
                                               max_trials=2000,
                                               stop_probability=.995,
                                               residual_threshold=2)
            except Exception, e:
                logging.error(e)

            logging.info("FOUND inliers %d" %inliers.shape[0])
            if inliers.shape[0]:
                num_correct = inliers.shape[0]
                num_matches = src.shape[0]
                num_false = num_matches-num_correct
                ransac_matches = matches[inliers]
                perc_correct = 1-float(num_false)/float(num_matches)
                return model_robust, ransac_matches, matches, inliers, perc_correct
        else:
            logging.info("Not enough matches: %d < min_matches: %d" %(src.shape[0], min_matches))
    except Exception, e:
        print("EXCEPTION JO",e)
        logging.error(e)
    return np.zeros((3,3)), ransac_matches, matches, inliers, 0




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
