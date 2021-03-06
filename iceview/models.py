#!/usr/bin/env python
import numpy as np
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import warp, SimilarityTransform, AffineTransform, ProjectiveTransform


def get_best_matches(k1, k2, matches):
    src = k1[matches[:,0]][:,::-1]
    dst = k2[matches[:,1]][:,::-1]
    # if there are not enough matches, this fails
    model_robust, inliers = ransac((src, dst), AffineTransform,
                                   min_samples=20, residual_threshold=1,
                                   max_trials=40)

    return model_robust, inliers



def find_two_matches(base_img, img, base_k, img_k, base_d, img_d, min_matches=10):

    matches = match_descriptors(base_d, img_d, cross_check=True)
   
    #   * src (image to be registered):
    #   * dst (reference image):
   
    src = img_k[matches[:,1]][:,::-1]
    dst = base_k[matches[:,0]][:,::-1]
    
    if matches.shape[0] > min_matches:
#        model_robust, inliers = ransac((src, dst), ProjectiveTransform,
#                                   min_samples=10, residual_threshold=10,
#                                   stop_sample_num=100, max_trials=300)
#
        model_robust, inliers = ransac((src, dst), AffineTransform,
                                       min_samples=6, residual_threshold=3,
                                       max_trials=100)
        ransac_matches = matches[inliers]
        inlierRatio = ransac_matches.shape[0]/float(matches.shape[0])
        return model_robust, ransac_matches, inlierRatio
    else:
        return np.zeros((0, 2)), np.zeros((0, 2)), 0.0
