from subprocess import Popen, PIPE
from multiprocessing import Pool, freeze_support, cpu_count
import itertools
import os
from skimage.data import imread
from skimage.feature import (ORB, match_descriptors, corner_harris,
                             plot_matches)

from iceview.features import detect_and_extract
from iceview.models import find_two_matches
from iceview.mask import find_mask
from iceview.utils import load_image_names
import numpy as np


def find_all_matches(unmatched,
                     matched=[],
                     fail_limit=3,
                     num_keypoints=800,
                     min_to_match=20,
                     do_plot=False):

    num_unmatched = len(unmatched)
    if num_unmatched == 0:
        print("NONE left unmatched")
        return matched
    if num_unmatched == 1:
        print("ONE left unmatched")
        matched.append(unmatched[0])
        return matched

    print("=====================================")
    base = unmatched[0]
    if 'img' in base.keys():
        base_img = base['img']
    else:
        base_img = imread(os.path.join(base['path']))

    orb = ORB(n_keypoints=num_keypoints, fast_threshold=0.05)

    base_unmatched = []


    base_name = base['name'].split('.')[0]
    base_matched = 0

    # go through each image that is not yet matched
    for xx, timg in enumerate(unmatched[1:]):
        # if the image has not yet been loaded, load it now
        if 'img' not in timg.keys():
            timg['img'] = imread(timg['path'])

        # for now, only use the 3rd channel
        img = timg['img'][:,:,2]


        base_k, base_d = detect_and_extract(orb, base_img[:,:,2])
        # if we haven't recorded the keypoints for this image, get them now
        if 'keypoints' in timg.keys():
            img_k = timg['keypoints']
            img_d = timg['descriptors']
        else:
            img_k, img_d = detect_and_extract(orb, img)
            timg['keypoints'] = img_k
            timg['descriptors'] = img_d


        matches = match_descriptors(base_d, img_d, cross_check=True)

        model_robust, ransac_matches = find_two_matches(base_img[:,:,2], img,
                                                            base_k, img_k,
                                                            base_d, img_d)
        if do_plot:
            print('matches', matches.shape[0], ransac_matches.shape[0])
            #plot_two_keypoints(ax, base_img, img, base_k, img_k)
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,12))
            plt.title('%s #### %s' %(base_name, timg['name']))
            plt.gray()

            ax[0].imshow(base_img)
            ax[0].axis('off')
            ax[0].scatter(base_k[:, 1], base_k[:, 0], facecolors='none', edgecolors='r')

            ax[1].imshow(img)
            ax[1].axis('off')
            ax[1].scatter(img_k[:, 1], img_k[:, 0], facecolors='none', edgecolors='r')
            plt.show()

        if ransac_matches.shape[0] < min_to_match:
            print("------------", matches.shape[0], "ransac", ransac_matches.shape[0])
            base_unmatched.append(timg)
            if len(base_unmatched) >= fail_limit:
                # add two since we've already added this timg
                base_unmatched.extend(unmatched[xx+2:])
                break
        else:
            #print('ransac matches', ransac_matches.shape)
            base_img = find_mask(base_img, timg['img'], model_robust)
            base_img = img
            base_matched += 1
            print("***********", base_matched)
            base_name+= '_' + timg['name'].split('.')[0]
            # AFTER an image has been matched, remove from memory



    # if we were able to match some images to this base_img that
    # were not matched in the last run, call again until
    # the number of unmatched images stops decreasing

    #print('num previous unmatched', num_unmatched)
    #print("could not match %s out of %s imgs" %(len(base_unmatched),
    #                                            len(unmatched)-1))
    # not_matched must be > 0
    # the new number of matches must be less than last time's not matched


    if 'base_matched' in base.keys():
        all_base_matched = base['base_matched'] + base_matched
    else:
        all_base_matched = 1 + base_matched
    rr = {'name':base_name, 'img':base_img, 'base_matched':base_matched}
    if (len(base_unmatched)) > 0:
        #if base_matched > 0 :
            #base_unmatched.insert(0, rr)
            #print("!!!!!!!!!!! 1 match %s, unmatch %s" %(len(matched), len(base_unmatched)))

            #return find_all_matches(base_unmatched, matched, num_keypoints, min_to_match)
        #else:
        print("DECLARING %s as matched, ending with %s matches" %(base_name, base_matched))
        matched.append(rr)
        #print("!!!!!!!!!!! 2 match %s, unmatch %s" %(len(matched), len(base_unmatched)))
        return find_all_matches(base_unmatched, matched, fail_limit, num_keypoints, min_to_match, do_plot)
    else:
        matched.append(rr)
        #print("!!!!!!!!!!! 3 match %s, unmatch %s" %(len(matched), len(base_unmatched)))
        return matched


def split_find_all_matches(i):
    """Convert list to arguments for find_all_matches to work with multiprocessing
    pool"""
    return find_all_matches(*i)

#all_matched = find_all_matches(unmatched, [], param[0], param[1], param[2], False)

#pool = Pool(processes=cpu_count())
#split_unmatched = [img_col[:6]]
#params = [[], 1, 500, 50]
#all_matched = pool.map(split_find_all_matches,
#                       itertools.izip(split_unmatched,
#                                      itertools.repeat(params)))





num_keypoints = 800
img_col = load_image_names('../data/jpg/', 'jpg')
img_feat = {}
num_imgs = len(img_col)
min_matches = 40


unmatched = img_col
params = [[1, 500, 50], [2, 800, 20], [1000, 1000, 10], [1000, 10000, 7]]
for param in params:
    matched = find_all_matches(unmatched, [], param[0], param[1], param[2], False)
    print('found %s matches with' %len(matched), param)
    print("HAVE %s MATCHES" %len(matched))
