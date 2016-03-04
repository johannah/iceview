#!/usr/bin/env python

from subprocess import Popen, PIPE
from multiprocessing import Pool, freeze_support, cpu_count
import matplotlib.pyplot as plt
import itertools
import os
from glob import glob
import numpy as np
import argparse
import sys
import logging

from skimage.data import imread
from skimage.feature import (ORB, match_descriptors, corner_harris,
                             plot_matches)

from iceview.features import detect_and_extract
from iceview.models import find_two_matches
from iceview.mask import find_mask
from iceview.utils import load_image_names
outdir='mosaics'
output_image_type='jpg'

def get_keypoint_filename(imgname, outdir):
    kpname = os.path.join(outdir, imgname)
    kpname = ''.join(kpname.split('.')[:-1]) + '.kpt'
    return kpname

def save_keypoints(kppath, img_k, img_d):
    kfp = open(kppath, 'w')
    np.save(kfp, img_k)
    np.save(kfp, img_d)
    kfp.close()

def load_keypoints(kppath):
    kfp = open(kppath, 'r')
    img_k = np.load(kfp)
    img_d = np.load(kfp)
    return img_k, img_d

def find_all_matches(
                     unmatched,
                     matched=[],
                     fail_limit=3,
                     num_keypoints=800,
                     min_to_match=10,
                     channel=2):

    num_unmatched = len(unmatched)
    if num_unmatched == 0:
        return matched
#    if num_unmatched == 1:
#        matched.append(unmatched[0])
#        return matched

    print("=====================================")
    base = unmatched[0]
    if 'img' in base.keys():
        base_img = base['img']
        #base_img = imread(base_path)
    else:
        base_img = imread(os.path.join(base['path']))
    base_path = base['path']
    orb = ORB(n_keypoints=num_keypoints, fast_threshold=0.05)

    base_unmatched = []
    # get the name without the filetype
    base_name = base['name'].split('.')[0]
    base_matched = 0

    base_matches = {'imgs':[], 'models':[]}
    # go through each image that is not yet matched
    for xx, timg in enumerate(unmatched[1:]):
        # if the image has not yet been loaded, load it now
        if 'img' not in timg.keys():
            timg['img'] = imread(timg['path'])
            kpname = get_keypoint_filename(timg['name'], outdir)
            if os.path.exists(kpname):
                img_k, img_d = load_keypoints(kpname)
                timg['keypoints'] = img_k
                timg['descriptors'] = img_d
        img = timg['img']
        ## if we haven't recorded the keypoints for this image, get them now
        if 'keypoints' in timg.keys():
            img_k = timg['keypoints']
            img_d = timg['descriptors']
        else:
            img_k, img_d = detect_and_extract(orb, img[:,:,channel])
            kpname = get_keypoint_filename(timg['name'], outdir)
            save_keypoints(kpname, img_k, img_d)
            timg['keypoints'] = img_k
            timg['descriptors'] = img_d

        # get new keypoints for the updated base
        base_k, base_d = detect_and_extract(orb, base_img[:,:,channel])

        matches = match_descriptors(base_d, img_d, cross_check=True)

        model_robust, ransac_matches = find_two_matches(base_img[:,:,channel],
                                                        img[:,:,channel],
                                                        base_k, img_k,
                                                        base_d, img_d)
        if ransac_matches.shape[0] < min_to_match:
            print("------------ could not match with only %s ransac_matches out of %s"
                  %(ransac_matches.shape[0], matches.shape[0]))
                  #ransac_matches.shape[0])
            fig, ax = plt.subplots(nrows=1, ncols=1)
            plot_matches(ax, base_img, img, base_k, img_k, ransac_matches)
            plt.show()
            base_unmatched.append(timg)
            if len(base_unmatched) >= fail_limit:
                # add two since we've already added this timg
                base_unmatched.extend(unmatched[xx+2:])
                break
        else:
            base_matched += 1
            print("*********** matched with %s" %ransac_matches.shape[0])
            base_name+= '_' + timg['name'].split('.')[0]
            # AFTER an image has been matched, remove from memory
            base_matches['imgs'].append(timg['img'])
            base_matches['models'].append(model_robust)
            print("Finding mask")
            base_img = find_mask(base_img, timg['img'], model_robust, channel)
            print("matched %s with base_img. new shape:" %timg['name'], base_img.shape)
            base_path = os.path.join(outdir, base_name+'.'+output_image_type)

    plt.imsave(base_path, base_img)
#
#    # if we were able to match some images to this base_img that
#    # were not matched in the last run, call again until
#    # the number of unmatched images stops decreasing
#
#    print('num previous unmatched', num_unmatched)
#    print("could not match %s out of %s imgs" %(len(base_unmatched),
#                                                len(unmatched)-1))
#    # not_matched must be > 0
#    # the new number of matches must be less than last time's not matched
#
#
#    if 'base_matched' in base.keys():
#        all_base_matched = base['base_matched'] + base_matched
#    else:
#        all_base_matched = 1 + base_matched
#    rr = {'name':base_name, 'img':base_img, 'base_matched':base_matched}
#    if (len(base_unmatched)) > 0:
#        #if base_matched > 0 :
#            #base_unmatched.insert(0, rr)
#            #print("!!!!!!!!!!! 1 match %s, unmatch %s" %(len(matched), len(base_unmatched)))
#
#            #return find_all_matches(base_unmatched, matched, num_keypoints,
#            #                        min_to_match, channel)
#        #else:
#        print("DECLARING %s as matched, ending with %s matches" %(base_name, base_matched))
#        matched.append(rr)
#        #print("!!!!!!!!!!! 2 match %s, unmatch %s" %(len(matched), len(base_unmatched)))
#        return find_all_matches(base_unmatched, matched, fail_limit,
#                                num_keypoints, min_to_match,channel)
#    else:
#        matched.append(rr)
#        #print("!!!!!!!!!!! 3 match %s, unmatch %s" %(len(matched), len(base_unmatched)))
#        return matched


def split_find_all_matches(i):
    """Convert list to arguments for find_all_matches to work with multiprocessing
    pool"""
    return find_all_matches(*i)

def multicore_stuff():
    # TODO
    # need to address how images are stored to be blended before this can work
    pool = Pool(processes=cpu_count())
    split_unmatched = [img_col[:6]]
    params = [[], 1, 500, 50]
    all_matched = pool.map(split_find_all_matches,
                           itertools.izip(split_unmatched,
                                          itertools.repeat(params)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creating mosaic")
    parser.add_argument('input_dir', help="directory with images to be mosaiced")
    parser.add_argument('output_dir', default=outdir, help="directory to store mosaic/s")
    parser.add_argument('-i', '--input-image-type', dest='input_image_type',
                        default='jpg', choices=['jpg', 'png', 'tif'],
                        help='Type of input image to be considered')
    parser.add_argument('-o', '--output-image-type', dest='output_image_type',
                        default='jpg', choices=['jpg', 'png', 'tif'],
                        help='Type of image to output mosaic as')
    # TODO: check to see if image is in output directory before adding it to
    # unmatched images
    parser.add_argument('-c', dest='do_clear', action="store_true", default=False,
                       help='remove any existing images from the output directory')
    # parse command line
    try:
        args = parser.parse_args()
    except :
        parser.print_help()
        sys.exit()

    output_image_type = args.output_image_type
    outdir = args.output_dir

    if not os.path.exists(args.input_dir):
        print("Error: input directory, %s does not exist" %args.input_dir)
        sys.exit()

    if not os.path.exists(args.output_dir):
        logging.info("Creating output directory: %s" %args.output_dir)
        os.mkdir(args.output_dir)
    else:
        if args.do_clear:
            # if clear flag is set, remove all existing image files from dir
            print("Removing all %s files from %s" %(output_image_type,
                                                    outdir))
            search = os.path.join(args.output_dir, '*' + output_image_type)
            for f in glob(search):
                os.remove(f)

    unmatched = load_image_names(args.input_dir, args.input_image_type)
    channel = 2
    matched = find_all_matches(unmatched, [], 1, 800, 20, channel)


