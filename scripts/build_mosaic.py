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

from copy import copy
from skimage.data import imread
from skimage.feature import (ORB, match_descriptors, corner_harris,
                             plot_matches)

from iceview.features import detect_and_extract
from iceview.models import find_two_matches
from iceview.mask import find_mask
from iceview.utils import load_image_names
absolute_min_matches = 10

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
        print("loaded %s from file" %ikdname)
    except IOError, e:
        print(e)
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

def find_all_matches(unmatched, run_num, min_to_match=40):
    channel = 2
    num_keypoints=1000
    # get previous base
    print("=====================================")
    ## setup detector
    orb = ORB(n_keypoints=num_keypoints, downscale=1.2, n_scales=20, harris_k=.04,
              fast_threshold=0.05)
    base_num = 0
    unmatched.sort()
    init_num_unmatched = len(unmatched)
    while len(unmatched):
        print("MY BASE NUM", run_num)
        # get next unmatched image
        base_path = unmatched.pop(0)
        # read the image
        base_img = imread(base_path)
        # if the image has gps data, remove it for now
        if base_img.shape[0] == 2:
            base_img = base_img[0]
        base_gray = base_img[:,:,channel]

        ikdname = os.path.join(base_path.replace('.'+output_image_type, ''))
        #base_k, base_d = detect_and_extract(orb, base_gray)
        base_k, base_d = get_keypoints_and_descriptors(ikdname, base_gray, orb)

        match_num = get_matches(base_path)
        base_name = get_basename(run_num, base_num, match_num)
        bad_match = 0
        matched_files = [base_path]

        print("New base file is: %s" %os.path.split(base_path)[1])
        #print('new unmatched', unmatched)
        for xx, img_path in enumerate(unmatched):
            # get new keypoints for the updated base
            img = imread(img_path)
            if img.shape[0] == 2:
                img = img[0]
            img_name = os.path.split(img_path)[1]
            print("working on img: %s" %img_name)
            img_gray = img[:,:,channel]
            ikdname = os.path.join(img_path.replace('.'+output_image_type, ''))
            img_k, img_d = get_keypoints_and_descriptors(ikdname, img_gray, orb)

            matches = match_descriptors(base_d, img_d, cross_check=True)

            model_robust, ransac_matches = find_two_matches(base_gray,
                                                            img_gray,
                                                            base_k, img_k,
                                                            base_d, img_d)
            num_ransac_matches = ransac_matches.shape[0]
            if num_ransac_matches < min_to_match:
                print("------------ could not match %s, only %s ransac_matches out of %s"
                      %(img_name, num_ransac_matches, matches.shape[0]))
                # couldn't match with this base, save where we are and
                # get new base
                #fig, ax = plt.subplots(nrows=2, ncols=1)
                #plt.title('run %s' %run_num)
                #plot_matches(ax[0], base_gray, img_gray, base_k, img_k, matches)
                #plot_matches(ax[1], base_gray, img_gray, base_k, img_k, ransac_matches)
                #plt.show()
                bad_match += 1
                if bad_match > 1:
                    print("quiting this base img: %s" %base_path)
                    break
            else:
                match_num += 1
                print("*********** matched %s with %s" %(img_name,
                                                  num_ransac_matches))
                base_img = find_mask(base_name, base_img, img_name,
                                     img, model_robust, channel)
                base_gray = base_img[:,:,channel]
                base_name = get_basename(run_num, base_num, match_num)
                matched_files.append(img_path)
                base_k, base_d = detect_and_extract(orb, base_gray)

        base_out = os.path.join(outdir, base_name)
        print("WRITING", base_out)
        plt.imsave(base_out, base_img)
        [os.remove(f) for f in matched_files]
        unmatched = get_unmatched(run_num-1)
        # remove original file
        # increase to use next base_num
        base_num += 1

#    run_num += 1
#    next_unmatched = get_unmatched(run_num)
#    num_next_unmatched = len(next_unmatched)
#    if num_next_unmatched > 1:
#        if (init_num_unmatched - num_next_unmatched) > 0:
#            find_all_matches(next_unmatched, run_num, min_to_match)
#        elif min_to_match > absolute_min_matches:
#            # reduce our standards a bit and try again
#            find_all_matches(next_unmatched, run_num, min_to_match-5)
#
#def split_find_all_matches(i):
#    """Convert list to arguments for find_all_matches to work with multiprocessing
#    pool"""
#    return find_all_matches(*i)
#
#def multicore_stuff():
#    # TODO
#    # need to address how images are stored to be blended before this can work
#    pool = Pool(processes=cpu_count())
#    split_unmatched = [img_col[:6]]
#    params = [[], 1, 500, 50]
#    all_matched = pool.map(split_find_all_matches,
#                           itertools.izip(split_unmatched,
#                                          itertools.repeat(params)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creating mosaic")
    parser.add_argument('input_dir', help="directory with images to be mosaiced")
    parser.add_argument('output_dir', default="mosaic_dir", help="directory to store mosaic/s")
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

    input_dir = args.input_dir
    output_image_type = args.output_image_type
    input_image_type = args.input_image_type
    outdir = args.output_dir

    if not os.path.exists(args.input_dir):
        print("Error: input directory, %s does not exist" %input_dir)
        sys.exit()

    if not os.path.exists(outdir):
        logging.info("Creating output directory: %s" %outdir)
        os.mkdir(outdir)
    else:
        if args.do_clear:
            # if clear flag is set, remove all existing image files from dir
            print("Removing all %s files from %s" %(output_image_type,
                                                    outdir))
            search = os.path.join(outdir, '*.%s'%output_image_type)
            for f in glob(search):
                os.remove(f)

    channel = 2
    min_matches = 20
    run_num = 0
    from shutil import copyfile
    # copy from original directory to working directory
    in_files = glob(os.path.join(input_dir, '*.%s' %input_image_type))
    import subprocess
    for xx, ifile in enumerate(in_files):
        o_name = get_basename(run_num, xx, 0)
        o_path = os.path.join(outdir, o_name)
        cmd = ['convert', ifile, '-resize', '1500x2000', o_path ]
        if not os.path.exists(o_path):
            print("compressing and converting %s to %s" %(ifile, o_path))
            #copyfile(ifile, o_path)
            subprocess.call(cmd)
    unmatched = get_unmatched(run_num)
    find_all_matches(unmatched, run_num+1, min_matches)


