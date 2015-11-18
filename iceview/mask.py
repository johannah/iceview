import numpy as np
from skimage.transform import warp, SimilarityTransform, AffineTransform, ProjectiveTransform
from subprocess import Popen, PIPE
import matplotlib.pylab as plt
import config
import os

from skimage.data import imread
def generate_costs(diff_image, mask, vertical=True, gradient_cutoff=2.):
    """
    Ensures equal-cost paths from edges to region of interest.

    Parameters
    ----------
    diff_image : ndarray of floats
        Difference of two overlapping images.
    mask : ndarray of bools
        Mask representing the region of interest in ``diff_image``.
    vertical : bool
        Control operation orientation.
    gradient_cutoff : float
        Controls how far out of parallel lines can be to edges before
        correction is terminated. The default (2.) is good for most cases.

    Returns
    -------
    costs_arr : ndarray of floats
        Adjusted costs array, ready for use.
    """
    if vertical is not True:
        return tweak_costs(diff_image.T, mask.T, vertical=vertical,
                           gradient_cutoff=gradient_cutoff).T

    # Start with a high-cost array of 1's
    costs_arr = np.ones_like(diff_image)

    # Obtain extent of overlap
    row, col = mask.nonzero()
    cmin = col.min()
    cmax = col.max()

    # Label discrete regions
    cslice = slice(cmin, cmax + 1)
    labels = label(mask[:, cslice])

    # Find distance from edge to region
    upper = (labels == 0).sum(axis=0)
    lower = (labels == 2).sum(axis=0)

    # Reject areas of high change
    ugood = np.abs(np.gradient(upper)) < gradient_cutoff
    lgood = np.abs(np.gradient(lower)) < gradient_cutoff

    # Give areas slightly farther from edge a cost break
    costs_upper = np.ones_like(upper, dtype=np.float64)
    costs_lower = np.ones_like(lower, dtype=np.float64)
    costs_upper[ugood] = upper.min() / np.maximum(upper[ugood], 1)
    costs_lower[lgood] = lower.min() / np.maximum(lower[lgood], 1)

    # Expand from 1d back to 2d
    vdist = mask.shape[0]
    costs_upper = costs_upper[np.newaxis, :].repeat(vdist, axis=0)
    costs_lower = costs_lower[np.newaxis, :].repeat(vdist, axis=0)

    # Place these in output array
    costs_arr[:, cslice] = costs_upper * (labels == 0)
    costs_arr[:, cslice] +=  costs_lower * (labels == 2)

    # Finally, place the difference image
    costs_arr[mask] = diff_image[mask]

    return costs_arr

def find_output_shape(base_img, model_robust):
    r, c = base_img.shape[:2]

    corners = np.array([[0,0],
                        [0,r],
                        [c,0],
                        [c,r]])

    warped_corners = model_robust(corners)
    all_corners = np.vstack((warped_corners, corners))
    # The overally output shape will be max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)
    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape[::-1]).astype(int)
    return output_shape, corner_min

def remove_empty_edges(img):
    def get_mask(sums):
        if sum(sums) > 0:
            first = sums.index(1)
            last = sums[::-1].index(1)

            num_ones = (len(sums)-first)-last
            out = [0]*first + [1]*num_ones + [0]*last
            return out
        else:
            return sums

    #for ax in range(len(img.shape)):
    axes = [0, 1]
    for ax in range(2):
        sums = np.sum(img, axis=axes[ax])
        # make a mask of zero lines in image
        sums= [bool(x) for x in sums]
        empty = get_mask(list(sums))
        img = np.compress(empty, img, axis=axes[ax-1])
    return img

def add_alpha(img, mask=None):
    """
    Adds a masked alpha channel to an image.

    Parameters
    ----------
    img : (M, N[, 3]) ndarray
        Image data, should be rank-2 or rank-3 with RGB channels
    mask : (M, N[, 3]) ndarray, optional
        Mask to be applied. If None, the alpha channel is added
        with full opacity assumed (1) at all locations.
    """
    if mask is None:
        mask = np.ones_like(img)

    if img.ndim == 2:
        img = gray2rgb(img)

    return np.dstack((img, mask))


def add_alpha_channel(img, background=-1):
    """Add an alpha layer to the image.

    The alpha layer is set to 1 for foreground and 0 for background.
    """
    if img.ndim == 2:
        img = gray2rgb(img)
    return np.dstack((img, (img != background)))

def minimum_cost_merge(base_warped, img_warped, base_mask, img_mask):
    # Start with the absolute value of the difference image.
    # np.abs is necessary because we don't want negative costs!
    costs = generate_costs(np.abs(img_warped - base_warped),
                           img_mask & base_mask)
    costs[0,  :] = 0
    costs[-1, :] = 0

    output_shape = base_warped.shape
    # Arguments are:
    #   cost array
    #   start pt
    #   end pt
    #   can it traverse diagonally
    ymax = output_shape[1] - 1
    xmax = output_shape[0] - 1

    # Start anywhere along the top and bottom, left of center.
    mask_pts01 = [[0,    ymax // 3],
                  [xmax, ymax // 3]]

    # Start anywhere along the top and bottom, right of center.
    mask_pts12 = [[0,    2*ymax // 3],
                  [xmax, 2*ymax // 3]]

    pts, _ = route_through_array(costs, mask_pts01[0], mask_pts01[1], fully_connected=True)

    # Convert list of lists to 2d coordinate array for easier indexing
    pts = np.array(pts)

    # Start with an array of zeros and place the path
    _img_mask = np.zeros_like(img_warped, dtype=np.uint8)
    _img_mask[pts[:, 0], pts[:, 1]] = 1


    # Labeling starts with zero at point (0, 0)
    _img_mask[label(_img_mask, connectivity=1) == 0] = 1

    _base_mask = ~(_img_mask).astype(bool)

    base_color = gray2rgb(base_warped)
    img_color = gray2rgb(img_warped)
    base_final = add_alpha(base_warped, _base_mask)
    img_final = add_alpha(img_warped, _img_mask)

    # Start with empty image
    base_combined = np.zeros_like(base_warped)

    base_combined += base_warped * _base_mask
    base_combined += img_warped * _img_mask

    return base_combined

def find_mask(base_img, img, model_robust):
    # what type of interpolation
    # 0: nearest-neighbor
    # 1: bi-linear
    warp_order = 1

    output_shape, corner_min = find_output_shape(base_img, model_robust)
    #print("output_shape", output_shape, corner_min)
    #print(model_robust.scale, model_robust.translation, model_robust.rotation)

    # This in-plane offset is the only necessary transformation for the base image
    offset = SimilarityTransform(translation= -corner_min)
    base_warped = warp(base_img[:,:,2], offset.inverse, order=warp_order,
                      output_shape = output_shape, cval=-1)
    base_color = warp(base_img, offset.inverse, order=warp_order,
                      output_shape = output_shape, cval=-1)
    # warp image corners to new position in mosaic
    transform = (model_robust + offset).inverse

    img_warped = warp(img[:,:,2], transform, order=warp_order,
                      output_shape=output_shape, cval=-1)
    img_color = warp(img, transform, order=warp_order,
                      output_shape=output_shape, cval=-1)
    base_mask = (base_warped != -1)
    base_warped[~base_mask] = 0

    img_mask = (img_warped != -1)
    img_warped[~img_mask] = 0

    #convert to rgb
    #base_alpha = add_alpha(base_color, base_mask)
    img_alpha = np.dstack((img_color, img_mask))
    base_alpha = np.dstack((base_color, base_mask))

    plt.imsave(config.tmp_base, base_alpha )
    plt.imsave(config.tmp_img, img_alpha )
    cmd = [config.path_to_enblend, config.tmp_base, config.tmp_img,
           '-o', config.tmp_out]

    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode
    # remove alpha channel

    if os.path.exists(config.tmp_out):
        out = imread(config.tmp_out)[:,:,:3]
    else:
        print("couldnt find out image")
        print(rc, output, err)
        plt.figure()
        plt.imshow(base_alpha)
        plt.figure()

        plt.imshow(img_alpha)
        plt.show()
        out = base_alpha[:,:,:3]
    #if you don't have enblend, you can use one of these
    #merged_img = simple_merge(base_warped, img_warped, base_mask, img_mask)
    #merged_img = minimum_cost_merge(base_warped, img_warped, base_mask, img_mask)
    #merged_edges = remove_empty_edges(merged_img)
    return out


def find_alpha(base_img, img, model_robust):
    # what type of interpolation
    # 0: nearest-neighbor
    # 1: bi-linear
    warp_order = 1

    output_shape, corner_min = find_output_shape(base_img, model_robust)
    #print("output_shape", output_shape, corner_min)
    #print(model_robust.scale, model_robust.translation, model_robust.rotation)

    # This in-plane offset is the only necessary transformation for the base image
    offset = SimilarityTransform(translation= -corner_min)
    base_warped = warp(base_img[:,:,2], offset.inverse, order=warp_order,
                      output_shape = output_shape, cval=-1)
    base_color = warp(base_img, offset.inverse, order=warp_order,
                      output_shape = output_shape, cval=-1)
    # warp image corners to new position in mosaic
    transform = (model_robust + offset).inverse

    #img_warped = warp(img[:,:,2], transform, order=warp_order,
    #                  output_shape=output_shape, cval=-1)
    img_color = warp(img, transform, order=warp_order,
                      output_shape=output_shape, cval=-1)
    #base_mask = (base_warped != -1)
    #base_warped[~base_mask] = 0

    img_mask = (img_warped != -1)
    #img_warped[~img_mask] = 0

    #convert to rgb
    #base_alpha = add_alpha(base_color, base_mask)
    img_alpha = np.dstack((img_color, img_mask))
    #base_alpha = np.dstack((base_color, base_mask))

    #plt.imsave(tmp_base, base_alpha )
    #plt.imsave(tmp_img, img_alpha )
    #cmd = [path_to_enblend, tmp_base, tmp_img, '-o', tmp_out]

    #p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    #output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    #rc = p.returncode
    # remove alpha channel

    #if os.path.exists(tmp_out):
    #    out = imread(tmp_out)[:,:,:3]
    #else:
    #    print("couldnt find out image")
    #    print(rc, output, err)
    #    plt.figure()
    #    plt.imshow(base_alpha)
    #    plt.figure()#

    #    plt.imshow(img_alpha)
    #    plt.show()
    #    out = base_alpha[:,:,:3]
    #if you don't have enblend, you can use one of these
    #merged_img = simple_merge(base_warped, img_warped, base_mask, img_mask)
    #merged_img = minimum_cost_merge(base_warped, img_warped, base_mask, img_mask)
    #merged_edges = remove_empty_edges(merged_img)
    return tmp_alpha



def simple_merge(base_warped, img_warped, base_mask, img_mask):

    # Add the three images together. This could create dtype overflows!
    # We know they are are floating point images after warping, so it's OK.
    merged = (base_warped + img_warped)

    # Track the overlap by adding the masks together
    # Multiply by 1.0 for bool -> float conversion
    overlap = (base_mask * 1.0 + img_mask)

    # Normalize through division by `overlap` - but ensure the minimum is 1
    norm = merged / np.maximum(overlap, 1)

    return norm
