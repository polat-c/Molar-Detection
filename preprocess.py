from scipy.ndimage.measurements import center_of_mass
import numpy as np
import utils


def get_ctrd_patch(image):
    shape = image.shape
    ctr1, ctr2 = [int(shape[0] / 2), int(shape[1] / 2)]
    return image[ctr1 - 400:ctr1 + 700, ctr2 - 1300:ctr2 + 1300]


def get_halves(image):
    """
    Splits image in two halves
    :param image: image_nda
    :return:
    """
    _, width = image.shape
    left_image = image[:, :int(width / 2)]
    right_image = image[:, int(width / 2):]


    return left_image, np.flip(right_image,1)


def extract_patches(image):
    image_nda = utils.image2nda(image)
    image_nda = utils.verify2dimage(image_nda)
    # Get centered image (all with same size)
    image_nda = get_ctrd_patch(image_nda)

    # Split in halves
    l_image, r_image = get_halves(image_nda)

    return [l_image, r_image]

def _get_mask_size(mask):
    y_min = mask.shape[0]
    y_max = 0
    for i in range(mask.shape[0]):

        ones = np.where(mask[i, :] == 1)[0]
        if len(ones) > 1:
            if ones[0] < y_min:
                y_min = ones[0]
            if ones[-1] > y_max:
                y_max = ones[-1]

    x_min = mask.shape[1]
    x_max = 0
    for i in range(mask.shape[1]):

        ones = np.where(mask[:, i] == 1)[0]
        if len(ones) > 1:
            if ones[0] < x_min:
                x_min = ones[0]
            if ones[-1] > x_max:
                x_max = ones[-1]
    return (x_max-x_min,y_max-y_min)


def find_largest_mask(masks):
    max_sx = 0
    max_sy = 0
    for i,mask in enumerate(masks):
        (sx, sy) = _get_mask_size(mask)
        if sx>max_sx:
            max_sx = sx
        if sy>max_sy:
            max_sy = sy

    return max_sx, max_sy

def build_new_mask(mask):

    # Get mask centroid
    c1, c2 = center_of_mass(mask)
    c1, c2 = int(c1), int(c2)
    # Reshape mask according to largest possible bbox (530,530)
    n_mask = np.zeros(mask.shape)
    n_mask[c1 - 275:c1 + 275, c2 - 275:c2 + 275] = 1

    return n_mask, c1, c2

def preprocess_full(images, masks):
    images_patches = [extract_patches(image) for image in images]
    masks_patches = [extract_patches(mask) for mask in masks]
    flatten = lambda l: [item for sublist in l for item in sublist]
    images_patches = flatten(images_patches)
    masks_patches = flatten(masks_patches)

    new_masks_patches = []
    cs = []
    for mask in masks_patches:
        n_mask_patch, c1, c2 = build_new_mask(mask)
        new_masks_patches.append(n_mask_patch)
        cs.append((c1,c2))
    #new_masks_patches = [(build_new_mask(mask) for mask in masks_patches[:]]

    return images_patches, masks_patches, new_masks_patches, cs

def extract_molar_patch(image, cs):

    image = utils.image2nda(image)
    c1 = cs[0]
    c2 = cs[1]

    molar_patch = image[c1 - 275:c1 + 275, c2 - 275:c2 + 275]
    return molar_patch

def find_mean_std(images):
    # group all images
    images  = np.concatenate(images)
    std = np.std(images)
    mean = np.mean(images)


    return images,mean

################################################

def prepare_yolo_labels_from_masks(mask, box_dimensions=530):
    '''
    Args:
        mask: ndarray of mask image pixel values
    Returns:
        string of a yolo label row: '0 x_center y_center width height'
    '''
    # Image dimensions
    y_len, x_len = mask.shape
    # Get mask centroid
    y, x = center_of_mass(mask)
    # Normalize
    # default bbox dimensions (530,530)
    x_out = x / x_len
    y_out = y / y_len
    width = box_dimensions / x_len
    height = box_dimensions / y_len

    return '0 '+str(x_out)+' '+str(y_out)+' '+str(width)+' '+str(height)
