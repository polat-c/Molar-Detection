import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from torch.nn.functional import interpolate


def check_pair_exists():
    dir_images = 'data//annotated/originals'
    check_table = []
    for file in os.listdir(dir_images):
        filename = os.fsdecode(file)

        if filename.endswith('.dcm'):
            maskname = filename[:-3] + 'gipl'
            file_exists = os.path.isfile('data//annotated/originals/' + maskname)

            check_table.append([filename, file_exists])
    return check_table


def load_data(dir_data='annotated/originals'):
    image_ids = []
    images = []
    masks = []
    dir_images = 'data//' + dir_data
    for file in os.listdir(dir_images):
        filename = os.fsdecode(file)

        # load images
        if filename.endswith('.dcm'):
            # load x-ray
            path_image = os.path.join(dir_images, filename)
            image = sitk.ReadImage(path_image)
            images.append(image)

            # load mask
            maskname = filename[:-3] + 'gipl'
            path_mask = os.path.join(dir_images, maskname)
            mask = sitk.ReadImage(path_mask)
            masks.append(mask)

            # save img id
            image_ids.append(filename[:-4])
        # # load masks
        # elif filename.endswith('.gipl'):
        #
        #     path_mask = os.path.join(dir_images, filename)
        #     mask = sitk.ReadImage(path_mask)
        #     masks.append(mask)

    return images, masks, image_ids


def save_image_and_mask(image_nda, mask_nda, new_mask_nda, filename):
    image = sitk.GetImageFromArray(image_nda)
    mask = sitk.GetImageFromArray(mask_nda)
    new_mask = sitk.GetImageFromArray(new_mask_nda)

    image_path = 'data/annotated/patches/images_nmasks/' + filename + '.dcm'
    mask_path = 'data/annotated/patches/images_nmasks_original/' + filename + '.gipl'
    new_mask_path = 'data/annotated/patches/images_nmasks/' + filename + '.gipl'

    sitk.WriteImage(image, image_path)
    sitk.WriteImage(mask, mask_path)
    sitk.WriteImage(new_mask, new_mask_path)


def save_image(image_nda, filename):
    image = sitk.GetImageFromArray(image_nda)
    image_path = 'data/annotated/patches/images_nmasks_original/' + filename + '.dcm'

    sitk.WriteImage(image, image_path)


def image2nda(image):
    if (type(image).__module__ == np.__name__) is False:
        image_nda = sitk.GetArrayFromImage(image)[0, :, :]
    else:
        image_nda = image
    return image_nda


def verify2dimage(image):
    if len(image.shape) > 2:
        image = image[0, :, :]
    return image


def visualize_image(image):
    # Convert to np
    if (type(image).__module__ == np.__name__) is False:
        image = sitk.GetArrayFromImage(image)

    # Verify 2D
    image = verify2dimage(image)

    # Plot
    plt.imshow(image[:, :])


def visualize_nimages(images, pairs=False):
    if pairs is True:
        fig, axs = plt.subplots(int(len(images) / 2), 2)
    else:
        fig, axs = plt.subplots(len(images), 1)
    for i, image in enumerate(images):
        # Convert to numpy

        if (type(image).__module__ == np.__name__) is False:
            image = sitk.GetArrayFromImage(image)

        # Make sure is 2d
        image = verify2dimage(image)

        # Plot
        if pairs is True:
            axs[int(i / 2), i % 2].imshow(image[:, :])
        else:
            axs[i].imshow(image[:, :])


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def correct_label_nb(label):
    if label <= 3:
        label = label - 1
    elif 4 <= label <= 6:
        label = label - 4

    elif label >= 7:
        label = label - 7

    return int(label)

#################################

def load_data_new(dir_data='annotated'):
    image_ids = []
    images = []
    masks = []
    dir_images = 'data//' + dir_data
    for file in os.listdir(dir_images):
        filename = os.fsdecode(file)

        # load images
        if filename.endswith('.dcm'):

            maskname = filename[:-3] + 'gipl'

            if os.path.exists(os.path.join(dir_images, maskname)):
                # do not try to load if mask doesn'' exist
                # load x-ray
                path_image = os.path.join(dir_images, filename)
                image = sitk.ReadImage(path_image)
                images.append(image)

                # load mask
                path_mask = os.path.join(dir_images, maskname)
                mask = sitk.ReadImage(path_mask)
                masks.append(mask)

                # save img id
                image_ids.append(filename[:-4])
        # # load masks
        # elif filename.endswith('.gipl'):
        #
        #     path_mask = os.path.join(dir_images, filename)
        #     mask = sitk.ReadImage(path_mask)
        #     masks.append(mask)

    return images, masks, image_ids

def save_image_new(image_nda, filename, path, extension = '.dcm'):
    image = sitk.GetImageFromArray(image_nda)
    image_path = path + filename + extension

    sitk.WriteImage(image, image_path)

def load_images(dir_images):
    images = []
    image_ids = []

    for filename in os.listdir(dir_images):
        if filename.endswith('.dcm'):
            path_image = os.path.join(dir_images, filename)
            image = sitk.ReadImage(path_image)
            images.append(image)
            image_ids.append(filename[:-4])

    return images, image_ids

def to_tensor(image):
    image = torch.from_numpy(image.astype(float))
    return image

def normalize(image):
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return image

def interpolate_(image, window_size):
    image = interpolate(image.unsqueeze(0).unsqueeze(0), size=(window_size, window_size))
    image = image.squeeze(0)
    return image


