from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
import SimpleITK as sitk
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.ndimage.measurements import center_of_mass
import preprocess

import utils


class DentalImagingDataset(Dataset):

    def __init__(self, csv_file,
                 root_dir,
                 molar_guarantee=False,
                 classific_model_n=None,
                 transform=None,
                 one_hot=False):

        self.annotations = pd.read_csv(csv_file)
        if molar_guarantee is True:  # molar problem classification
            self.annotations = self.annotations[self.annotations['molar_yn'] == 1].iloc[:, [0, 2, 3, 4]]
            self.classific_model_n = classific_model_n

        else:
            self.classific_model_n = classific_model_n + 1

        self.rootdir = root_dir
        self.transform = transform
        self.molar_guarantee = molar_guarantee

        self.one_hot = one_hot

        # One-hot encoding
        if one_hot is True:
            self.one_hot_enc = OneHotEncoder()
            self.one_hot_enc.fit(self.annotations.iloc[:, 1:].values)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Image
        file_name = self.annotations.iloc[idx, 0] + '.dcm'
        file_path = os.path.join(self.rootdir, file_name)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))

        # Labels

        labels = pd.to_numeric(self.annotations.iloc[idx, self.classific_model_n])
        if self.molar_guarantee is True:
            labels = utils.correct_label_nb(labels)
        else:
            labels = int(labels)

        ## Guarantee 2D
        # if labels.shape[-1] == 1 and self.one_hot is True:
        #     labels = [labels]

        # convert to one hot encoding
        if self.one_hot is True:
            labels = self.one_hot_enc.transform(labels).toarray()

        sample = [image, labels]

        if self.transform:
            sample = self.transform(sample)

        return sample


# class Image2Tensor(object):
#     """Convert SITK images to torch tensor"""

#     def __call__(self, sample):
#         image = sitk.GetTensorFromImage(sample[0])
#         labels = torch.from_numpy(sample[1])

#         return [image,labels]

class Image2NDArray(object):
    """Convert SITK images to ndarray (and correct labels size"""

    def __call__(self, sample):
        image = utils.image2nda(sample[0])
        if type(sample[1]) == tuple: # ROI center coordinates
            labels = sample[1]
        elif len(sample[1].shape) == 3:  # if it's a mask
            labels = utils.image2nda(sample[1])
        # labels = np.reshape(labels,labels.shape[1])
        return [image, labels]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = torch.from_numpy(sample[0].astype(float))
        image = image.float()
        if type(sample[1]) == tuple:  # ROI center coordinates
            labels = torch.from_numpy(np.array(sample[1]))
        elif len(sample[1].shape) == 3: # if it's a mask
            labels = torch.from_numpy(sample[1].astype(float))
            labels = labels.float()

        return [image,
                labels]


class Normalize(object):
    """Normalizes the image tensor"""

    def __call__(self, sample):
        mean = 12526.53
        std = 30368.829025877254
        image = sample[0]
        image = F.normalize(sample[0], [mean], [std])

        labels = sample[1]
        return [image,
                labels]


class Interpolate(object):
    """Reduces size of image"""

    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, sample):
        image = sample[0]
        image = interpolate(image.unsqueeze(0), size=(self.window_size, self.window_size))
        image = image.squeeze(0)

        if (type(sample[1]) == tuple) or len(sample[1].shape) == 1:  # ROI center coordinates
            labels = sample[1]
        elif len(sample[1].shape) == 3:  # if it's a mask
            labels = sample[1]
            labels = interpolate(labels.unsqueeze(0), size=(self.window_size, self.window_size))
            labels = labels.squeeze(0)

        return [image, labels]


def return_data(args):
    molar_guarantee = args.molar_guarantee
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    test_portion = args.test_portion
    num_workers = args.num_workers
    csv_file = args.csv_file
    one_hot = args.one_hot
    window_size = args.window_size
    classific_model_n = args.classific_model_n

    def split_sets(dset, test_portion):
        # Creating data indices for training and validation splits:
        dataset_size = len(dset)
        test_size = int(dataset_size * test_portion)
        train_size = int(dataset_size - test_size)

        valid_set, test_set = random_split(dset, [train_size, test_size])

        return valid_set, test_set

    dset = DentalImagingDataset(csv_file,
                                root_dir=dset_dir,
                                molar_guarantee=molar_guarantee,
                                classific_model_n=classific_model_n,
                                transform=transforms.Compose([
                                    Image2NDArray(),
                                    ToTensor(),
                                    Normalize(),
                                    Interpolate(window_size=window_size)
                                ]))

    train_set, test_set = split_sets(dset, test_portion)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False)

    return train_loader, test_loader

##################################################################

class MolarDetectionDataset(Dataset):

    def __init__(self, csv_file,
                 root_dir,
                 molar_guarantee=False,
                 predictor='Regressor',
                 transform=None):

        self.annotations = pd.read_csv(csv_file)
        self.rootdir = root_dir
        self.transform = transform
        self.molar_guarantee = molar_guarantee
        self.predictor = predictor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Image
        file_name = self.annotations.iloc[idx, 0] + '.dcm'
        file_path = os.path.join(self.rootdir, file_name)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))

        # ROI centers
        mask_name = self.annotations.iloc[idx, 0] + '.gipl'
        mask_path = os.path.join(self.rootdir, mask_name)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        #mask = mask.reshape(1100,1300) # hardcoded mask image dimensions, after preprocessing

        if self.predictor == 'Regressor':
            roi_center = center_of_mass(mask) # roi_center[0] = y , roi_center[1] = x
            roi_center =(roi_center[1], roi_center[2])
            sample = [image, roi_center]
        elif self.predictor == 'Classifier':
            sample = [image, mask] # image.shape = (1,1100,1300)  ,  mask.shape = (1,1100,1300)

        ## Guarantee 2D
        # if labels.shape[-1] == 1 and self.one_hot is True:
        #     labels = [labels]

        if self.transform: # no need to change transform, works same for labels and ROI centers
            sample = self.transform(sample)

        return sample


def return_detection_data(args):
    molar_guarantee = args.molar_guarantee
    predictor = args.predictor
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    test_portion = args.test_portion
    num_workers = args.num_workers
    csv_file = args.csv_file
    window_size = args.window_size

    def split_sets(dset, test_portion):
        # Creating data indices for training and validation splits:
        dataset_size = len(dset)
        test_size = int(dataset_size * test_portion)
        train_size = int(dataset_size - test_size)

        valid_set, test_set = random_split(dset, [train_size, test_size])

        return valid_set, test_set

    dset = MolarDetectionDataset(csv_file,
                                root_dir=dset_dir,
                                molar_guarantee=molar_guarantee,
                                predictor=predictor,
                                transform=transforms.Compose([
                                    Image2NDArray(),
                                    ToTensor(),
                                    Normalize(),
                                    Interpolate(window_size=window_size)
                                ]))

    train_set, test_set = split_sets(dset, test_portion)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False)

    return train_loader, test_loader