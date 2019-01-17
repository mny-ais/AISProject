#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
"""DrivingSimDataset.

This module creates a class that extends the torch Dataset object for our
specific use case. Each data point in the dataset contains an image and a tuple.
The tuple contains steering and throttle, and the higher level command. This
module also augments the training dataset.

Authors:
    Maximilian Roth
    Nina Pant
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>

References:
    Sequential for The Image Augmentations
    courtesy: https://github.com/mvpcom/carlaILTrainer/blob/master/
              carlaTrain.ipynb
"""
from __future__ import print_function, division

import os

import torch

import pandas as pd

from skimage import io

from torch.utils.data import Dataset

"""
import imgaug as ia
from imgaug import augmenters as iaa
"""

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


# Here we define probabilities
def st(aug):
    """Defines the "sometimes" probability value."""
    return iaa.Sometimes(0.4, aug)


def oc(aug):
    """Defines the "occasionally" probability value."""
    return iaa.Sometimes(0.3, aug)


def rl(aug):
    """Defines the "rarely" probability value."""
    return iaa.Sometimes(0.09, aug)

# Now we define the sequential
"""
seq = iaa.Sequential([
    # blur images with a sigma between 0 and 1.5
    rl(iaa.GaussianBlur((0, 1.5))),
    # randomly remove up to X% of the pixels
    oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),
    # randomly remove up to X% of the pixels
    oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),
                         per_channel=0.5)),
    # change brightness of images (by -X to Y of original value)
    oc(iaa.Add((-40, 40), per_channel=0.5)),
    # change brightness of images (X-Y% of original value)
    st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),
    # improve or worsen the contrast
    rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
], random_order=True)
"""


class DrivingSimDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """Dataset object that turns the images and csv file into a dataset.
            Args:
                csv_file (string): The CSV data file address
                root_dir (string): The directory of both the images and csv file
        """
        super().__init__()

        # TODO : Check if default for header works
        self.dataset = []
        self.drive_data = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir

    def __len__(self):
        """Returns length of the data."""
        return len(self.drive_data)

    def __getitem__(self, idx):
        """Returns dataset

        Args:
            **kwargs: Not sure yet. Base class has them.
        """

        item  = process_img(idx)

        return item

    def process_img(self, idx):
        """Returns next transformed datapoint in correct format for the model.
        """
        file_name = 'image_' + str(idx) + '.png'
        img_name = os.path.join(self.root_dir, file_name)
        image = io.imread(img_name)

        cur_row = self.drive_data.iloc[idx, 0:5].as_matrix()
        cur_row = cur_row.astype('float')

        sample = [image, cur_row]
        sample = to_tensor(sample)

        return sample

    @staticmethod
    def to_tensor(sample):
        """ converts images and data to tensor format
        """
        image = sample[0]
        drive_data = sample[1]
        # apply image augmentation sequential
        # image = seq.augment_images(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(drive_data)

    def __to_processed_package(self):

        for i in range(0, self.__len__()):
            self.__add__(self.to_tensor(self.process_img(i)))


"""
transformed_dataset = DrivingSimDataset(csv_file='test.csv',
                                           root_dir='/home/mr492/Documents/5.Semester/RC-Learn/test')

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['drive_data'].size())
"""


"""
image_num = csv_file.iloc[0, 0]
steering = csv_file.iloc[0, 1]
throttle = csv_file.iloc[0, 2]
handbrake = csv_file.iloc[0, 3]
noise = csv_file.iloc[0, 4]
gear = csv_file.iloc[0, 5]
hl_comm = csv_file.iloc[0, 6]

print(image_num)
print(steering)
print(throttle)
print(handbrake)
print(noise)
print(gear)
print(hl_comm)

print(csv_file)
"""
