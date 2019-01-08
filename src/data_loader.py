#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-


from __future__ import print_function, division
import os
import os.path as op
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imgaug as ia
from imgaug import augmenters as iaa


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('CSV:\n')


class DrivingSimDataset(Dataset):
    """
        Driving Simulation Dataset
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
            Args:
                The CSV Data Filename
                The Directory of images and csv file
                The transforms used (blur, brightness, contrast, saturation, hue)
	"""
        # TODO : Check if default for header works
        self.drive_data = pd.read_csv(os.path.join(root_dir, csv_file), sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
            Returns length of the data
        """
        return len(self.drive_data)

    def __getitem__(self, idx):
        """
            Returns next transformed datapoint in correct format for the model
        """


        file_name = 'image_' + str(idx) + '.png'
        img_name = os.path.join(self.root_dir, file_name)
        image = io.imread(img_name)

        cur_row = self.drive_data.iloc[idx, 0:5].as_matrix()
        cur_row = cur_row.astype('float')

        sample = {'image': image, 'drive_data': cur_row}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, drive_data = sample['image'], sample['drive_data']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        [B,G,R] = np.dsplit(image,image.shape[-1])

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'drive_data': torch.from_numpy(drive_data)}

"""
The Image Augmentations
courtesy: https://github.com/mvpcom/carlaILTrainer/blob/master/carlaTrain.ipynb
"""

st = lambda aug: iaa.Sometimes(0.4, aug)
oc = lambda aug: iaa.Sometimes(0.3, aug)
rl = lambda aug: iaa.Sometimes(0.09, aug)
seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))), # blur images with a sigma between 0 and 1.5
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)), # add gaussian noise to images
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iaa.Add((-40, 40), per_channel=0.5)), # change brightness of images (by -X to Y of original value)
        st(iaa.Multiply((0.10, 2.5), per_channel=0.2)), # change brightness of images (X-Y% of original value)
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)), # improve or worsen the contrast
        #rl(iaa.Grayscale((0.0, 1))), # put grayscale
], random_order=True)



transformed_dataset = DrivingSimDataset(csv_file='test.csv',
                                           root_dir='/home/mr492/Documents/5.Semester/RC-Learn/test',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['drive_data'].size())


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

