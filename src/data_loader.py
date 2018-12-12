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

        self.drive_data = pd.read_csv(os.path.join(root_dir, csv_file), sep=',', header=None)
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

        
        cur_row = self.drive_data.iloc[idx, 0:6].as_matrix()

        # Convert data to floatable types

        # Handbrake value conversion : FALSE=0 TRUE=1
        if cur_row[idx, 4] = "FALSE":
            cur_row[idx, 4] = 0
        else:
            cur_row[idx, 4] = 1
            
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
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'drive_data': torch.from_numpy(drive_data)}


transformed_dataset = DrivingSimDataset(csv_file='test.csv',
                                           root_dir='/home/mr492/Documents/5.Semester/RC-Learn/test',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['drive_data'].size())

    if i == 3:
        break

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

