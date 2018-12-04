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

path_csv = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test.csv")
# path_csv = '~/Documents/5.Semester/RC-Learn/pytorch-tut/test.csv'

csv_file = pd.read_csv(path_csv, sep=',', header=None)

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

# print(csv_file)
