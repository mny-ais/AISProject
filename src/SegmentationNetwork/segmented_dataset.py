#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
"""Segmented Dataset.

This module provides a dataset that contains the real image, a segmented
version of the image, a tuple with the steering and throttle, and the higher
level command. It is also capable of augmenting this dataset.

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import os
import torch
import csv
import numpy as np
from skimage import io
from torch.utils.data import Dataset
try:
    import imgaug as ia
    from imgaug import augmenters as iaa
    with_aug = True
except ImportError:
    print("imgaug not installed. Running without augmentation.")
    with_aug = False


if with_aug:
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


class SegmentedDataset(Dataset):
    def __init__(self, root_dir):
        """Dataset object that turns the images and csv file into a dataset.
            Args:
                root_dir (string): The directory of both the images and csv file
        """
        super().__init__()

        self.dataset = []
        # self.drive_data = pd.read_csv(csv_file, sep=',')
        csv_file = os.path.join(root_dir, "control_input.csv")
        self.drive_data = list(csv.reader(open(csv_file, mode='r')))
        del self.drive_data[0]
        self.root_dir = root_dir

    def __len__(self):
        """Returns length of the data."""
        return len(self.drive_data)

    def __getitem__(self, idx):
        """Returns dataset.
        """
        actual_index = int(self.drive_data[idx][0])
        # print(int(self.drive_data.iloc[idx][0]))
        item = self.process_sample(idx, actual_index)

        while item is None:
            idx += 1
            actual_index = int(self.drive_data[idx][0])
            item = self.process_sample(idx, actual_index)
            if idx >= len(self.drive_data):
                idx = 0

        return item

    def process_sample(self, idx, actual_index):
        """Returns next transformed datapoint in correct format for the model.

        Returns:
            (dict) in the form {"image": torch.Tensor,
                                "seg": torch.Tensor,
                                "vehicle_commands": torch.Tensor,
                                "cmd": int
        """
        file_name = 'image_{:0>5d}-cam_0.png'.format(actual_index)
        seg_name = 'seg_{:0>5d}-cam_0.png'.format(actual_index)

        img_name = os.path.join(self.root_dir, file_name)
        seg_name = os.path.join(self.root_dir, seg_name)

        sample = None

        if os.path.isfile(img_name):
            rgb = io.imread(img_name)
            seg = self.process_seg(io.imread(seg_name))

            cur_row = self.drive_data[idx]

            for i in range(len(cur_row)):
                cur_row[i] = float(cur_row[i])

            # This is if we're training both the steering and the throttle
            # vehicle_commands = torch.tensor([cur_row[1], cur_row[2]]).float()

            vehicle_commands = torch.tensor([cur_row[1]]).float()  # only
                                                                   # steering
            sample = {"image": self.to_tensor(rgb),
                      "seg": seg,
                      "vehicle_commands": vehicle_commands,
                      "cmd": cur_row[5]}

        else:
            print("image not found by data_loader.py: {}".format(actual_index))
            pass

        return sample

    @staticmethod
    def to_tensor(image):
        """Converts images and data to tensor format and augments it."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if with_aug:
            # apply image augmentation sequential
            image = seq.augment_images(image)



        return torch.from_numpy(image).to(dtype=torch.float)

    @staticmethod
    def process_seg(segmented_image):
        """Processes the segmented image to be used for ground truth.

        The idea is that the segmented image should be processed such that it
        becomes only a 3 dimensional tensor, with the depth dimension holding
        only information for drivable roads and road limits. 1 will be drivable
        roads and 2 will be road edges. Also transposes it to H X W.
        """
        shape = segmented_image.shape
        seg = np.ndarray([64, 320])
        seg.fill(3)

        for i in range(shape[0]):
            for j in range(shape[1]):
                # Here we modify each class to only show roads and road limits
                color = segmented_image[i][j]
                # Nothing
                if 34 < color[0] < 120 and 165 < color[1]:
                    seg[i][j] = 0
                else:
                    # road 1
                    if color[0] < 31 and 119 < color[1]:
                        seg[i][j] = 1
                    # road 2
                    elif 234 < color[1]:
                        seg[i][j] = 1
                    # Intersections
                    elif 100 < color[0] < 140:
                        seg[i][j] = 1
                    # Lines (reds)
                    elif 190 < color[0]:
                        seg[i][j] = 2
                    # Lines (blue)
                    elif color[0] < 70 and 190 < color[2]:
                        seg[i][j] = 2

                    # Final pass
                    if seg[i][j] == 3:
                        seg[i][j] = 1
        return seg
