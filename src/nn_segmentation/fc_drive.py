# -*- coding: utf-8 -*-
"""Fully Connected Drive Network.

A neural-network that takes a segmented image and drives with it.

Authors:
    Maximilian Roth
    Nina Pant
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>

References:
    End-to-end Driving via Conditional Imitation Learning
    arXiv:1710.02410v2 [cs.RO] 2 Mar 2018
"""

from torch import unsqueeze, cat
import torch.nn as nn


class FCD(nn.Module):
    def __init__(self):
        """This neural network drives an car based on input images and commands.

        This neural network is made up of 2 fully connected layers which take
        an input image from the segmentation network, and then branches it based
        on which high-level command is given to one of three more fully
        connected layers which output the commands to be given to the vehicle.
        """
        super(FCD, self).__init__()  # First initialize the superclass

        # 2 fully connected layers to extract the features in the images
        self.fc1 = self.make_fc(61440, dropout=0)  # This must be changed later
        self.fc2 = self.make_fc(512, dropout=0)

        # 2 fully connected layers for each high-level command branch
        self.fc_left_1 = self.make_fc(512, dropout=0.5)
        self.fc_left_2 = self.make_fc(512, dropout=0.5)
        self.fc_forward_1 = self.make_fc(512, dropout=0.5)
        self.fc_forward_2 = self.make_fc(512, dropout=0.5)
        self.fc_right_1 = self.make_fc(512, dropout=0.5)
        self.fc_right_2 = self.make_fc(512, dropout=0.5)

        # Output layer which turns the previous values into steering, throttle,
        # and brakes
        self.fc_out_left = nn.Linear(512, 1)
        self.fc_out_forward = nn.Linear(512, 1)
        self.fc_out_right = nn.Linear(512, 1)

    def forward(self, img, cmd):
        """Describes the connections within the neural network.

        Args:
            img (torch.Tensor): The input image as a tensor.
            cmd (list(int)): The high level command being given to the model.

        Returns (torch.Tensor):
            The steering commands to be given to the vehicle to drive in a 1x1
            tensor.
        """

        for i in range(img.shape[0]):
            # Forward through fully connected layers
            x = self.fc1(img[i])
            x = self.fc2(x)

            current_cmd = cmd[i]

            # Branch according to the higher level commands
            # -1 left, 0 forward, 1 right
            if current_cmd == 0:
                x = self.fc_forward_1(x)
                x = self.fc_forward_2(x)
                x = self.fc_out_forward(x)

            elif current_cmd == -1:
                x = self.fc_left_1(x)
                x = self.fc_left_2(x)
                x = self.fc_out_left(x)

            else:
                x = self.fc_right_1(x)
                x = self.fc_right_2(x)
                x = self.fc_out_right(x)
            if i == 0:
                out = x
            else:
                out = cat((out, x))

        return out

    @staticmethod
    def make_fc(size, output=512, dropout=0.5):
        """Makes a set modules which represent a fully connected layer.

        Args:
            size (int): The number of units in the fully connected layer.
            output (int): The number of units in the output.

        Returns (nn.Sequential):
            A sequential layer object that represents the required modules for
            each fully connected layer.
        """
        layer = nn.Sequential(
            nn.Linear(size, output),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        return layer
