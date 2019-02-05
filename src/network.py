# -*- coding: utf-8 -*-
"""Network.

A neural-network based on conditional imitation learning that is capable of
self-driving. Based on the paper "End-to-end Driving via Conditional Imitation
Learning" by Codevilla et al.

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


class DriveNet(nn.Module):
    def __init__(self):
        """This neural network drives an car based on input images and commands.

        This neural network is made up of 8 convolutions and 2 fully connected
        layers which take an input image from the cameras on the vehicle, and
        then branches it based on which high-level command is given to one of
        three more fully connected layers which output the commands to be given
        to the vehicle.
        """
        super(DriveNet, self).__init__()  # First initialize the superclass

        # Input images are pushed through 8 convolutions to extract features
        self.conv1 = NetworkUtils.make_conv(3, 32, 5, 2, pad=2, dropout=0)
        self.conv2 = NetworkUtils.make_conv(32, 32, 3, 1, dropout=0)
        self.conv3 = NetworkUtils.make_conv(32, 64, 3, 2, dropout=0)
        self.conv4 = NetworkUtils.make_conv(64, 64, 3, 1, dropout=0)
        self.conv5 = NetworkUtils.make_conv(64, 128, 3, 2, dropout=0)
        self.conv6 = NetworkUtils.make_conv(128, 128, 3, 1, dropout=0)
        self.conv7 = NetworkUtils.make_conv(128, 256, 3, 1, dropout=0)
        self.conv8 = NetworkUtils.make_conv(256, 256, 3, 1, dropout=0)

        # 2 fully connected layers to extract the features in the images
        self.fc1 = NetworkUtils.make_fc(81920, dropout=0)  # This must be changed later
        self.fc2 = NetworkUtils.make_fc(512, dropout=0)

        # 2 fully connected layers for each high-level command branch
        self.fc_left_1 = NetworkUtils.make_fc(512, dropout=0)
        self.fc_left_2 = NetworkUtils.make_fc(512, dropout=0)
        self.fc_forward_1 = NetworkUtils.make_fc(512, dropout=0)
        self.fc_forward_2 = NetworkUtils.make_fc(512, dropout=0)
        self.fc_right_1 = NetworkUtils.make_fc(512, dropout=0)
        self.fc_right_2 = NetworkUtils.make_fc(512, dropout=0)

        # Output layer which turns the previous values into steering, throttle,
        # and brakes
        self.fc_out_left = nn.Linear(512, 1)
        self.fc_out_forward = nn.Linear(512,1)
        self.fc_out_right = nn.Linear(512, 1)

        self.counter = 0

    def forward(self, img, car_data, batch_size):
        """Describes the connections within the neural network.

        Args:
            img (torch.Tensor): The input image as a tensor.
            cmd (int): The high level command being given to the model.

        Returns (torch.Tensor):
            The commands to be given to the vehicle to drive in a 3 channel
            tensor representing steering and throttle.
        """

        # Split into single images and data/command
        cmd = car_data[1]

        for i in range(len(img)):

            # Counter used to get the right command from the cmd tensor
            if self.counter >= batch_size:
                self.counter = 0

            # Forward through Convolutions
            x = self.conv1(img[i])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)

            # Flatten to prepare for fully connected layers
            x = x.view(-1, self.num_flat_features(x))

            # Forward through fully connected layers
            x = self.fc1(x)
            x = self.fc2(x)

            # x = self.fc_forward_1(x)
            # x = self.fc_forward_2(x)
            # out = self.fc_out_forward(x)

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

            self.counter += 1

        return out

    @staticmethod
    def num_flat_features(x):
        """Multiplies the number of features for flattening a convolution.

        References:
            https://pytorch.org/tutorials/beginner/blitz/neural_networks_
            tutorial.html
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetworkUtils:
    @staticmethod
    def make_conv(input_channels, output_channels, kernel, stride=1, dropout=0.2, pad=1):
        """Makes a set of modules which represent is convolutional layer.

        Makes a convolution, batchnorm, dropout, and ReLU module set that is
        required by each convolutional layer in the network.

        Args:
            input_channels (int): The number of input channels of the
                                  convolution.
            output_channels (int): The number of output channels of the
                                   convolution.
            kernel (int): The size of the kernel.
            stride (int): The stride of the convolution, defaults to 1.

        Returns (nn.Sequential):
            A sequential layer object that represents the required modules for
            each convolution layer.
        """
        layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel,
                      stride=stride, padding=pad),
            # nn.BatchNorm2d(output_channels),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        return layer

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
