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

import torchvision

import torch

import torch.nn as nn

import torch.nn.functional as F


def make_conv(input, output, kernel, stride=1):
    """Makes a set of modules which represent is convolutional layer.

    Makes a convolution, batchnorm, dropout, and ReLU module set that is
    required by each convolutional layer in the network.

    Args:
        input (int): The number of input channels of the convolution.
        output (int): The number of output channels of the convolution.
        kernel (int): The size of the kernel.
        stride (int): The stride of the convolution, defaults to 1.

    Returns (nn.Sequential):
        A sequential layer object that represents the required modules for each
        convolution layer.
    """
    layer = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=kernel, stride=stride),
            nn.BatchNorm2d(output),
            nn.Dropout(0.2),
            nn.ReLU()
            )
    return layer


def make_fc(size, output=512):
    """Makes a set modules which represent a fully connected layer.

    Args:
        size (int): The number of units in the fully connected layer.

    Returns (nn.Sequential):
        A sequential layer object that represents the required modules for each
        fully connected layer.
    """
    layer = nn.Sequential(
            nn.Linear(size, output),
            nn.Dropout(0.5),
            nn.ReLU()
            )
    return layer


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

        # Save the last output, so we can calculate the loss using it
        self.out = None

        # Save the loss, so we can use it to backpropagate
        self.loss = None
        self.criterion = nn.MSELoss()  # This only needs to be initialized once

        # Input images are pushed through 8 convolutions to extract features
        self.conv1 = make_conv(3, 32, 5, 2)
        self.conv2 = make_conv(32, 32, 3, 1)
        self.conv3 = make_conv(32, 64, 3, 2)
        self.conv4 = make_conv(64, 64, 3, 1)
        self.conv5 = make_conv(64, 128, 3, 2)
        self.conv6 = make_conv(128, 128, 3, 1)
        self.conv7 = make_conv(128, 256, 3, 1)
        self.conv8 = make_conv(256, 256, 3, 1)

        # 2 fully connected layers to extract the features in the images
        self.fc1 = make_fc(8192)
        self.fc2 = make_fc(512)

        # 2 fully connected layers for each high-level command branch
        self.fc_left_1 = make_fc(512)
        self.fc_left_2 = make_fc(512)
        self.fc_forward_1 = make_fc(512)
        self.fc_forward_2 = make_fc(512)
        self.fc_right_1 = make_fc(512)
        self.fc_right_2 = make_fc(512)

        # Output layer which turns the previous values into steering, throttle,
        # and brakes
        self.fc_out = make_fc(512, 3)

    def forward(self, img, cmd):
        """Describes the connections within the neural network.

        Args:
            img (torch.Tensor): The input image as a tensor.
            cmd (int): The high level command being given to the model.

        Returns (torch.Tensor):
            The commands to be given to the vehicle to drive.
        """

        # Forward through Convolutions
        x = self.conv1(img)
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

        # Branch according to the higher level commands
        # -1 left, 0 forward, 1 right
        if cmd == 0:
            x = self.fc_forward_1(x)
            x = self.fc_forward_2(x)

        elif cmd == -1:
            x = self.fc_left_1(x)
            x = self.fc_left_2(x)

        else:
            x = self.fc_right_1(x)
            x = self.fc_right_2(x)

        self.out = self.fc_out(x)

        return self.out

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


    def loss_function(self, target):
        """Calculates the loss based on a target tensor.

        Args:
            target (torch.Tensor): The (1 x 3) ground truth tensor,

        Returns:
            The loss criterion. Also saves this internally.
        """
        if self.out == None:
            raise ValueError("forward() has not been run.")

        self.loss = self.criterion(self.out, target)

        return self.loss  # Return the loss, in case it is necessary


if __name__ == "__main__":
    net = DriveNet()
    #print(net)

    input = torch.randn(1, 3, 200, 88)
    out = net(input, 0)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 3))
