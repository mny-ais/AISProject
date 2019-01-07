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

def make_conv(input, output, kernel, striding):
    layer = nn.Sequential(
            nn.Conv2D(input, output, kernel_size=kernel, stride=striding),
            nn.BatchNorm2d(output),
            nn.Dropout(0.2),
            nn.ReLU()
            )
    return layer

def make_fc(size):
    layer = nn.Sequential(
            nn.Linear(size, 512),
            nn.Dropout(0.5),
            nn.ReLU()
            )
    return layer


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input image channel, output channels, square convolution

        self.conv1 = make_conv(3, 32, 5, 2)
        self.conv2 = make_conv(32, 32, 3, 1)
        self.conv3 = make_conv(32, 64, 3, 2)
        self.conv4 = make_conv(64, 64, 3, 1)
        self.conv5 = make_conv(64, 128, 3, 2)
        self.conv6 = make_conv(128, 128, 3, 1)
        self.conv7 = make_conv(128, 256, 3, 1)
        self.conv8 = make_conv(256, 256, 3, 1)

        # Error
        self.fc1 = make_fc(1) # 96512?!?
        self.fc2 = make_fc(512)

    def forward(self, input):
        """Forwards function that describes the connections of the network.

        Args:
            ??
        """

        # Forward through Convs
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        #Flatten
        x = x.view(-1, self.num_flat_features(x))

        # Forward through Fully Connected
        x = self.fc1(x)
        x = self.fc2(x)

        # We now need the three main branches

        """
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        """

    def num_flat_features(self, x):
        """
        Courtesy of https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
