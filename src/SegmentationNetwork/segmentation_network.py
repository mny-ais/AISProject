# -*- coding: utf-8 -*-
"""Segmentation Network.

A neural-network based on FCN. It is meant to work together with a network with
fully connected layers to produce steering output for the AirSim simulation. We
use the GoogLeNet model as our encoder. Lots also taken from "Image Segmentation
and Object Detection in Pytorch"

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>

References:
    Fully Convolutional Networks for Semantic Segmentation
    arXiv:1605.06211v1 [cs.CV] 20 May 2016

    Going Deeper with Convolutions
    arXiv:1409.4842 [cs.CV] 17 Sep 2014

    Image Segmentation and Object Detection in Pytorch
    https://github.com/warmspringwinds/pytorch-segmentation-detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from SegmentationNetwork.encoder import Encoder


class SegmentationNetwork(nn.Module):
    def __init__(self):
        """Neural network that can do road segmentation.

        Args:
            num_classes (int): The number of classes to classify.

        """
        super(SegmentationNetwork, self).__init__()

        # Load GoogLeNet
        self.googlenet = GoogLeNet()

        classifier_conv = nn.Conv2d(1024, 3, 1)
        self._normal_initialization(classifier_conv)
        self.classifier_conv = classifier_conv
        self.softmax = nn.Softmax(dim=1)

    def _normal_initialization(self, layer):
        """Initializes a layer with some weights.

        Args:
            layer (nn.Module): The layer to be initialized.
        """
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        """Runs the network.

        Args:
            x (torch.Tensor or torch.cuda.Tensor): A tensor with the input image
            in it.
        """
        # Calculate the spatial dimension for output later
        input_spatial_dim = x.size()[2:]

        # Run the network
        out = self.googlenet(x)
        out = self.classifier_conv(out)

        out = functional.interpolate(input=out, size=(32,160),
                                mode="bilinear")
        print("Out shape: {}, skip2 shape: {}".format(out.shape, self.googlenet.skip2.shape))
        out = torch.cat((out, self.googlenet.skip2))

        out = self.softmax(out)
        return out
