""""Segmentation Drive.

This class provides the combined network that drives the vehicle based on
images from the camera by first segmenting it. It uses the SegmentationNetwork
class as well as the FCD class

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import torch.nn as nn
from SegmentationNetwork.segmentation_network import SegmentationNetwork
from SegmentationNetwork.fc_drive import FCD

class SegmentationDrive(nn.Module):
    def __init__(self):
        """Creates a network that can drive based on camera imagery.

        This network drives by first segmenting the camera imagery then using
        fully connected layers to determine steering output.
        """
        super(SegmentationDrive, self).__init__()
        self.seg_net = SegmentationNetwork("googlenet")
        self.fcd = FCD()

    def forward(self, img, command):
        # Segment
        x = self.seg_net(img)

        # Flatten
        x = x.view(-1, self._num_flat_features(x))

        # Analyze for steering
        x = self.fcd(x, command)

        return x

    @staticmethod
    def _num_flat_features(x):
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