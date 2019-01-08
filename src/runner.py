# -*- coding: utf-8 -*-
"""Runner.

This module is used to run and train the model described in network.py.

Authors:
    Maximilian Roth
    Nina Pant
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>

"""

import torch

import torch.nn as nn

from network import DriveNet

class Trainer:
    def __init__(self, network):
        """This class is used to train a model.

        Args:
            network (nn.Module): The network that is to be trained.
        """
        self.network = network
        # Save the last output, so we can calculate the loss using it
        self.out = None

        # Save the loss, so we can use it to backpropagate
        self.loss = None
        self.criterion = nn.MSELoss()  # So this is only initialized once

        # We use the Adam optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0002)

    def run_model(self, input_image, input_command):
        """Runs the model forward.

        Args:
            input_image (torch.Tensor): The input image as a tensor.
            input_command (int): The input command as an integer value.
                                 -1 is left,
                                 0 is center,
                                 1 is right

        Returns (torch.Tensor):
            The output as a 2 channel tensor representing steering and throttle.
        """
        self.network(input)
    def calculate_loss(self, target):
        """Calculates the loss based on a target tensor.

        Args:
            target (torch.Tensor): The (1 x 3) ground truth tensor,

        Returns:
            The loss criterion. Also saves this internally.
        """
        if self.out is None:
            raise ValueError("forward() has not been run.")

        self.loss = self.criterion(self.out, target)

        return self.loss  # Return the loss, in case it is necessary


if __name__ == "__main__":
    net = DriveNet()
    #print(net)

    # For non training only
    # net.eval()

    input = torch.randn(1, 3, 200, 88)
    out = net(input, 0)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 3))
