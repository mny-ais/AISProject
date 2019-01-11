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

from torch.utils.data.dataloader import DataLoader  # Using this to load data

from network import DriveNet

from data_loader import DrivingSimDataset


class Trainer:
    def __init__(self, network, csv_file, root_dir, batch_size=120):
        """This class is used to train a model.

        Args:
            network (nn.Module): The network that is to be trained.
            csv_file (string): File address of the csv_file for training.
            root_dir (string): Root directory of the data for training.
            batch_size (int): Size of the batches for processing
        """
        self.network = network
        # Save the last output, so we can calculate the loss using it
        self.out = None

        # Save the loss, so we can use it to backpropagate
        self.loss = None
        self.criterion = nn.MSELoss()  # So this is only initialized once

        # We use the Adam optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0002)

        # Datasets
        self.training_data = DrivingSimDataset(csv_file, root_dir)

        # TODO Add a way to create the packaged datasets

        # Dataloaders
        self.train_loader = DataLoader(dataset=self.training_data,
                                       batch_size=batch_size,
                                       shuffle=True)

    def train_model(self, num_epochs):
        """Trains the model.
        Args:
            num_epochs (int): Number of epochs to train for
        """
        total_step = len(self.train_loader)
        loss_list = []
        acc_list = []

        for epoch in range(num_epochs):
            for i, (images, data) in enumerate(self.train_loader):
                # run the forward pass
                # data[0] is the steering info, data[1] is the drive command
                self.__run_model(images, data[1])
                loss = self.__calculate_loss(data[0])
                loss_list.append(loss.item())

                # Backprop and perform Adam optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track the accuracy
                total = data[0].size(0)
                _, predicted = torch.max(self.output.data, 1)
                correct = (predicted == data[0]).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step[{}/{}], Loss: {:4f}, Accuracy"
                          .format(epoch + 1, num_epochs, i + 1, total_step,
                                  loss.item())
                          + ": {:2f}%".format(correct / total) * 100)

    def test_model(self):
        """Connects the model with the AirSim API to drive the car."""
        # TODO connect the model with the AirSim API
        pass

    def __calculate_loss(self, target):
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

    def __run_model(self, input_image, input_command):
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
        self.out = self.network(input_image, input_command)
        return self.out


if __name__ == "__main__":
    net = DriveNet()
    # print(net)

    # This is to test if it works
    input = torch.randn(1, 3, 200, 88)
    out = net(input, 0)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 2))