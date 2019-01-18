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

from data_loader import DrivingSimDataset

from network import DriveNet

from os import path


class Runner:
    def __init__(self, save_dir):
        """This class is used to train and run a model.

        Args:
            save_dir (string): The directory to save the model parameters to.
        """
        self.network = DriveNet()
        self.device = torch.device("cuda")  # Set the device to a CUDA device

        # Save the last output, so we can calculate the loss using it
        self.out = None

        # Save the loss, so we can use it to backpropagate
        self.loss = None
        self.criterion = nn.MSELoss()  # So this is only initialized once

        # We use the Adam optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0002)

        # Weight file location and name
        self.save_dir = save_dir

    def train_model(self, csv_file, root_dir, num_epochs, batch_size):
        """Trains the model.

        Args:
            csv_file (string): File address of the csv_file for training.
            root_dir (string): Root directory of the data for training.
            num_epochs (int): Number of epochs to train for
            batch_size (int): Number of objects in each batch
        """
        # Datasets
        training_data = DrivingSimDataset(csv_file, root_dir)

        # TODO Add a way to create the packaged datasets

        # Dataloaders
        train_loader = DataLoader(dataset=training_data,
                                  batch_size=batch_size,
                                  shuffle=True)

        total_step = len(train_loader)
        loss_list = []
        acc_list = []

        for epoch in range(num_epochs):
            for i, (images, data) in enumerate(train_loader):
                # run the forward pass
                # data[0] is the steering info, data[1] is the drive command
                print("running forwards")
                print("images:")
                print(images)
                print("data:")
                print(data)
                self.run_model(images[i].to(self.device, dtype=torch.float),
                               data.data.numpy()[i][-1], eval_mode=False)
                target = data[i][1:3].to(self.device, dtype=torch.float)
                loss = self.__calculate_loss(target)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track the accuracy
                total = data.size(0)
                _, predicted = torch.max(self.output.data, 1)
                correct = (predicted == data[0]).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step[{}/{}], Loss: {:4f}, Accuracy"
                          .format(epoch + 1, num_epochs, i + 1, total_step,
                                  loss.item())
                          + ": {:2f}%".format(correct / total) * 100)

        # Now save the file
        torch.save(self.network.state_dict(),
                   self.save_dir)

    def __calculate_loss(self, target):
        """Calculates the loss based on a target tensor.

        Args:
            target (torch.Tensor): The (1 x 3) ground truth tensor,

        Returns:
            The loss criterion. Also saves this internally.
        """
        if self.out is None:
            raise ValueError("forward() has not been run.")
        print("running loss calculation")
        print("out:")
        print(self.out)
        print("target:")
        print(target)
        self.loss = self.criterion(self.out, target)

        return self.loss  # Return the loss, in case it is necessary

    def run_model(self, input_image, input_command, eval_mode=True):
        """Runs the model forward.

        Args:
            input_image (torch.Tensor): The input image as a tensor.
            input_command (int): The input command as an integer value.
                                 -1 is left,
                                 0 is center,
                                 1 is right
            eval_mode (bool): Sets whether the model should be in evaluation
                              mode.

        Returns (torch.Tensor):
            The output as a 2 channel tensor representing steering and throttle.
        """
        if eval_mode:
            self.network.eval()
        else:
            self.network.train()

        input_image = input_image.to(self.device)

        # load the model parameters from file, if it exists.
        if path.isfile(self.save_dir):
            self.network.load_state_dict(torch.load(self.save_dir))
        else:
            print("No state dictionary found. Will run with randomized input.")
            # Assumption: Network starts with random when nothing is found.

        self.network.to(self.device)
        self.out = self.network(input_image, input_command)
        return self.out


# Test code to see if the model runs
if __name__ == "__main__":
    net = DriveNet()
    # print(net)

    # This is to test if it works
    input = torch.randn(1, 3, 200, 88)
    out = net(input, 0)
    out= out.detach().numpy()
    print(out)

    # net.zero_grad()
    # out.backward(torch.randn(1, 2))

