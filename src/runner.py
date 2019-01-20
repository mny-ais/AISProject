# -*- coding: utf-8 -*-
"""Runner.

This module is used to run and train the model described in network.py.

The train_model method also visualizes the data being trained. It first creates
a GUI window that shows a sample from the current batch being trained along
with information about the current progress of the training. When it is done,
it outputs a plot of the loss over time.

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

from plot import PlotIt


class Runner:
    def __init__(self, save_dir, cpu=True):
        """This class is used to train and run a model.

        Args:
            save_dir (string): The directory to save the model parameters to.
            cpu (bool): Use cpu or CUDA. True means use the CPU only
        """
        self.network = DriveNet()
        if cpu:
            self.device = torch.device("cpu")  # Set it to be a CPU only device
        else:
            self.device = torch.device('cuda')  # Set the device to CUDA


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

        This method also visualizes the training of the network. It is able to
        show the progress in terms of step, batch, and epoch number, loss, and
        the first image from the batch. It also outputs a plot of the loss over
        time after training is finished.

        Args:
            csv_file (string): File address of the csv_file for training.
            root_dir (string): Root directory of the data for training.
            num_epochs (int): Number of epochs to train for
            batch_size (int): Number of objects in each batch
        """
        # Start by making the tkinter parts
        # root = tk.Tk()
        #  root.title("DriveNet Training")

        # Configure the grid
        # ________________________
        # |         image        |
        # |                      |
        # |----------------------|
        # |   step   |  progress |
        # |----------------------|
        # |   epoch   |   loss   |
        # |----------------------|
        # | status message       |
        # ------------------------
        # root.grid_columnconfigure(0, minsize=160)
        # root.grid_columnconfigure(1, minsize=160)
        # root.grid_rowconfigure(0, minsize=60)

        # Prepare the dataset
        training_data = DrivingSimDataset(csv_file, root_dir)

        # Prepare the dataloader
        train_loader = DataLoader(dataset=training_data,
                                  batch_size=batch_size,
                                  shuffle=True)

        total_step = len(train_loader)
        # print("len(train_loader) = {0}".format(total_step))
        # print("Dataset:")
        # for i in enumerate(train_loader):
        #     print(i[1]['image'])
        #     print(i[1]['vehicle_commands'])

        acc_list = []

        for epoch in range(num_epochs):
            for data in enumerate(train_loader):
                # run the forward pass
                # data[0] is the iteration, data[1] is the data
                				
                print(torch.cuda.memory_cached())

                images = data[1]['image']
                vehicle_commands = data[1]['vehicle_commands']
                command = data[1]['cmd'].numpy()


                self.run_model(images.to(self.device, dtype=torch.float),
                               command,
                               batch_size,
                               eval_mode=False)
                # Prep target by turning it into a CUDA compatible format
                target = vehicle_commands
                target = target.to(self.device, dtype=torch.float)

                # now calculate the loss
                loss = self.__calculate_loss(target)

                # Backprop and perform Adam optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # # Track the accuracy
                # total = data.size(0)
                # _, predicted = torch.max(self.output.data, 1)
                # correct = (predicted == data[0]).sum().item()
                # acc_list.append(correct / total)

                # TODO: Calculate accuracy


                lossy = loss.item()

                if (data[0] + 1) % 50 == 0:
                    print("Epoch [{}/{}], Step[{}/{}], Loss: {:4f}"
                          .format(epoch + 1, num_epochs, data[0] + 1, total_step,
                                  lossy))
                with open('plotdata.txt','a') as file:
                    file.write("{:4f}\n".format(lossy))
		

        # Now save the file
        torch.save(self.network.state_dict(),
                   self.save_dir)
        torch.cuda.empty_cache()

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

    def run_model(self, input_image, input_command, batch_size, eval_mode=True):
        """Runs the model forward.

        Args:
            input_image (torch.Tensor): The input image as a tensor.
            input_command (numpy.array): The input command as an integer value
                                         in a tensor.
                                         -1 is left,
                                         0 is center,
                                         1 is right
            eval_mode (bool): Sets whether the model should be in evaluation
                              mode.

        Returns:
            (torch.Tensor) The output as a 2 channel tensor representing
            steering and throttle.
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
        self.out = self.network(input_image, input_command, batch_size)
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
    plotit = PlotIt('plotdata.txt')
    

    # net.zero_grad()
    # out.backward(torch.randn(1, 2))

