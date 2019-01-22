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
import tkinter as tk
from torch.utils.data.dataloader import DataLoader  # Using this to load data
from data_loader import DrivingSimDataset
from network import DriveNet
from utils.timer import Timer
from os import path
from plot import PlotIt
from time import strftime, gmtime


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
            self.network.cuda()

        # Save the last output, so we can calculate the loss using it
        self.out = None

        self.criterion = nn.MSELoss()  # So this is only initialized once

        # We use the Adam optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0002)

        # Weight file location and name
        self.save_dir = save_dir

    def train_model(self, csv_file, root_dir, num_epochs, batch_size,
                    silent=False):
        """Trains the model.

        This method also visualizes the training of the network. It is able to
        show the progress in terms of step, batch, and epoch number, loss, and
        the plot from the batch. It also outputs a plot of the loss over
        time.

        Args:
            csv_file (string): File address of the csv_file for training.
            root_dir (string): Root directory of the data for training.
            num_epochs (int): Number of epochs to train for
            batch_size (int): Number of objects in each batch
            silent (bool): Whether to show the plot or not. True hides the plot.
        """
        # Start by making the tkinter parts
        root = tk.Tk()
        root.title("DriveNet Training")
        root.geometry("350x100")

        # Create timer and counter to calculate processing rate
        timer = Timer()
        counter = 0

        # Plot save location
        plot_loc = path.join(path.split(self.save_dir)[0],
                             strftime("%Y_%m_%d_%H-%M-%S", gmtime())
                             + '-loss_data.txt')

        # Configure the grid and geometry
        # -----------------------
        # | step     | rate     |
        # | epoch    | loss     |
        # | status message      |
        # -----------------------
        root.columnconfigure(0, minsize=175)
        root.columnconfigure(1, minsize=175)

        # Prepare tk variables with default values
        step_var = tk.StringVar(master=root, value="Step: 0/0")
        rate_var = tk.StringVar(master=root, value="Rate: 0 steps/s")
        epoch_var = tk.StringVar(master=root, value="Epoch: 1/{}"
                                 .format(num_epochs))
        loss_var = tk.StringVar(master=root, value="Loss: 0")
        status = tk.StringVar(master=root, value="Preparing dataset")

        # Prepare tk labels to be put on the grid
        tk.Label(root, textvariable=step_var).grid(row=0, column=0, sticky="W",
                                                   padx=5, pady=5)
        tk.Label(root, textvariable=rate_var).grid(row=0, column=1,
                                                   sticky="W", padx=5,
                                                   pady=5)
        tk.Label(root, textvariable=epoch_var).grid(row=1, column=0,
                                                    sticky="W", padx=5, pady=5)
        tk.Label(root, textvariable=loss_var).grid(row=1, column=1,
                                                   sticky="W", padx=5, pady=5)
        tk.Label(root, textvariable=status).grid(row=2, column=0, columnspan=2,
                                                 sticky="SW", padx=5, pady=5)

        # Update root so it actually shows something
        root.update_idletasks()
        root.update()

        # Prepare the dataset
        training_data = DrivingSimDataset(csv_file, root_dir)

        # Prepare the dataloader
        train_loader = DataLoader(dataset=training_data,
                                  batch_size=batch_size,
                                  shuffle=True)

        total_step = len(train_loader)
        step_var.set("Step: 0/{}".format(total_step))
        status.set("Training")

        if not path.isfile(self.save_dir):
            status.set("No state dictionary found. Will run with randomized "
                       "weights.")

        # print("len(train_loader) = {0}".format(total_step))
        # print("Dataset:")
        # for i in enumerate(train_loader):
        #     print(i[1]['image'])
        #     print(i[1]['vehicle_commands'])

        # acc_list = []

        for epoch in range(num_epochs):
            for data in enumerate(train_loader):
                # run the forward pass
                # data[0] is the iteration, data[1] is the data
                images = data[1]['image']
                vehicle_commands = data[1]['vehicle_commands']
                command = data[1]['cmd'].numpy()


                # Prep target by turning it into a CUDA compatible format
                target = vehicle_commands
                target = target.to(self.device, dtype=torch.float)


                self.optimizer.zero_grad()

                self.run_model(images.to(self.device, dtype=torch.float),
                              command,
                              batch_size,
                              eval_model0=False)

                # calculate the loss
                if self.out is None:
                    raise ValueError("forward() has not been run properlz.")
                loss = self.criterion(self.out, target)

                # Backdrop and preform Adam optimization
                loss.backward()
                self.optimizer.step()

                # # Track the accuracy
                # total = data.size(0)
                # _, predicted = torch.max(self.output.data, 1)
                # correct = (predicted == data[0]).sum().item()
                # acc_list.append(correct / total)

                # Update data

                step_var.set("Step: {0}/{1}".format(data[0] + 1, total_step))
                epoch_var.set("Epoch: {0}/{1}".format(epoch + 1, num_epochs))
                loss_var.set("Loss: {:.3f}".format(loss.item()))

                counter += 1
                if timer.elapsed_seconds_since_lap() > 0.3:
                    sps = float(counter) / timer.elapsed_seconds_since_lap()
                    rate_var.set("Rate: {:.0f} steps/s".format(sps))
                    timer.lap()
                    counter = 0

                root.update()
                root.update_idletasks()

                if (data[0] + 1) % 20 == 0:
                    with open(plot_loc, 'a') \
                            as file:
                        file.write("{}\n".format(loss.item()))

        # Now save the file
        torch.save(self.network.state_dict(),
                   self.save_dir)
        torch.cuda.empty_cache()

        status.set("Done")
        PlotIt(plot_loc)


    def run_model(self, input_image, input_command, batch_size, eval_mode=True):
        """Runs the model forward.

        Args:
            input_image (torch.Tensor): The input image as a tensor.
            input_command (numpy.array): The input command as an integer value
                                         in a tensor.
                                         -1 is left,
                                         0 is center,
                                         1 is right
            batch_size (int): The size of the batch to run.
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

        self.network.to(self.device)
        self.out = self.network(input_image, input_command, batch_size)
        return self.out

# Test code to see if the model runs
# if __name__ == "__main__":
#     net = DriveNet()
#     # print(net)
#
#     # This is to test if it works
#     input = torch.randn(1, 3, 200, 88)
#     out = net(input, 0)
#     out= out.detach().numpy()
#     print(out)
#     plotit = PlotIt('plotdata.txt')


# net.zero_grad()
# out.backward(torch.randn(1, 2))
