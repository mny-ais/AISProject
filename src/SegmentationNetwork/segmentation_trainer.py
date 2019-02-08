# -*- coding: utf-8 -*-
"""Segmentation Trainer.

This module is designed to train the segmentation network.

Author:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""

# Create a tk window with the info and one of the images and outputs for each
# batch
# Import dataset
# enumerate it using a dataloader
# run through it, and run the seg network every step of every epoch
# Show only the first image
# calculate loss based on ground truth data
# optimize and repeat

from SegmentationNetwork.segmentation_network import SegmentationNetwork
import torch
import torch.nn as nn
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
import argparse

from torch.utils.data.dataloader import DataLoader
from SegmentationNetwork.segmented_dataset import SegmentedDataset
from utils.timer import Timer
from os import path
from plot import PlotIt
from time import strftime, gmtime

class Trainer:
    def _init__(self, save_dir):
        """This class is used to train the segmentation network.

        Args:
            save_dir (string): The directory to save the model parameters to.
        """
        self.network = SegmentationNetwork()
        self.network.cuda()
        self.network.train()

        self.device = torch.device('cuda')

        self.criterion = nn.CrossEntropyLoss()  # Recommended for pixel-wise
        self.optimizer = torch.optim.Adam(self.network.parameters())

        self.save_dir = save_dir

        torch.backends.cudnn.benchmark = True

    def train_segmentation(self, csv_file, root_dir, num_epochs, batch_size):
        """Trains the model.

        Args:
            csv_file (string): File address of the csv_file for training.
            root_dir (string): Root directory of the data for training.
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Number of objects in each batch.
        """
        # Start by making Tk parts
        root = tk.Tk()
        root.title("DriveNet Training")
        root.geometry("350x258")

        # Timers
        # Create timer and counter to calculate processing rate
        timer = Timer()
        counter = 0

        # Plot save location
        plot_loc = path.join(path.split(self.save_dir)[0],
                             strftime("%Y_%m_%d_%H-%M-%S", gmtime())
                             + '-loss_data.csv')

        # Configure the grid and geometry
        # -----------------------
        # | step     | rate     |
        # | epoch    | loss     |
        # | status message      |
        # -----------------------
        root.columnconfigure(0, minsize=175)
        root.columnconfigure(1, minsize=175)
        root.rowconfigure(0, minsize=64)
        root.rowconfigure(1, minsize=64)

        # Prepare tk variables with default values
        step_var = tk.StringVar(master=root, value="Step: 0/0")
        rate_var = tk.StringVar(master=root, value="Rate: 0 steps/s")
        epoch_var = tk.StringVar(master=root, value="Epoch: 1/{}"
                                 .format(num_epochs))
        loss_var = tk.StringVar(master=root, value="Loss: 0")
        time_var = tk.StringVar(master=root, value="Time left: 0:00")
        status = tk.StringVar(master=root, value="Preparing dataset")

        # Prepare tk labels to be put on the grid
        target_label = tk.Label(root, image=None)
        target_label.grid(row=0, column=0, columnspan=2)
        seg_label = tk.Label(root, image=None)
        seg_label.grid(row=1, column=1, columnspan=2)

        tk.Label(root, textvariable=step_var).grid(row=2, column=0, sticky="W",
                                                   padx=5, pady=5)
        tk.Label(root, textvariable=rate_var).grid(row=2, column=1,
                                                   sticky="W", padx=5,
                                                   pady=5)
        tk.Label(root, textvariable=epoch_var).grid(row=3, column=0,
                                                    sticky="W", padx=5, pady=5)
        tk.Label(root, textvariable=loss_var).grid(row=3, column=1,
                                                   sticky="W", padx=5, pady=5)
        tk.Label(root, textvariable=time_var).grid(row=4, column=0,
                                                   sticky="W", padx=5, pady=5)
        tk.Label(root, textvariable=status).grid(row=5, column=0, columnspan=2,
                                                 sticky="SW", padx=5,
                                                 pady=5)

        # Update root so it actually shows something
        root.update_idletasks()
        root.update()

        # Open file for loss data plot
        loss_file = open(plot_loc, 'a')

        # Prepare the datasets and their corresponding dataloaders
        data = SegmentedDataset(csv_file, root_dir)
        train_loader = DataLoader(dataset=data,
                                  batch_size=batch_size,
                                  shuffle=True)
        status.set("Data sets loaded")

        root.update_idletasks()
        root.update()

        # total_step = 0
        status.set("Training")
        root.update_idletasks()
        root.update()

        if path.isfile(self.save_dir):
            self.network.load_state_dict(torch.load(self.save_dir))

        for epoch in range(num_epochs):
            status.set("Training: all")
            root.update_idletasks()
            root.update()

            #  Former different loaders are now one single loader for all data
            total_step = len(train_loader)

            for data in enumerate(train_loader):
                # run the forward pass
                # data[0] is the iteration, data[1] is the data
                #  print(images)
                image = data[1]['image']
                seg = data[1]["seg"]

                # Prep target by turning it into a CUDA compatible format
                seg_cuda = seg.to(self.device, non_blocking=True)

                # self.optimizer.zero_grad()
                image = torch.unsqueeze(image, 0)

                out = self.network(image.to(self.device, non_blocking=True),
                                   batch_size)

                # Show the result



                # calculate the loss
                if self.out is None:
                    raise ValueError("forward() has not been run properly.")
                loss = self.criterion(out, seg_cuda)

                # Zero grad
                self.optimizer.zero_grad()
                # Backprop and preform Adam optimization
                loss.backward()
                self.optimizer.step()

                # # Track the accuracy{{{
                # total = data.size(0)
                # _, predicted = torch.max(self.output.data, 1)
                # correct = (predicted == data[0]).sum().item()
                # acc_list.append(correct / total)

                # Update data
                step_var.set("Step: {0}/{1}".format(data[0] + 1, total_step))
                epoch_var.set("Epoch: {0}/{1}".format(epoch + 1, num_epochs))
                loss_var.set("Loss: {:.3f}".format(loss.item()))# }}}

                counter += 1
                if timer.elapsed_seconds_since_lap() > 0.3:# {{{
                    sps = float(counter) / timer.elapsed_seconds_since_lap()
                    rate_var.set("Rate: {:.2f} steps/s".format(sps))
                    timer.lap()
                    counter = 0

                    if sps == 0:
                        time_left = "NaN"
                    else:
                        time_left = int(((total_step * num_epochs)
                                         - ((float(data[0]) + 1.0)
                                         + (total_step * epoch))) / sps)
                        time_left = datetime.timedelta(seconds=time_left)
                        time_left = str(time_left)
                    time_var.set("Time left: {}".format(time_left))

                # update image only once per step
                if data[0] == 0:
                    target = F.to_pil_image(seg)
                    seg_img = F.to_pil_image(out.cpu().detach())
                    target_label.configure(image=target)
                    seg_label.configure(image=seg_img)

                root.update()
                root.update_idletasks()# }}}

                loss_file.write("{}\n".format(loss.item()))

        # Now save the loss file and the weights
        loss_file.close()

        # Save the bias and weights
        torch.save(self.network.state_dict(),
                   self.save_dir)

        torch.cuda.empty_cache()

        status.set("Done")
        PlotIt(plot_loc)
        root.mainloop()


def parse_args():
    """Parses arguments from terminal."""
    description = "Trains DriveNet on a dataset or runs it in AirSim."
    parser = argparse.ArgumentParser(description=description)

    # Required:
    parser.add_argument('weights', metavar='W', type=str, nargs=1,
                        help="file that has the weights, or where the weights"
                             "should be stored.")

    # Training and training arguments
    parser.add_argument('data', metavar='D', type=str, nargs=1,
                        help="the data directory.")
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=20,
                        help="batch size of the training.")
    parser.add_argument("-p", "--epoch", type=int, nargs='?', default=1,
                        help="number of epochs to train for.")

    return parser.parse_args()

if __name__ == "__main__":
    arguments = parse_args()
    trainer = Trainer()

    csv_loc = path.join(arguments.data, "control_input.csv")
    trainer.train_segmentation(csv_loc, arguments.data, arguments.batch_size,
                               arguments.epoch)