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
from SegmentationNetwork.fc_drive import FCD
from SegmentationNetwork.segmentation_drive import SegmentationDrive
import datetime
import torch
import torch.nn as nn
import tkinter as tk
import argparse
import numpy as np
from skimage import io

from torch.utils.data.dataloader import DataLoader
from SegmentationNetwork.segmented_dataset import SegmentedDataset
from utils.timer import Timer
from os import path
from plot import PlotIt
from time import strftime, gmtime
from PIL import ImageTk, Image

ENCODER = "googlenet"

class Trainer:
    def __init__(self, save_dir):
        """This class is used to train the segmentation network.

        Args:
            save_dir (string): The directory to save the model parameters to.
        """
        self.network = None
        self.device = torch.device('cuda')

        self.criterion = None
        self.optimizer = None

        self.weights_save_dir = save_dir

        torch.backends.cudnn.benchmark = True

    def load_network(self, network):
        """Loads the network that will be trained.

        Args:
            network (str): Either "segmentation" or "full".
        """
        if network == "segmentation":
            self.network = SegmentationNetwork(ENCODER)
            self.criterion = nn.CrossEntropyLoss()  # Recommended for pixel-wise
            self.optimizer = torch.optim.Adam(self.network.parameters())
            self.network.train()
        elif network == "full":
            self.network = SegmentationDrive()
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.network.fcd.parameters(),
                                             lr=0.002)
            self.network.seg_net.eval()
            self.network.fcd.train()
        self.network.cuda()


    def train_segmentation(self, image_dir, batch_size, num_epochs):
        """Trains the model.

        Args:
            image_dir (string): Root directory of the data for training.
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Number of objects in each batch.
        """
        self.load_network("segmentation")
        # Start by making Tk parts
        root = tk.Tk()
        root.title("DriveNet Training")
        root.geometry("350x258")

        # Timers
        # Create timer and counter to calculate processing rate
        timer = Timer()
        counter = 0

        # Plot save location
        plot_loc = path.dirname(path.dirname(path.abspath(__file__)))
        plot_loc = path.join(plot_loc, "plot_csv")
        plot_loc = path.join(plot_loc, strftime("%Y_%m_%d_%H-%M-%S", gmtime())
                             + '-loss_data.csv')

        # Configure the grid and geometry
        # -----------------------
        # | Target segmentation |
        # | Output              |
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

        # Prepare image canvases
        black_array = io.imread(path.join(image_dir, "seg_00001-cam_0.png"))
        black_array.fill(0)
        black_image = ImageTk.PhotoImage(image=Image.fromarray(black_array))

        target_canvas = tk.Canvas(root, width=320, height=64)
        target_canvas_img = target_canvas.create_image(0, 0, anchor="nw", image=black_image)
        target_canvas.grid(row=0, column=0, columnspan=2)
        seg_canvas = tk.Canvas(root, width=320, height=64)
        seg_canvas_img = seg_canvas.create_image(0, 0, anchor="nw", image=black_image)
        seg_canvas.grid(row=1, column=0, columnspan=2)

        # Prepare tk labels to be put on the grid
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
        data = SegmentedDataset(image_dir)
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

        if path.isfile(self.weights_save_dir):
            self.network.load_state_dict(torch.load(self.weights_save_dir))

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
                seg_cuda = data[1]["seg"].to(device=self.device,
                                             dtype=torch.long,
                                             non_blocking=True)
                seg = seg.numpy()

                out = self.network(image.to(self.device, non_blocking=True))

                # calculate the loss
                if out is None:
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
                loss_var.set("Loss: {:.3f}".format(loss.item()))

                counter += 1
                if timer.elapsed_seconds_since_lap() > 0.3:
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
                if data[0] % 2 == 0:
                    target_array = self.class_to_image_array(seg)
                    target = ImageTk.PhotoImage(image=Image
                                                .fromarray(target_array))
                    seg_array = self.class_prob_to_image_array(out.cpu()
                                                               .detach()[0])
                    seg_img = ImageTk.PhotoImage(image=Image
                                                 .fromarray(seg_array))
                    target_canvas.itemconfig(target_canvas_img, image=target)
                    seg_canvas.itemconfig(seg_canvas_img, image=seg_img)

                root.update()
                root.update_idletasks()

                loss_file.write("{}\n".format(loss.item()))

        # Now save the loss file and the weights
        loss_file.close()

        # Save the bias and weights
        torch.save(self.network.state_dict(),
                   self.weights_save_dir)

        torch.cuda.empty_cache()

        status.set("Done")
        PlotIt(plot_loc)
        root.mainloop()

    def train_fc(self, image_dir, batch_size, num_epochs, fc_weights):
        """Trains the fully connected layers using a pretrained segmentation.
        """
        self.load_network("full")
        # Start by making Tk parts
        root = tk.Tk()
        root.title("DriveNet Training")
        root.geometry("350x258")

        # Timers
        # Create timer and counter to calculate processing rate
        timer = Timer()
        counter = 0

        # Plot save location
        plot_loc = path.dirname(path.dirname(path.abspath(__file__)))
        plot_loc = path.join(plot_loc, "plot_csv")
        plot_loc = path.join(plot_loc, strftime("%Y_%m_%d_%H-%M-%S", gmtime())
                             + '-loss_data.csv')

        # Configure the grid and geometry
        # -----------------------
        # | Target segmentation |
        # | Output              |
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
        time_var = tk.StringVar(master=root, value="Time left: 0:00")
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
        tk.Label(root, textvariable=time_var).grid(row=2, column=0,
                                                   sticky="W", padx=5, pady=5)
        tk.Label(root, textvariable=status).grid(row=3, column=0, columnspan=2,
                                                 sticky="SW", padx=5,
                                                 pady=5)

        # Update root so it actually shows something
        root.update_idletasks()
        root.update()

        # Open file for loss data plot
        loss_file = open(plot_loc, 'a')

        # Prepare the datasets and their corresponding dataloaders
        data = SegmentedDataset(image_dir)
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

        if path.isfile(self.weights_save_dir):
            self.network.seg_net.load_state_dict(torch.load(
                self.weights_save_dir))

        if path.isfile(fc_weights):
            self.network.fcd.load_state_dict(torch.load(fc_weights))

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
                cmd = data[1]['cmd']

                # Prep target by turning it into a CUDA compatible format
                steering = data[1]["vehicle_commands"].to(self.device,
                                                          non_blocking=True)

                out = self.network(image.to(self.device, non_blocking=True),
                                   cmd)

                # calculate the loss
                if out is None:
                    raise ValueError("forward() has not been run properly.")
                loss = self.criterion(out, steering)

                # Zero grad
                self.optimizer.zero_grad()
                # Backprop and preform optimization
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
                loss_var.set("Loss: {:.3f}".format(loss.item()))

                counter += 1
                if timer.elapsed_seconds_since_lap() > 0.3:
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
                root.update()
                root.update_idletasks()

                loss_file.write("{}\n".format(loss.item()))

        # Now save the loss file and the weights
        loss_file.close()

        # Save the bias and weights
        torch.save(self.network.state_dict(),
                   self.weights_save_dir)

        torch.cuda.empty_cache()

        status.set("Done")
        PlotIt(plot_loc)
        root.mainloop()


    @staticmethod
    def class_to_image_array(input):
        """Transforms classes into an image array.

        Args:
            input (np.array): numpy array of classes in each pixel in the shape
                              [Batch, H, W]

        Returns:
            (np.array) in the form [H, W, Color]
        """
        input = input.transpose(1, 2, 0)
        out = np.ndarray([64, 320, 3])
        for i in range(64):
            for j in range(320):
                if input[i][j][0] == 0:
                    out[i][j] = np.array([0, 0, 0])
                elif input[i][j][0] == 1:
                    out[i][j] = np.array([0, 0, 255])
                else:
                    out[i][j] = np.array([255, 0, 0])
        return out.astype('uint8')

    @staticmethod
    def class_prob_to_image_array(input):
        """Takes class probabilities and turns it into an image array.

        Args:
            input (torch.Tensor): Input with shape [2, 64, 320]

        Returns:
            (np.array) with shape [64, 320, 3]
        """
        input_array = input.numpy()
        input_array = input_array.transpose(1, 2, 0)

        output_array = np.ndarray([64, 320, 3])
        for i in range(64):
            for j in range(320):
                index_max = np.argmax(input_array[i][j])
                if index_max == 0:
                    output_array[i][j].fill(0)
                elif index_max == 1:
                    output_array[i][j] = np.array([0, 0, 255])
                else:
                    output_array[i][j] = np.array([255, 0, 0])

        return output_array.astype('uint8')


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
    parser.add_argument('-f', '--fully-connected', type=str, nargs='?',
                        help='train fully connected layers and specify the '
                             'weights for it')
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=20,
                        help="batch size of the training.")
    parser.add_argument("-p", "--epoch", type=int, nargs='?', default=1,
                        help="number of epochs to train for.")

    return parser.parse_args()


def main(weights, data, batch_size, epochs, fc_weights=None):
    """Main function to run everything.

    Args:
        weights (str): Path to the weights.
        data (str): Path to the image data.
        batch_size (int): Size of the batches to run.
        epochs (int): Number of epochs to run
        fc_weights (Union[str, None]): if not None, then where the fc_weights
        are.
    """
    trainer = Trainer(weights)
    if fc_weights is None:
        trainer.train_segmentation(data, batch_size, epochs)
    else:
        trainer.train_fc(data, batch_size, epochs, fc_weights)



if __name__ == "__main__":
    arguments = parse_args()

    if arguments.fully_connected:
        main(arguments.weights[0], arguments.data[0], arguments.batch_size,
             arguments.epoch, arguments.fully_connected)
    else:
        main(arguments.weights[0], arguments.data[0], arguments.batch_size,
             arguments.epoch)
