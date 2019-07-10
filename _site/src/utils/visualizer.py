# -*- coding: utf-8 -*-
"""Visualizer.

This module is used to visualize the training of the network.

Authors:
    Maximilian Roth
    Nina Pant
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import tkinter as tk


class Visualizer:
    def __init__(self, network, csv_dir, training_path, epoch, batch_size):
        """Initializes a visualizer class that visualizes network training.

        Args:
            network (Runner): A network runner that wraps the PyTorch network.
            csv_dir (string): The path of the csv file containing the data.
            training_path (string): The path of the training data.
            epoch (int): Number of epochs to run.
            batch_size (int): Size of the batch to run.
        """
        # Initialize network and its parameters
        self.network = network
        self.csv_dir = csv_dir
        self.training_path = training_path
        self.epoch = epoch
        self.batch_size = batch_size

    def execute(self):
        """Executes the training with a tkinter based visualizer."""
        # Start by making the tkinter parts
        root = tk.Tk()
        root.title("DriveNet Visualizer")

        # Set up tk variables
        step = tk.StringVar(master=root, value="Step: {0}/{1}"
                            .format(0, total_step))
        batch = tk.StringVar(master=root, value="Batch: {0}/{1}"
                             .format())

        # Configure the grid
        # ________________________
        # |         image        |
        # |                      |
        # |----------------------|
        # |   step    |   batch  |
        # |----------------------|
        # |   epoch   |   loss   |
        # ------------------------
        root.grid_columnconfigure(0, minsize=160)
        root.grid_columnconfigure(1, minsize=160)
        root.grid_rowconfigure(0, minsize=60)

        # Set text

        self.network.train_model(self.csv_dir, self.training_path,
                                 self.epoch, self.batch_size)
