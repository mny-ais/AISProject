# -*- coding: utf-8 -*-
"""PlotIt.

This module visualizes the plot of the loss function created while training the
network.

Authors:
    Maximilian Roth
    Nina Pant
"""
import argparse
import matplotlib.pyplot as plt


def parse_args():
    """Parses command line arguments."""
    description = "Plots loss data from DriveNet"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('path', metavar='P', type=str, nargs='?',
                        help='path of the loss data to be plotted.')
    return parser.parse_args()

class PlotIt:
    """Generates plot of loss function from .txt file."""
    def __init__(self, plot_location=None):
        if plot_location == None:
            file_name = "/home/aisgrp3/Documents/src_ln/plotdata.txt"
        else:
            file_name = plot_location
        
        with open(file_name, "r") as file:
            data = file.read()

        open(file_name, "w").close()
        x_axis = []
        print("yay im here")
        data = data.splitlines()
        for i in range(0, len(data)):
            data[i] = float(data[i])
            x_axis.append(i)
        plt.bar(x_axis, data, 1/1.5, color="blue")
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.show(block=True)


if __name__ == "__main__":
        print("hi im nina")
        arguments = parse_args()
        PlotIt(arguments.path)
