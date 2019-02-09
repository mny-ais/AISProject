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
import tkinter as tk
from tkinter.filedialog import askopenfilename


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
            root = tk.Tk()
            root.withdraw()
            file_name = askopenfilename()
            root.destroy()
        else:
            file_name = plot_location
        
        data = open(file_name, "r").read()

        x_axis = []
        print("yay im here")
        data = data.splitlines()
        for i in range(0, len(data)):
            data[i] = float(data[i])
            x_axis.append(i)
        plt.bar(x_axis, data, 1.0, color="blue")
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.show(block=True)


if __name__ == "__main__":
        print("hi im nina")
        arguments = parse_args()
        PlotIt(arguments.path)
