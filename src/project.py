"""Project.

Provides a GUI interface to start the neural networks in this project.

Authors:
    Nina Pant
    Maximilian Roth <max@die-roths.de>
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import last_image_too
import nn_segmentation
import seg_and_normal
import segmented
import standard
import two_cams

def main():
    """Provides the main function that starts everything else."""
    # All the sub-functions first
    def weights_2_active_check(*args):
        """Does a check to see if weights_2 is active."""
        if branch_var.get() == "self segmenting":
            try:
                weights_2_label.config(fg="black")
                weight_2_button.config(state="active")
            except NameError:
                pass
        else:
            try:
                weights_2_label.config(fg="gray60")
                weight_2_button.config(state="disabled")
            except NameError:
                pass

    def weight_on_click(*args):
        """Chooses weights for weights_1"""
        weight_path = askopenfilename()
        if weight_path == "":
            weight_path_var.set("Choose File...")
        else:
            weight_path_var.set(os.path.split(weight_path)[1])

    def weight_2_on_click(*args):
        """Chooses weights for weights_2."""
        weight_2_path = askopenfilename()
        if weight_2_path == "":
            weight_2_path_var.set("Choose File...")
        else:
            weight_2_path_var.set(os.path.split(weight_2_path)[1])

    def data_on_click(*args):
        """Chooses data path."""
        data_path = askdirectory()
        if data_path == "":
            data_path_var.set("Choose Folder...")
        else:
            data_path_var.set(os.path.split(data_path)[1])


    def train_switch(*args):
        """Switches state of the right side of the screen."""
        if not train_bool.get():
            for label in train_labels:
                label.config(fg="black")
            try:
                for param in train_params:
                    param.config(state="normal")
                data_button.config(state="active")
            except NameError:
                pass
        else:
            for label in train_labels:
                label.config(fg="gray60")
            try:
                for param in train_params:
                    param.config(state="disabled")
                data_button.config(state="disabled")
            except NameError:
                pass

    def run():
        """Runs the actual neural network stuff and destroys this screen."""
        mode = branch_var.get()
        w1 = weight_path_var.get()
        w2 = weight_2_path_var.get()

        lr = lr_entry.get()
        root.destroy()


    # Create and configure root
    root = tk.Tk()
    root.title = "DriveNet"
    root.geometry = "420x300"

    root.columnconfigure(0, minsize=105)
    root.columnconfigure(1, minsize=150)
    root.columnconfigure(2, minsize=80)
    root.columnconfigure(3, minsize=105)

    # Paths
    weight_path = None
    weight_2_path = None
    data_path = None
    weight_path_var = tk.StringVar(root, value="Choose File...")
    weight_2_path_var = tk.StringVar(root, value="Choose File...")
    data_path_var = tk.StringVar(root, value="Choose Folder...")

    # Modes:
    branch_var = tk.StringVar(root, value="standard")
    branch_var.trace_add("read", weights_2_active_check)
    train_bool = tk.BooleanVar(root, value=True)
    train_bool.trace_add("read", train_switch)

    # Optimizer choice (0 = Adam, 1 = SGD)
    optimizer_var = tk.IntVar(root, value=0)

    # Labels
    branch_label = tk.Label(root, text="Mode")
    grid_pad(branch_label, 0, 0)

    weights_label = tk.Label(root, text="Weights file")
    grid_pad(weights_label, 1, 0)

    weights_2_label = tk.Label(root, text="Weights 2 file", fg="gray60")
    grid_pad(weights_2_label, 2, 0)

    # Labels specific to training
    train_labels = []

    data_label = tk.Label(root, text="Image Data")
    grid_pad(data_label, 3, 0)
    train_labels.append(data_label)

    lr_label = tk.Label(root, text="LR")
    grid_pad(lr_label, 1, 2)
    train_labels.append(lr_label)

    batch_size_label = tk.Label(root, text="Batch size")
    grid_pad(batch_size_label, 2, 2)
    train_labels.append(batch_size_label)

    epochs_label = tk.Label(root, text="Epochs")
    grid_pad(epochs_label, 3, 2)
    train_labels.append(epochs_label)

    optimizer_label = tk.Label(root, text="Optimizer")
    grid_pad(optimizer_label, 4, 2)
    train_labels.append(optimizer_label)

    # Branch dropdown
    branch_select = tk.OptionMenu(root, branch_var,
                                  "standard",
                                  "segmented",
                                  "seg and normal",
                                  "last image too",
                                  "two cams",
                                  "self segmenting")
    grid_pad(branch_select, 0, 1)

    # Weight selection buttons
    weight_button = tk.Button(root, textvariable=weight_path_var,
                              command=weight_on_click)
    grid_pad(weight_button, 1, 1)
    weight_2_button = tk.Button(root, textvariable=weight_2_path_var,
                                command=weight_2_on_click,
                                state="disabled")
    grid_pad(weight_2_button, 2, 1)

    # Data selection buttons
    data_button = tk.Button(root, textvariable=data_path_var,
                            command=data_on_click)
    grid_pad(data_button, 3, 1)

    # Training selection checkbox
    train_checkbox = tk.Checkbutton(root, text="Train", variable=train_bool)
    grid_pad(train_checkbox, 0, 2)

    # Train parameters
    train_params = []

    lr_entry = tk.Entry(root, width=10)
    lr_entry.insert(0, "0.01")
    grid_pad(lr_entry, 1, 3)
    train_params.append(lr_entry)

    batch_entry = tk.Entry(root, width=10)
    batch_entry.insert(0, "20")
    grid_pad(batch_entry, 2, 3)
    train_params.append(batch_entry)

    epochs_entry = tk.Entry(root, width=10)
    epochs_entry.insert(0, "8")
    grid_pad(epochs_entry, 3, 3)
    train_params.append(epochs_entry)

    optimizer_radio_0 = tk.Radiobutton(root, text="Adam",
                                       variable=optimizer_var, value=0)
    grid_pad(optimizer_radio_0, 4, 3)
    train_params.append(optimizer_radio_0)
    optimizer_radio_1 = tk.Radiobutton(root, text="SGD",
                                       variable=optimizer_var, value=1)
    grid_pad(optimizer_radio_1, 5, 3)
    train_params.append(optimizer_radio_1)

    # Run button
    run_button = tk.Button(root, text="Run", command=run, width=10)
    grid_pad(run_button, 6, 3, pady=10)

    train_switch()

    root.mainloop()



def grid_pad(widget, row=0, column=0, columnspan=1, rowspan=1, sticky="W",
             padx=3, pady=3):
    widget.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan,
                sticky=sticky, padx=padx, pady=pady)


if __name__ == "__main__":

    main()
