import math
import time

import tkinter as tk

try:
    import pygame

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is'
                       'installed')


def init_pygame_joysticks():
    """ Initializes pygame joysticks and returns a list of available joysticks.
    """
    pygame.display.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in
                 range(pygame.joystick.get_count())]

    for j in joysticks:
        j.init()

    return joysticks

def update_joystick_values(j_attributes, root):
    """ Updates multiple joystick values in only one function call."""

    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            # Handles axis motion
            value = truncate(deadzone(event.value), 4)
            j_attributes[event.joy]["Axis_state"][event.axis].\
                set("Axis {0}: {1}".format(event.axis, value))
        elif event.type == pygame.JOYBALLMOTION:
            # Handles ball motion
            j_attributes[event.joy]["Ball_state"][event.ball].\
                set("Ball {0}: {1}".format(event.ball, event.value))
        elif event.type == pygame.JOYBUTTONDOWN:
            # Handles button down
            j_attributes[event.joy]["Button_state"][event.button].\
                set("Button {0}: 1".format(event.button))
        elif event.type == pygame.JOYBUTTONUP:
            # Handles button up
            j_attributes[event.joy]["Button_state"][event.button].\
                set("Button{0}: 0".format(event.button))
        elif event.type == pygame.JOYHATMOTION:
            # Handles hat motion
            j_attributes[event.joy]["Hat_state"][event.hat]. \
                set("Hat {0}: {1}".format(event.hat, event.value))


    root.after(10, update_joystick_values, j_attributes, root)

def deadzone(value):
    if -0.002 < value < 0.002:
        return 0
    else:
        return value

def truncate(number, digits):
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper

def controller_monitor(joystick_list):
    """ Main function that represents the app and extends tk.Tk

    Concept:
        Create a window with a bunch of labels in it. The window will have a
        drop-down menu to choose which controller to use. Window uses StringVar
        to constantly update values.

    Args:
        joystick_list(list): List of the joysticks to be shown
    """
    # Create the window
    root = tk.Tk()
    root.title("Pygame Joystick Monitor")

    # Configure the grid
    for i in range(4):
        root.grid_columnconfigure(i, minsize=100)

    # Set text
    tk.Label(root, text="{0} Joystick(s) found".
             format(len(joystick_list))).\
        grid(row=0, columnspan=4)

    # Create a list of dictionaries for each joystick
    j_attributes = []

    # initialize the joystick_attributes list
    for j in joystick_list:
        num_axis = j.get_numaxes()
        axis_values = [tk.StringVar(master=root, value="Axis {0}: 0.0".
                                    format(i))
                       for i in range(num_axis)]
        num_balls = j.get_numballs()
        ball_values = [tk.StringVar(master=root, value="Ball {0}: (0, 0)".
                                    format(i))
                         for i in range(num_balls)]
        num_buttons = j.get_numbuttons()
        button_values = [tk.StringVar(master=root, value="Button {0}: 0".
                                      format(i))
                         for i in range(num_buttons)]
        num_hats = j.get_numhats()
        hat_values = [tk.StringVar(master=root, value="Hat {0}: (0, 0)".
                                   format(i))
                      for i in range(num_hats)]

        joystick_state = {
            "Name": j.get_name(),
            "Axes": num_axis,
            "Axis_state": axis_values,
            "Balls": num_balls,
            "Ball_state": ball_values,
            "Buttons": num_buttons,
            "Button_state": button_values,
            "Hats": num_hats,
            "Hat_state": hat_values
        }
        j_attributes.append(joystick_state)

    # Fill the window with the joystick states
    current_row = 3

    for j in range(len(joystick_list)):
        # Name label
        tk.Label(root,
                 text="Name: {0}".format(j_attributes[j]["Name"])).\
            grid(row=current_row, columnspan=4)
        current_row += 1

        # Axis labels
        tk.Label(root,
                 text="Joystick has {0} axes".
                 format(j_attributes[j]["Axes"])).\
            grid(row=current_row, columnspan=4)
        current_row += 1
        for i in range(j_attributes[j]["Axes"]):
            axis_mod_2 = i % 2
            tk.Label(root,
                     textvariable=j_attributes[j]["Axis_state"][i]).\
                grid(row=current_row, column=axis_mod_2)
            if axis_mod_2 == 1:
                current_row += 1

        # If finished on an even value, i.e. odd number of axes, add an extra
        # blank line to compensate
        if j_attributes[j]["Axes"] % 2 == 0:
            current_row += 2
        else:
            current_row += 1

        # Ball labels
        tk.Label(root,
                 text="Joystick has {0} balls".
                 format(j_attributes[j]["Balls"])). \
            grid(row=current_row, columnspan=4)
        current_row += 1
        for i in range(j_attributes[j]["Balls"]):
            axis_mod_2 = i % 2
            tk.Label(root,
                     textvariable=j_attributes[j]["Ball_state"][i]). \
                grid(row=current_row, column=axis_mod_2)
            if axis_mod_2 == 1:
                current_row += 1
        # If finished on an even value, i.e. odd number of axes, add an extra
        # blank line to compensate
        if j_attributes[j]["Balls"] % 2 == 0:
            current_row += 2
        else:
            current_row += 1

        # Button labels
        tk.Label(root,
                 text="Joystick has {0} buttons".
                 format(j_attributes[j]["Buttons"])).\
            grid(row=current_row, columnspan=4)
        current_row += 1
        for i in range(j_attributes[j]["Buttons"]):
            button_mod_4 = i % 4
            tk.Label(root,
                     textvariable=j_attributes[j]["Button_state"][i]).\
                grid(row=current_row, column=button_mod_4)
            if button_mod_4 == 3:
                current_row += 1

        # If finished on a value mod 4 != 3, i.e. not a factor of 4, add an
        # extra blank line to compensate
        if j_attributes[j]["Buttons"] % 4 != 3:
            current_row += 2
        else:
            current_row += 1

        # Hat labels
        tk.Label(root,
                 text="Joystick has {0} hats".
                 format(j_attributes[j]["Hats"])).\
            grid(row=current_row, columnspan=4)
        current_row += 1
        for i in range(j_attributes[j]["Hats"]):
            hat_mod_2 = i % 2
            tk.Label(root,
                     textvariable=j_attributes[j]["Hat_state"][i]).\
                grid(row=current_row, column=hat_mod_2)
            if hat_mod_2 == 1:
                current_row += 1

        # If finished on an even value, i.e. odd number of axes, add an extra
        # blank line to compensate
        if j_attributes[j]["Hats"] % 2 == 0:
            current_row += 3
        else:
            current_row += 2

        current_row += 1

    root.after(10, update_joystick_values, j_attributes, root)
    root.mainloop()



if __name__ == '__main__':
    joysticks = init_pygame_joysticks()
    controller_monitor(joysticks)
