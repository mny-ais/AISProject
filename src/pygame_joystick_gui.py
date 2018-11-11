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

def update_joystick_values(joysticks, j_attributes):
    """ Updates multiple joystick values in only one function call."""
    current_stick_number = 0
    for j in joysticks:
        update_axis_values(j, j_attributes[current_stick_number]
                           ["Axis_state"])
        update_button_values(j, j_attributes[current_stick_number]
                             ["Button_state"])
        update_hat_values(j, j_attributes[current_stick_number]
                          ["Hat_state"])


def update_axis_values(joystick, axis_states):
    """ Updates a list of all axis values of a joystick in order."""
    for i in range(len(axis_states)):
        axis_states[i].set(joystick.get_axis(i))


def update_button_values(joystick, button_states):
    """ Updates a list of all button values of a joystick in order."""
    for i in range(len(button_states)):
        button_states[i].set(joystick.get_button(i))


def update_hat_values(joystick, hat_states):
    """ Updates a list of all hat values of a joystick in order."""
    for i in range(len(hat_states)):
        hat_states[i].set(str(joystick.get_hat(i)))


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

    # Set text
    tk.Label(root, text="{0} Joystick(s) found".
             format(len(joystick_list))).\
        grid(rowspan=4)

    # Create a list of dictionaries for each joystick
    j_attributes = []

    # initialize the joystick_attributes list
    for j in joystick_list:
        num_axis = j.get_numaxes()
        axis_values = [tk.DoubleVar(root) for i in range(num_axis)]
        num_buttons = j.get_numbuttons()
        button_values = [tk.DoubleVar(root) for i in range(num_buttons)]
        num_hats = j.get_hats()
        hat_values = [tk.StringVar for i in range(num_hats)]

        joystick_state = {
            "Name": j.get_name(),
            "Axes": num_axis,
            "Axis_state": axis_values,
            "Buttons": num_buttons,
            "Button_state": button_values,
            "Hats:": num_hats,
            "Hat_state": hat_values
        }
        j_attributes.append(joystick_state)

    # Fill the window with the joystick states
    current_row = 2
    for j in range(len(joystick_list)):
        # Name label
        tk.Label(root,
                 text="Name: {0}".format(j_attributes[j]["Name"])).\
            grid(row=current_row, rowspan=4)
        current_row += 1

        # Axis labels
        tk.Label(root,
                 text="Joystick has {0} axes".format(j_attributes[j]["Axes"])).\
            grid(row=current_row)
        for i in range(j_attributes[j]["Axes"]):
            axis_mod_2 = i % 2
            tk.Label(root,
                     "Axis {0}: {1}".
                     format(i, j_attributes[j]["Axis_state"][i].get())).\
                grid(row=current_row, column=axis_mod_2)
            if axis_mod_2 == 1:
                current_row += 1

        # If finished on an even value, i.e. odd number of axes, add an extra
        # blank line to compensate
        if j_attributes[j]["Axes"] % 2 == 0:
            current_row += 2
        else:
            current_row += 1

        # Button labels
        tk.Label(root,
                 text="Joystick has {0} buttons".
                 format(j_attributes[j]["Buttons"])).\
            grid(row=current_row, rowspan=4)
        for i in range(j_attributes[j]["Buttons"]):
            button_mod_4 = i % 4
            tk.Label(root,
                     "Button {0}: {1}".
                     format(i, j_attributes[j]["Button_state"][i].get())).\
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
            grid(row=current_row, rowspan=4)
        for i in range(j_attributes[j]["Hats"]):
            hat_mod_2 = i % 2
            tk.Label(root,
                     "Hat {0}: {1}".
                     format(i, j_attributes[j]["Hat_state"][i].get())).\
                grid(row=current_row, column=hat_mod_2)
            if hat_mod_2 == 1:
                current_row += 1

        # If finished on an even value, i.e. odd number of axes, add an extra
        # blank line to compensate
        if j_attributes[j]["Hats"] % 2 == 0:
            current_row += 2
        else:
            current_row += 1

    root.after(10, func=update_joystick_values(joystick_list, j_attributes))
    root.mainloop()


if __name__ == '__main__':
    joysticks = init_pygame_joysticks()
    controller_monitor(joysticks)

"""
    initial += 1
    stdscr.clear()

    # List the number of joysticks and show that the screen is refreshing
    num_joysticks = len(joysticks)
    stdscr.addstr(0, 0, "Found {0} joysticks".format(num_joysticks))
    stdscr.addstr(2, 0, "Refresh count: {0}".format(initial))
    stdscr.clrtobot()

    if num_joysticks > 0:
        num_lines = 3
        for joy_num in range(num_joysticks):
            # For each joystick, do the following:
            # State its name
            stdscr.addstr(num_lines, 0, "Name: {0}".format(joysticks[joy_num].
                                                           get_name()))
            num_lines += 1
            stdscr.addstr(num_lines, 0, "=============================")
            num_lines += 1

            # Get the number of axes and state their state
            num_axis = joysticks[joy_num].get_numaxes()
            stdscr.addstr(num_lines, 0, "Joystick has {0} axes".
                          format(num_axis))
            num_lines += 1

            if num_axis > 0:
                for num in range(num_axis):
                    axis_even = num % 2 == 0
                    if axis_even:
                        stdscr.addstr(num_lines, 0, "Axis {0}: {1}".
                                      format(num, joysticks[joy_num].
                                             get_axis(num)))
                    else:
                        stdscr.addstr(num_lines, 40, "Axis {0}: {1}".
                                      format(num, joysticks[joy_num].
                                             get_axis(num)))
                        num_lines += 1

                # if finished with all axes on an even axis value:
                if num_axis % 2 == 0:
                    num_lines += 1

            # Get the number of buttons and state their state
            num_buttons = joysticks[joy_num].get_numbuttons()
            num_lines += 1
            stdscr.addstr(num_lines, 0, "Joystick has {0} buttons".
                          format(num_buttons))
            num_lines += 1

            if num_buttons > 0:
                for num in range(num_buttons):
                    button_mod_3 = num % 3
                    if button_mod_3 == 0:
                        stdscr.addstr(num_lines, 0, "Button {0}: {1}".
                                      format(num, joysticks[joy_num].
                                             get_button(num)))
                    elif button_mod_3 == 1:
                        stdscr.addstr(num_lines, 20, "Button {0}: {1}".
                                      format(num, joysticks[joy_num].
                                             get_button(num)))
                    else:
                        stdscr.addstr(num_lines, 40, "Button {0}: {1}".
                                      format(num, joysticks[joy_num].
                                             get_button(num)))
                        num_lines += 1

                # If finished with buttons but not at mod 3 == 2, add an
                # empty line
                if num_buttons % 3 != 2:
                    num_lines += 1


            # Get the number of hats and state their state
            num_hats = joysticks[joy_num].get_numhats()
            num_lines += 1
            stdscr.addstr(num_lines, 0, "Joystick has {0} hats".
                          format(num_hats))
            num_lines += 1

            if num_hats > 0:
                for num in range(num_hats):
                    stdscr.addstr(num_lines, 0, "Hat {0}: {1}".
                                  format(num, joysticks[joy_num].
                                         get_hat(num)))
                    num_lines += 1

            num_lines += 1
"""