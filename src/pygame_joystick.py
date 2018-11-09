import curses

import time

try:
    import pygame

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is'
                       'installed')

pygame.display.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in
             range(pygame.joystick.get_count())]

for j in joysticks:
    j.init()

stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
stdscr.keypad(1)
stdscr.scrollok(1) # enable scrolling
stdscr.nodelay(True)

key = ''
initial = 0

stdscr.clear()
stdscr.refresh()
while key != ord('q'):
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

    stdscr.move(0, 0)
    stdscr.refresh()
    pygame.event.pump()
    key = stdscr.getch()
    time.sleep(0.01)

curses.endwin()
