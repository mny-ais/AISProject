# -*- coding: utf-8 -*-
"""Manual Control of AIScar simulator.

Allows manual control and recording of AIScar simulator with joystick input.

Authors:
    Johan Vertens
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
    Maximilian Roth
"""
# Uncomment if using python 2.x
# from __future__ import print_function

import airsim

import logging

import time

import cv2

import sys

import os

import platform

import csv

import random

import cProfile

try:
    import pygame
    from pygame.locals import K_q
    from pygame.locals import K_z
    from pygame.locals import K_n
    from pygame.locals import K_KP4
    from pygame.locals import K_KP6
    from pygame.locals import K_KP8
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is '
                       'installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is '
                       'installed')

from time import gmtime, strftime

AIRSIM_RES_WIDTH = 640
AIRSIM_RES_HEIGHT = 128
ADDITIONAL_CROP_TOP = 0

SAVE_DIR = 'save_dir/'

show_segmentation = True

WINDOW_WIDTH = AIRSIM_RES_WIDTH
WINDOW_HEIGHT = AIRSIM_RES_HEIGHT * 2 if show_segmentation \
    else AIRSIM_RES_HEIGHT

grab_image_distance = 0.1  # meters

max_lanes = 6


class VehicleControl(object):
    def __init__(self, mode, max_steering, max_throttle, max_brakes):
        """ Holds a CarControl object and some extra functionality.

            Allows the addition of noise into the system by holding the airsim
            CarControls object, user input, and noise input separately. Only at
            the end of the update call is the airsim CarControls object
            updated. Before that, only the user input and noise input are
            updated.

            Only the steering and brakes can have noise added to them.

        Args:
            mode (string): Available options are:
                           - "left": Only the left analog stick is used for
                                     control.
                           - "right": Only the right analog stick is used for
                                      control.
                           - "rc": Mimics an rc car, where the left stick is
                                   used to control throttle and the right stick
                                   controls steering.
                           - "game": Mimics default video game bindings, with
                                     throttle and brake controls bound to the
                                     triggers. Only available when using a PS4
                                     controller.
            max_steering (float): Maximum value for steering.
            max_throttle (float): Maximum value for throttle.
            max_brakes (float): Maximum value for breaks.
        """
        self.request_new_episode = False
        self.request_quit = False

        self.car_control = airsim.CarControls()
        self.car_control.is_manual_gear = True
        self.car_control.manual_gear = 1

        # Requested Direction
        self.requested_direction = 0

        # User and noise input (Required to allow for inserting noise)
        self.user_steering = 0.0
        self.user_brakes = 0.0
        self.noise_steering = 0.0
        self.noise_brakes = 0.0
        self.noisy = False
        self.noise_max_amp = float(0.3)
        self.noise_how_long = 0
        self.noise_already = 0
        self.noise_left_or_right = 0

        # Max values
        self.max_steering = float(max_steering)
        self.max_throttle = float(max_throttle)
        self.max_brakes = float(max_brakes)

        # First assign bindings based on OS.
        self.current_os = platform.system()
        if self.current_os == "Linux":
            self.l_x = 0
            self.l_y = 1
            self.r_x = 3
            self.r_y = 4
            self.l2 = 2
            self.r2 = 5
            self.start_button = 9
            self.l1 = 4  # Handbrakes
            self.r1 = 5  # Reverse enable
            self.deadzone = 0.02

        elif self.current_os == "Windows":
            self.l_x = 0
            self.l_y = 1
            self.r_x = 4
            self.r_y = 3
            # XInput only supports a single axis for both triggers. Positive
            # for l2, negative for r2
            self.triggers = 2
            self.start_button = 7
            self.l1 = 4
            self.r1 = 5
            self.deadzone = 0.04

        # Then assign the control scheme

        # use a switcher
        switcher = {
            "left": self._update_left,
            "right": self._update_right,
            "rc": self._update_rc,
            "game": self._update_game_scheme
        }
        self.func = switcher[mode]

    def print_state(self):  # {{{
        print("Steering: {0}".format(self.car_control.steering))
        print("Throttle: {0}".format(self.car_control.throttle))
        print("Brake: {0}".format(self.car_control.throttle))
        print("Gear: {0}".format(self.car_control.manual_gear))

    def update_car_controls(self):
        """ Gets the pygame event queue and uses it to control the car.

            Gets the pygame event queue and uses that information to update the
            object values. If no events exist, then simply don't update. If a
            new episode is requested either by pressing r or start, then
            changes the boolean flag to True, and waits for it to be reset by
            the _on_new_episode function of the AISGame class.

            Assumes that Linux will run with a Playstation controller and
            Windows with XInput (e.g. Steam Controller).

            No longer supports keyboard controls because it's just a waste of
            time to program something we'll never use.
        """
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.JOYAXISMOTION:
                self.func(event)
            elif event.type == pygame.JOYBUTTONDOWN:
                self._update_button_downs(event)
            elif event.type == pygame.JOYBUTTONUP:
                self._update_button_ups(event)
            elif event.type == pygame.KEYDOWN:
                self._update_key_downs(event)
            elif event.type == pygame.QUIT:
                self.request_quit = True

        if self.noisy:
            self._make_some_noise()

        self.car_control.steering = (self.user_steering
                                     + self.noise_steering) * self.max_steering
        self.car_control.brake = (self.user_brakes
                                  + self.noise_brakes) * self.max_brakes

    def _update_left(self, event):
        """ Updates based on left control scheme."""
        # Steering event
        if event.axis == self.l_x:
            self.user_steering = self._deadzone(float(event.value))

        # Throttle/brake event
        elif event.axis == self.l_y:
            self._throttle_brake_combined(self._deadzone(event.value))

    def _update_right(self, event):
        """ Updates based on right control scheme."""
        # Steering event
        if event.axis == self.r_x:
            self.user_steering = self._deadzone(float(event.value))

        # Throttle/brake event
        elif event.axis == self.r_y:
            self._throttle_brake_combined(self._deadzone(event.value))

    def _update_rc(self, event):
        """ Updates with a control scheme similar to an RC car."""
        # Steering event
        if event.axis == self.r_x:
            self.user_steering = self._deadzone(float(event.value))

        # Throttle/brake event
        elif event.axis == self.l_y:
            self._throttle_brake_combined(self._deadzone(event.value))

    def _update_game_scheme(self, event):
        """ Updates based on game control scheme."""
        # Steering event
        if event.axis == self.l_x:
            self.user_steering = self._deadzone(float(event.value))

        # Throttle/brake event
        elif self.current_os == "Linux":
            # Throttle event
            if event.axis == self.r2:
                self.car_control.throttle = self._deadzone(event.value + 1) \
                                            * 0.5 * self.max_throttle
            # Brake or reverse event
            elif event.axis == self.l2 and self.car_control.manual_gear > 0:
                self.user_brakes = self._deadzone(event.value + 1) \
                                   * 0.5
            elif event.axis == self.l2 and self.car_control.manual_gear < 0:
                self.car_control.throttle = self._deadzone(event.value + 1) \
                                            * 0.5 * self.max_throttle

        elif self.current_os == "Windows":
            if event.axis == self.triggers:
                self._throttle_brake_combined(self._deadzone(event.value))

    def _update_button_downs(self, event):
        """ Handles buttons since all control schemes use the same buttons.

            Handles all button up events.
        """
        if event.button == self.r1:
            # Reverse
            self.car_control.manual_gear = -1
        elif event.button == self.l1:
            # Handbrakes
            self.car_control.handbrake = True

        elif event.button == self.start_button:
            self.request_new_episode = True

    def _update_button_ups(self, event):
        """ Handles buttons since all control schemes use the same buttons.

            Handles all button up events.
        """
        if event.button == self.r1:
            # Reverse
            self.car_control.manual_gear = 1
        elif event.button == self.l1:
            # Handbrakes
            self.car_control.handbrake = False

    def _update_key_downs(self, event):
        """
            0 is Forward
            -1 is Left
            1 is Right
        """
        if event.key == K_KP8:
            self.requested_direction = 0
        elif event.key == K_KP4:
            self.requested_direction = -1
        elif event.key == K_KP6:
            self.requested_direction = 1
        elif event.key == K_n:
            if self.noisy:
                self.noise_steering = 0
                print('Removing noise to steering and throttle.')
            else:
                print('Adding noise to steering and throttle.')
            self.noisy = not self.noisy

    def _deadzone(self, value):
        """ Handles deadzone based on controller."""
        if -self.deadzone < value < self.deadzone:
            return 0

        return value

    def _throttle_brake_combined(self, value):
        """ Modifies the value of the throttle or brake.

        Args:
            value (float): A floating point value between -1 and 1.
        """
        if value < 0 and self.car_control.manual_gear > 0:
            self.car_control.throttle = abs(value) * self.max_throttle

        if value > 0 and self.car_control.manual_gear > 0:
            self.user_brakes = value

        if value > 0 and self.car_control.manual_gear < 0:
            self.car_control.throttle = value * self.max_throttle

        if value == 0:
            self.car_control.throttle = 0
            self.user_brakes = 0  # }}}

    def _make_some_noise(self):
        """ Changes the noise value to add noise to the system."""
        if self.noise_how_long == self.noise_already:
            self.noise_steering = 0
            self.noise_already = 0
            self.noise_how_long = random.randint(5, 20)
            self.noise_left_or_right = random.randint(0, 1)
        else:
            self.noise_already += 1
            steering_noise = float(self.noise_max_amp) / 20
            if self.noise_already < (self.noise_how_long / 2):
                if self.noise_left_or_right == 0:
                    self.noise_steering += steering_noise
                else:
                    self.noise_steering -= steering_noise
            else:
                if self.noise_left_or_right == 0:
                    self.noise_steering -= steering_noise
                else:
                    self.noise_steering += steering_noise


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) \
               / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class AISGame(object):
    def __init__(self, control_mode, max_steering, max_throttle, max_brakes):
        """ Initializes the AISGame object.
        Args:
            control_mode (string): The control mode to used with the joystick.
        """
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Change camera direction (in radians)
        self.client.simSetCameraOrientation(1,
                                            airsim.to_quaternion(0,
                                                                 0,
                                                                 0.523599))
        self.client.simSetCameraOrientation(2,
                                            airsim.to_quaternion(0,
                                                                 0,
                                                                 -0.523599))

        # Internally represents the vehicle control state
        self.vehicle_controls = VehicleControl(control_mode, max_steering,
                                               max_throttle, max_brakes)

        self._timer = None
        self.save_timer = None
        self._display = None
        self._main_image = None
        self._seg_image = None
        self._is_on_reverse = False
        self._position = None
        self.counter = 0
        self.color_map = {}
        self.val_map = {}
        self.set_segmentation_ids()

        self.recording = False
        self.request_stop_recording = False
        self.record_path = None
        self.save_counter = 0
        self.csv_data = []

        self.last_pos = np.zeros(3)

        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(x)
                          for x in range(pygame.joystick.get_count())]
        for j in self.joysticks:
            j.init()
        print("Found %d joysticks" % (len(self.joysticks)))

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        while True:
            self._on_loop()
            self._on_render()

    def set_segmentation_ids(self):

        # rgb_file = open("seg_rgbs.txt", "r")
        # lines = rgb_file.readlines()
        # for l in lines:
        #     s = l.split('[')
        #     self.color_map[int(s[0].rstrip())] = eval('[' + s[1].rstrip())
        #     self.val_map[tuple(eval('[' + s[1].rstrip()))] = int(s[0].
        #                                                      rstrip())

        found = self.client.simSetSegmentationObjectID("[\w]*", 0, True)
        print("Reset all segmentations to zero: %r" % found)

        self.client.simSetSegmentationObjectID("ParkingAnnotRoad[\w]*",
                                               22,
                                               True)
        self.client.simSetSegmentationObjectID("CrosswalksRoad[\w]*",
                                               23,
                                               True)
        self.client.simSetSegmentationObjectID("Car[\w]*",
                                               24,
                                               True)

        self.client.simSetSegmentationObjectID("GroundRoad[\w]*",
                                               25,
                                               True)
        self.client.simSetSegmentationObjectID("SolidMarkingRoad[\w]*",
                                               26,
                                               True)
        self.client.simSetSegmentationObjectID("DashedMarkingRoad[\w]*",
                                               27,
                                               True)
        self.client.simSetSegmentationObjectID("StopLinesRoad[\w]*",
                                               28,
                                               True)
        self.client.simSetSegmentationObjectID("ParkingLinesRoad[\w]*",
                                               29,
                                               True)
        self.client.simSetSegmentationObjectID("JunctionsAnnotRoad[\w]*",
                                               30,
                                               True)

        for i in range(max_lanes):
            self.client.simSetSegmentationObjectID("LaneRoadAnnot"
                                                   + str(i) + "[\w]*", i + 31,
                                                   True)

    def _initialize_game(self):
        self._on_new_episode()

        self._display = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')

    def _on_new_episode(self):
        print('Starting new episode...')
        self.client.reset()
        self._timer = Timer()
        self.save_timer = Timer()
        self.vehicle_controls.request_new_episode = False

    @staticmethod
    def response_to_cv(r, channels):
        if r.compress:
            image = cv2.imdecode(np.fromstring(r.image_data_uint8,
                                               dtype=np.uint8),
                                 1)
            image = image.reshape(r.height, r.width, channels)
            image = cv2.cvtColor(image[:, :, 0:channels], cv2.COLOR_RGB2BGR)

        else:
            image = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
            image = image.reshape(r.height, r.width, channels + 1)
            image = image[:, :, 0:channels]
        return image

    def seg_rgb_to_values(self, seg_rgb):
        val_map = np.zeros((seg_rgb.shape[0], seg_rgb.shape[1]), dtype=np.uint8)
        for v, color in self.color_map.items():
            v_map = np.all(seg_rgb == tuple(color), axis=-1).astype(np.uint8) \
                    * v
            val_map += v_map
        return val_map

    def gen_labels(self, label_map):

        val_map = np.zeros((label_map.shape[0], label_map.shape[1]))
        for x in range(label_map.shape[0]):
            for y in range(label_map.shape[1]):
                val_map[x, y] = self.val_map[tuple(label_map[x, y, :])]

        val_map = val_map.astype(np.uint8)

        return val_map

    def _on_loop(self):
        self._timer.tick()
        self.save_timer.tick()

        loc = self.client.getCarState()
        pos = loc.kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.y_val])

        if show_segmentation:
            responses = self.client.simGetImages(
                # Camera Scenes
                [airsim.ImageRequest("0",
                                     airsim.ImageType.Scene,
                                     False,
                                     False),
                 airsim.ImageRequest("1",
                                     airsim.ImageType.Scene,
                                     False,
                                     False),
                 airsim.ImageRequest("2",
                                     airsim.ImageType.Scene,
                                     False,
                                     False),
                 # Camera Segmentations
                 airsim.ImageRequest("0",
                                     airsim.ImageType.Segmentation,
                                     False,
                                     False),
                 airsim.ImageRequest("1",
                                     airsim.ImageType.Segmentation,
                                     False,
                                     False),
                 airsim.ImageRequest("1",
                                     airsim.ImageType.Segmentation,
                                     False,
                                     False)])


        else:
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0",
                                     airsim.ImageType.Scene,
                                     False,
                                     False),
                 airsim.ImageRequest("1",
                                     airsim.ImageType.Scene,
                                     False,
                                     False),
                 airsim.ImageRequest("2",
                                     airsim.ImageType.Scene,
                                     False,
                                     False)])

        rgb = [None, None, None]
        seg = [None, None, None]

        for i in range(3):
            if responses[i]:
                # Check each rgb image
                rgb[i] = self.response_to_cv(responses[i], 3)

                # Add to main image for rendering if it's the center camera
                if i == 0:
                    self._main_image = rgb[0]

        for i in range(3):
            # Check each segmentation image
            if responses[i + 3]:
                seg[i] = self.response_to_cv(responses[i + 3], 3)

            # Add to main image for rending if it's the center camera
            if i == 0:
                self._seg_image = seg[0]

        if self.recording:
            if len(responses) == 6 \
                    and np.linalg.norm(self.last_pos - pos)\
                        > grab_image_distance:
                # Record the rgb images
                if self.record_path is not None:
                    for i in range(3):
                        cv2.imwrite(self.record_path + 'image_'
                                    + str(self.save_counter)
                                    + "-cam_" + str(i)
                                    + '.png',
                                    cv2.cvtColor(rgb[i], cv2.COLOR_BGR2RGB))
                        cv2.imwrite(self.record_path + 'seg_'
                                    + str(self.save_counter)
                                    + "-cam_" + str(i)
                                    + '.png',
                                    cv2.cvtColor(seg[i], cv2.COLOR_BGR2RGB))

                    # Prepare csv data
                    if self.save_counter == 0:
                        self.csv_data.append(
                            ["No.",
                             "Steering",
                             "Throttle",
                             "Brakes",
                             "Handbrake",
                             "Gear",
                             "Requested_Direction"]
                        )
                    else:
                        self.csv_data.append(
                            [self.save_counter,
                             self.vehicle_controls.user_steering,
                             self.vehicle_controls.car_control.throttle,
                             self.vehicle_controls.user_brakes,
                             self.vehicle_controls.car_control.handbrake,
                             self.vehicle_controls.car_control.manual_gear,
                             self.vehicle_controls.requested_direction]
                        )
                    self.save_counter += 1
                self.last_pos = pos

        if self.request_stop_recording:
            # Stops recording, closes the csv file, and resets the counter
            self.recording = False
            with open(self.record_path + "control_input.csv", mode='x') \
                    as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(self.csv_data)
            csv_file.close()
            self.csv_data = []
            self.save_counter = 0
            self.request_stop_recording = False

        self.counter += 1

        # Print FPS
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            direction = "Forward"
            if self.vehicle_controls.requested_direction == 0:
                direction = "Go Forward Yviiiiii"
            elif self.vehicle_controls.requested_direction == -1:
                direction = "Left"
            else:
                direction = "Right"
            print("FPS: {0}, Drive direction: {1}".
                  format(self.counter,
                         direction))
            self.counter = 0
            self._timer.lap()

        # If a new episode is requested, create a new episode.
        if self.vehicle_controls.request_new_episode:
            self._on_new_episode()

        # Get key presses and parse them
        # pygame.event.pump()
        self._keyboard_controls(pygame.key.get_pressed())

        self.vehicle_controls.update_car_controls()
        self.client.setCarControls(self.vehicle_controls.car_control)

        # Check if the game should quit
        if self.vehicle_controls.request_quit:
            pygame.display.quit()
            pygame.quit()
            sys.exit()

        pygame.display.update()

    def _keyboard_controls(self, keys):
        """ Parses keyboard input into actions."""
        if keys[K_q]:
            if not self.recording:
                self.record_path = SAVE_DIR + strftime("%Y_%m_%d_%H-%M-%S",
                                                       gmtime()) + '/'
                if not os.path.exists(self.record_path):
                    os.makedirs(self.record_path)
                self.recording = True
                self.save_counter = 0
                print('Recording on, saving to: %s' % self.record_path)
        if keys[K_z]:
            if self.recording:
                self.request_stop_recording = True
                print('Recording off, saved to: %s' % self.record_path)

    def _on_render(self):
        if self._main_image is not None:
            surface_main = pygame.surfarray.make_surface(
                self._main_image.swapaxes(0, 1))
            self._display.blit(surface_main, (0, 0))

            if show_segmentation:
                surface_seg = pygame.surfarray.make_surface(
                    self._seg_image.swapaxes(0, 1))
                self._display.blit(surface_seg, (0, 85 * 1))

        pygame.display.flip()

    def execute_profile(self):
        """Launch the PyGame but for only 5 updates."""
        pygame.init()
        self._initialize_game()
        count = 0
        while count < 20:
            self._on_loop()
            self._on_render()
            count += 1


def main(control_mode="rc", max_steering=0.5, max_throttle=0.4,
         max_brakes=0.7):
    game = AISGame(control_mode, max_steering, max_throttle, max_brakes)
    game.execute()


def profile():
    game = AISGame('left', 1.0, 1.0, 1.0)
    cProfile.runctx("game.execute_profile()",
                    globals(),
                    locals(),
                    filename='profile.txt')


if __name__ == '__main__':

    try:
        if len(sys.argv) == 2:
            if sys.argv[1] == "profile":
                profile()
            else:
                main(sys.argv[1])
        elif len(sys.argv) == 3:
            main(sys.argv[1], sys.argv[2])
        elif len(sys.argv) == 4:
            main(sys.argv[1], sys.argv[2], sys.argv[3])
        elif len(sys.argv) == 5:
            main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        elif len(sys.argv) > 5:
            raise Exception("Too many arguments. \n"
                            "Usage: \n"
                            "project_kickstart.py [control_mode] "
                            "[max_steering] [max_throttle] [max_brakes]")
        else:
            main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
