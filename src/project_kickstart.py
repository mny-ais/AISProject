# Manual control of AIScar simulator
# Johan Vertens 2018
# Yvan Satyawan 2018

from __future__ import print_function

import airsim

import logging

import time

import cv2

import os

import sys

import platform

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
    from pygame.locals import K_z
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is'
                       'installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is'
                       'installed')

from time import gmtime, strftime

AIRSIM_RES_WIDTH = 640
AIRSIM_RES_HEIGHT = 128
ADDITIONAL_CROP_TOP = 0

SAVE_DIR = 'save_dir/'

show_segmentation = True

WINDOW_WIDTH = AIRSIM_RES_WIDTH
WINDOW_HEIGHT = AIRSIM_RES_HEIGHT*2 if show_segmentation else AIRSIM_RES_HEIGHT

grab_image_distance = 0.1  # meters

max_lanes = 6


class VehicleControl(object):
    def __init__(self, control_mode):
        """ Holds a CarControl object and some extra functionality."""
        self.request_new_episode = False

        self.car_control = airsim.CarControls()
        self.car_control.is_manual_gear = True
        self.car_control.manual_gear = 1

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
        self.func = switcher[control_mode]

    def print_state(self):
        print("Steering: {0}".format(self.car_control.steering))
        print("Throttle: {0}".format(self.car_control.throttle))
        print("Brake: {0}".format(self.car_control.throttle))
        print("Gear: {0}".format(self.car_control.manual_gear))

    def _update_car_controls(self):
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

    def _update_left(self, event):
        """ Updates based on left control scheme."""
        # Steering event
        if event.axis == self.l_x:
            self.car_control.steering = event.value / 2

        # Throttle/brake event
        elif event.axis == self.l_y:
            self._throttle_brake_combined(self._deadzone(event.value))

    def _update_right(self, event):
        """ Updates based on right control scheme."""
        # Steering event
        if event.axis == self.r_x:
            self.car_control.steering = event.value / 2

        # Throttle/brake event
        elif event.axis == self.r_y:
            self._throttle_brake_combined(self._deadzone(event.value))

    def _update_rc(self, event):
        """ Updates with a control scheme similar to an RC car."""
        # Steering event
        if event.axis == self.r_x:
            self.car_control.steering = event.value / 2

        # Throttle/brake event
        elif event.axis == self.l_y:
            self._throttle_brake_combined(self._deadzone(event.value))

    def _update_game_scheme(self, event):
        """ Updates based on game control scheme."""
        # Steering event
        if event.axis == self.l_x:
            self.car_control.steering = event.value / 2

        # Throttle/brake event
        elif self.current_os == "Linux":
            if event.axis == self.l2 or event.axis == self.r2:
                self.car_control.throttle = (event.value + 1 / 4)
                self.car_control.brake = (event.value + 1) / 4

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
            self.car_control.throttle = abs(value / 2)

        if value > 0 and self.car_control.manual_gear > 0:
            self.car_control.brake = value / 2

        if value > 0 and self.car_control.manual_gear < 0:
            self.car_control.throttle = value / 2

        if value == 0:
            self.car_control.throttle = 0
            self.car_control.brake = 0


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
        return float(self.step - self._lap_step)\
               / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class AISGame(object):
    def __init__(self, control_mode):
        """ Initializes the AISGame object.
        Args:
            control_mode (string): The control mode to used with the joystick.
        """
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Internally represents the vehicle control state
        self.vehicle_controls = VehicleControl(control_mode)

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
        self.record_path = None
        self.save_counter = 0

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
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

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

    def response_to_cv(self, r, channels):
        if r.compress:
            image = cv2.imdecode(np.fromstring(r.image_data_uint8,
                                               dtype=np.uint8),
                                 1)
            image = image.reshape(r.height, r.width, channels)
            image = cv2.cvtColor(image[:, :, 0:channels], cv2.COLOR_RGB2BGR)

        else:
            image = np.fromstring(r.image_data_uint8, dtype=np.uint8)
            image = image.reshape(r.height, r.width, channels+1)
            image = image[:, :, 0:channels]
        return image

    def seg_rgb_to_values(self, seg_rgb):
        val_map = np.zeros((seg_rgb.shape[0], seg_rgb.shape[1]), dtype=np.uint8)
        for v, color in self.color_map.items():
            v_map = np.all(seg_rgb == tuple(color), axis=-1).astype(np.uint8)\
                    * v
            val_map += v_map
        return val_map

    def gen_labels(self, map):

        val_map = np.zeros((map.shape[0], map.shape[1]))
        for x in range(map.shape[0]):
            for y in range(map.shape[1]):
                val_map[x, y] = self.val_map[tuple(map[x, y, :])]

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
                [airsim.ImageRequest("0",
                                     airsim.ImageType.Scene,
                                     False,
                                     False),
                 airsim.ImageRequest("0",
                                     airsim.ImageType.Segmentation,
                                     False,
                                     False)])

        else:
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0",
                                     airsim.ImageType.Scene,
                                     False,
                                     False)])

        if len(responses) > 0:
            rgb = self.response_to_cv(responses[0], 3)

            # crop_img = rgb[ADDITIONAL_CROP_TOP:, :]
            # rgb = cv2.resize(crop_img, (640, 85))
            self._main_image = rgb

        # Did we get a segmentation image?
        if len(responses) > 1:
            seg = self.response_to_cv(responses[1], 3)
            self._seg_image = seg

        if self.recording and len(responses) > 1:
            # if self.save_timer.elapsed_seconds_since_lap() > 0.2:
            # Record image every $grap_image_distance meters
            if np.linalg.norm(self.last_pos - pos) > grab_image_distance:
                    if self.record_path is not None:
                        cv2.imwrite(self.record_path + 'image_'
                                    + str(self.save_counter) + '.png',
                                    cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(self.record_path + 'seg_'
                                    + str(self.save_counter) + '.png',
                                    cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
                        self.save_counter += 1
                    self.last_pos = pos

        self.counter += 1

        # Print FPS
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            print("FPS: %d" % self.counter)
            self.counter = 0
            self._timer.lap()

        # If a new episode is requested, create a new episode.
        if self.vehicle_controls.request_new_episode:
            self._on_new_episode()

        # TODO Get keyboard keys to allow recording

        self.client.setCarControls(self.vehicle_controls.car_control)

        pygame.display.update()

    # def _get_keyboard_control(self, keys):
    #     """
    #     Return a VehicleControl message based on the pressed keys. Return None
    #     if a new episode was requested.
    #     """
    #     # Left/Right 0
    #     # Throttle 5
    #
    #     if keys[K_r]:
    #         return None
    #     control = VehicleControl()
    #     if keys[K_LEFT] or keys[K_a]:
    #         control.steer = -0.7
    #     if keys[K_RIGHT] or keys[K_d]:
    #         control.steer = 0.7
    #     if keys[K_UP] or keys[K_w]:
    #         control.throttle = 0.3
    #     if keys[K_DOWN] or keys[K_s]:
    #         control.brake = 1.0
    #     if keys[K_SPACE]:
    #         control.hand_brake = True
    #     if keys[K_q]:
    #         if not self.recording:
    #             self.record_path = SAVE_DIR + strftime("%Y_%m_%d_%H:%M:%S",
    #                                                    gmtime()) + '/'
    #             if not os.path.exists(self.record_path):
    #                 os.makedirs(self.record_path)
    #             self.recording = True
    #             self.save_counter = 0
    #             print('Recording on, saving to: %s' % self.record_path)
    #     if keys[K_z]:
    #         if self.recording:
    #             self.recording = False
    #             print('Recording off, saved to: %s' % self.record_path)
    #
    #     return control
    #
    # This function gets called everytime the screen needs to be rendered
    def _on_render(self):
        if self._main_image is not None:
            surface_main = pygame.surfarray.make_surface(
                self._main_image.swapaxes(0, 1))
            self._display.blit(surface_main, (0, 0))

            if show_segmentation:
                surface_seg = pygame.surfarray.make_surface(
                    self._seg_image.swapaxes(0, 1))
                self._display.blit(surface_seg, (0, 85*1))

        pygame.display.flip()


def main(control_mode="left"):
    game = AISGame(control_mode)
    game.execute()


if __name__ == '__main__':

    try:
        if len(sys.argv) > 1:
            main(sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
