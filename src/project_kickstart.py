# Manual control of AIScar simulator
# Johan Vertens 2018

from __future__ import print_function

import airsim

import logging

import time

import cv2

import os

import sys

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
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

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
    def __init__(self):
        self.steer = 0
        self.throttle = 0
        self.brake = 0
        self.hand_brake = 0
        self.reverse = False

    def print_state(self):
        print("Steering: {0}".format(self.steer))
        print("Throttle: {0}".format(self.throttle))
        print("Brake: {0}".format(self.brake))

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
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

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
        self.car_controls = airsim.CarControls()
        self.control_mode = control_mode

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
        self.setSegmentationIds()

        self.recording = False
        self.record_path = None
        self.save_counter = 0

        self.last_pos = np.zeros((3))

        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
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

    def setSegmentationIds(self):

        # rgb_file = open("seg_rgbs.txt", "r")
        # lines = rgb_file.readlines()
        # for l in lines:
        #     s = l.split('[')
        #     self.color_map[int(s[0].rstrip())] = eval('[' + s[1].rstrip())
        #     self.val_map[tuple(eval('[' + s[1].rstrip()))] = int(s[0].rstrip())

        found = self.client.simSetSegmentationObjectID("[\w]*", 0, True);
        print("Reset all segmentations to zero: %r" % (found))

        self.client.simSetSegmentationObjectID("ParkingAnnotRoad[\w]*", 22, True)
        self.client.simSetSegmentationObjectID("CrosswalksRoad[\w]*", 23, True)
        self.client.simSetSegmentationObjectID("Car[\w]*", 24, True)

        self.client.simSetSegmentationObjectID("GroundRoad[\w]*", 25, True)
        self.client.simSetSegmentationObjectID("SolidMarkingRoad[\w]*", 26, True)
        self.client.simSetSegmentationObjectID("DashedMarkingRoad[\w]*", 27, True)
        self.client.simSetSegmentationObjectID("StopLinesRoad[\w]*", 28, True)
        self.client.simSetSegmentationObjectID("ParkingLinesRoad[\w]*", 29, True)
        self.client.simSetSegmentationObjectID("JunctionsAnnotRoad[\w]*", 30, True)

        for i in range(max_lanes):
            self.client.simSetSegmentationObjectID("LaneRoadAnnot" + str(i) +"[\w]*", i+31, True)

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

    def response_to_cv(self, r, channels):
        if r.compress:
            image = cv2.imdecode(np.fromstring(r.image_data_uint8, dtype=np.uint8), 1)
            image = image.reshape(r.height, r.width, channels)
            image = cv2.cvtColor(image[:, :, 0:channels], cv2.COLOR_RGB2BGR)

        else:
            image = np.fromstring(r.image_data_uint8, dtype=np.uint8)
            image = image.reshape(r.height, r.width, channels+1)
            image = image[:, :, 0:channels]
        return image

    def seg_rgb_to_values(self, seg_rgb):
        val_map = np.zeros((seg_rgb.shape[0], seg_rgb.shape[1]), dtype=np.uint8)
        for v , color in self.color_map.items():
            v_map = np.all(seg_rgb == tuple(color), axis=-1).astype(np.uint8) * v
            val_map += v_map
        return val_map

    def genLabels(self, map):

        val_map = np.zeros((map.shape[0], map.shape[1]))
        for x in range(map.shape[0]):
            for y in range(map.shape[1]):
                val_map[x,y] = self.val_map[tuple(map[x,y,:])]

        val_map = val_map.astype(np.uint8)

        return val_map


    def _on_loop(self):
        self._timer.tick()
        self.save_timer.tick()

        loc = self.client.getCarState()
        pos = loc.kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.y_val])

        if show_segmentation:
          responses = self.client.simGetImages([
              airsim.ImageRequest("0", airsim.ImageType.Scene, False,
                                  False), airsim.ImageRequest("0", airsim.ImageType.Segmentation, False,
                              False)])
        else:
          responses = self.client.simGetImages([
              airsim.ImageRequest("0", airsim.ImageType.Scene, False,
                                  False)])

        if len(responses) > 0:
            rgb = self.response_to_cv(responses[0], 3)

            # crop_img = rgb[ADDITIONAL_CROP_TOP:, :]
            # rgb = cv2.resize(crop_img, (640, 85))
            self._main_image = (rgb)

        # Did we get a segmentation image?
        if len(responses) > 1:
            seg = self.response_to_cv(responses[1], 3)
            self._seg_image = seg

        if self.recording and len(responses)>1:
            # if self.save_timer.elapsed_seconds_since_lap() > 0.2:
            # Record image every $grap_image_distance meters
            if np.linalg.norm(self.last_pos - pos) > grab_image_distance:
                    if self.record_path is not None:
                        cv2.imwrite(self.record_path + 'image_' + str(self.save_counter) + '.png', cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
                        cv2.imwrite(self.record_path + 'seg_' + str(self.save_counter) + '.png', cv2.cvtColor(seg,cv2.COLOR_BGR2RGB))
                        self.save_counter += 1
                    self.last_pos = pos

        self.counter += 1

        # Print FPS
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            print("FPS: %d" % self.counter)
            self.counter = 0
            self._timer.lap()

        if self.control_mode == "keyboard":
            # Get Control from keyboard
            control = self._get_keyboard_control(pygame.key.get_pressed())
        else:
            # Get control from joystick
            control =  self._get_joystick_control(self.joysticks[0],
                                                  pygame.key.get_pressed(),
                                                  self.control_mode)
        # Apply control
        if control is None:
            self._on_new_episode()
        else:
            self.car_controls.throttle = control.throttle
            self.car_controls.steering = control.steer
            self.car_controls.brake = control.brake
            self.client.setCarControls(self.car_controls)

        pygame.display.update()

    def _get_joystick_control(self, joystick, keys, mode="left"):
        """ Returns a VehicleControl message based on joystick input. Return
            None if a new episode was requested.

            This assumes a Playstation DualShock3 controller is being used.

        Args:
            joystick (pygame.joystick.Joystick): The pygame joystick object
                                                 used as input
            mode (string): Available options are:
                           - "left": Only the left analog stick is used for
                                     control
                           - "right": Only the right analog stick is used for
                                      control
                           - "rc": Mimics an rc car, where the left stick is
                                   used to control throttle and the right stick
                                   controls steering
                           - "game": Mimics default video game bindings, with
                                     throttle and brake controls bound to the
                                     triggers. Only available when using a PS4
                                     controller
            keys (Pygame key): Keys for use with I have no clue tbh.
        """
        if keys[K_r]:
            return None
        ps3 = "PLAYSTATION(R)3" in joystick.get_name()\
              or "PS3" in joystick.get_name()
        ps4 = "Wireless" in joystick.get_name()

        if not (ps3 or ps4):
            raise Exception("Not designed for non-playstation"
                            "controllers")

        control = VehicleControl()

        # Button assignments
        # x defines the horizontal axis
        # y defines the vertical axis
        # l is defined as left
        # r is defined as right
        l_x = self._input_deadzone(joystick.get_axis(0))
        l_y = self._input_deadzone(joystick.get_axis(1))
        r_x = self._input_deadzone(joystick.get_axis(3))
        r_y = self._input_deadzone(joystick.get_axis(4))
        l2 = self._input_deadzone(joystick.get_axis(2))
        r2 = self._input_deadzone(joystick.get_axis(5))
        start_button = None  # start/stop recording
        l1 = None  # handbrakes
        r1 = None  # reverse enable

        if ps3:
            start_button = joystick.get_button(9)
            l1 = joystick.get_button(4)
            r1 = joystick.get_button(5)

        if ps4:
            start_button = joystick.get_button(9)
            l1 = joystick.get_button(4)
            r1 = joystick.get_button(5)

        control.hand_brake = l1
        control.reverse = r1 == 1

        if mode == "left":
            print("l_x: {0}".format(l_x))
            control.steer = l_x / 2
            # modify the vehicle control object for throttle and brake controls
            self._throttle_brake_combined(l_y, control)


        if mode == "right":
            control.steer = r_x / 2
            # modify the vehicle control object for throttle and brake controls
            self._throttle_brake_combined(r_y, control)

        if mode == "rc":
            control.steer = r_x / 2
            # modify the vehicle control object for throttle and brake controls
            self._throttle_brake_combined(l_y, control)

        if mode == "game":
            control.steer = l_x / 2
            control.throttle = (r2 + 1) / 4
            control.brake = (l2 + 1) / 4

        control.print_state()
        return control

        # TODO teach car how to drift
        # TODO implement recording
        # TODO implement button state to allow for recording with single button
        # tap

    def _input_deadzone(self, value, deadzone=0.02):
        """ Defines a deadzone for joystick input.

            Analog controls are often times noisy. This method sets a deadzone
            for Playstation controllers, no user input returns no controller
            input (as opposed to 0.019 input or thereabouts).

        Args:
            value (float): The raw controller value.
            deadzone (float): Deadzone value, defaults to 0.02.

        Returns:
            Input value with deadzone.
        """
        if (-1 * deadzone) < value < deadzone:
            return 0
        else:
            return value

    def _throttle_brake_combined(self, value, vehicle_control):
        """ Modifies the value of throttle or brake in vehicle_control.

        Args:
            value (float): A floating point value between -1 and 1.
            vehicle_control (VehicleControl): The VehicleControl object whose
                                              throttle and brake value are to be
                                              changed.
        """
        if value < 0 and not vehicle_control.reverse:
            vehicle_control.throttle = abs(value / 2)

        if value > 0 and not vehicle_control.reverse:
            vehicle_control.brake = value / 2

        if value > 0 and vehicle_control.reverse:
            vehicle_control.throttle = value / 2

        if value == 0:
            vehicle_control.throttle = 0
            vehicle_control.brake = 0

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        # Left/Right 0
        # Throttle 5

        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -0.7
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 0.7
        if keys[K_UP] or keys[K_w]:
            control.throttle = 0.3
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            if not self.recording:
                self.record_path = SAVE_DIR + strftime("%Y_%m_%d_%H:%M:%S", gmtime()) + '/'
                if not os.path.exists(self.record_path):
                    os.makedirs(self.record_path)
                self.recording = True
                self.save_counter = 0
                print('Recording on, saving to: %s' % self.record_path)
        if keys[K_z]:
            if self.recording:
                self.recording = False
                print('Recording off, saved to: %s' % self.record_path)

        return control

    # This function gets called everytime the screen needs to be rendered
    def _on_render(self):
        if self._main_image is not None:
            surface_main = pygame.surfarray.make_surface(self._main_image.swapaxes(0, 1))
            self._display.blit(surface_main, (0, 0))

            if show_segmentation:
                surface_seg = pygame.surfarray.make_surface(self._seg_image.swapaxes(0, 1))
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
