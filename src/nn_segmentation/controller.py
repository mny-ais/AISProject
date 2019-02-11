# -*- coding: utf-8 -*-
"""Controller.

This module is used to send commands to the AirSim simulation in Unreal Engine
from the network.

Authors:
    Maximilian Roth
    Nina Pant
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>

"""
import airsim
import cv2
import pygame
import torch
import numpy as np

from nn_segmentation.segmentation_network import SegmentationNetwork
from nn_segmentation.fc_drive import FCD

from utils.timer import Timer
from pygame.locals import K_KP4
from pygame.locals import K_KP6
from pygame.locals import K_KP8
from pygame.locals import K_q
from pygame.locals import K_SPACE


# Define global variables
AIRSIM_WIDTH = 320
AIRSIM_HEIGHT = 64
ADDITIONAL_CROP_TOP = 0

WINDOW_WIDTH = AIRSIM_WIDTH
WINDOW_HEIGHT = AIRSIM_HEIGHT + 99  # Space to put text below the camera view

GRAB_IMAGE_DISTANCE = 0.1  # Meters

MAX_THROTTLE_ONLY = True  # Run the car at max speed and ignore model throttle

max_lanes = 6


class Controller:
    def __init__(self, weight1, weight2):
        """Acts as a controller object that sends and receives data from AirSim.

            This class acts as the interface between the network and the AirSim
            simulation inside of Unreal Engine. It is able to receive camera
            images from AirSim as well as send the driving commands to it and
            is responsible for running the network.

            Control Scheme:
                NUM_4   : Left
                NUM_8   : Forwards
                NUM_6   : Right
                Q       : Quit
                SPACE   : Reset

        Args:
            weight1 (str): Path to the segmentation network weights
            weight2 (str): Path to the fc network weights
        """
        # Initialize AirSim connection
        self.client = airsim.CarClient()
        self.client.confirmConnection()

        # Set up the network
        self.seg_net = SegmentationNetwork("googlenet")
        self.seg_net.cuda()
        if (weight2 is None) or (weight2 != ""):
            self.seg_only = True
        else:
            self.fcd = FCD()
            self.fcd.cuda()
            self.seg_only = False
            self.client.enableApiControl(True)

        self.seg_out = None

        self.weight1 = weight1
        self.weight2 = weight2

        self.device = torch.device("cuda")

        # Set up timers for fps counting
        self._timer = Timer()
        self.counter = 0

        # Set up display variables
        self._display = None
        self._main_image = None
        self._overlay_image = None
        self.fps_text = "FPS: 0"
        self.direction_text = "Direction: Forwards"
        self._text_font = None

        self.set_segmentation_ids()

        # Directions:
        # -1 : Left
        # 0 : Forwards
        # 1 : Right
        self._direction = 0  # Direction defaults to forwards

        self.out = None  # Network output
        self.throttle = 0  # Throttle output

        self.max_throttle = 0.35  # Throttle limit

        # Quitting
        self._request_quit = False

    def execute(self):
        """"Launch PyGame."""
        pygame.init()
        self.__init_game()

        # Initialize fonts
        if not pygame.font.get_init():
            pygame.font.init()

        self._text_font = pygame.font.SysFont("helvetica", 24)

        while not self._request_quit:
            self.__on_loop()
            self.__on_render()

        if self._request_quit:
            pygame.display.quit()
            pygame.quit()
            self.client.enableApiControl(False)  # Give control back to user
            return

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

    def __init_game(self):
        """Initializes the PyGame window and creates a new episode.

            This is separate from the main init method because the init is
            intended to create an instance of the class, but not to start
            the game objects yet.
        """
        self.__on_reset()

        self._display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT),
                                                pygame.HWSURFACE
                                                | pygame.DOUBLEBUF)

        self.seg_net.load_state_dict(torch.load(self.weight1))
        if not self.seg_only:
            self.fcd.load_state_dict(torch.load(self.weight2))

        print("PyGame started")

    def __on_reset(self):
        """Resets the state of the client."""
        print("Resetting client")
        self.client.reset()
        self._timer = Timer()

    def __on_loop(self):
        """Commands to execute on every loop."""
        # Make time tick
        self._timer.tick()

        # Get an image from Unreal
        response = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
        rgb = None
        if response:
            rgb = self.__response_to_cv(response[0], 3)
            self._main_image = rgb

        # Get key presses and parse them
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.KEYDOWN:
                self.__parse_event(event)
            if event.type == pygame.QUIT:
                self._request_quit = True

        # Run the network
        # First convert the images to tensors
        rgb = self.__to_tensor(rgb).float().to(self.device)

        self.seg_out = self.seg_net.forward(torch.unsqueeze(rgb, 0))
        self._overlay_image = self.seg_out.cpu().detach()[0].numpy()
        self._overlay_image = self._overlay_image.transpose(1, 2, 0)

        if not self.seg_only:
            # Flatten
            x = self.seg_out.view(-1, self._num_flat_features(self.seg_out))

            # Analyze for steering
            x = self.fcd(x, [0, [self._direction]])

            # get its data, then to numpy, then to a tuple
            self.out = tuple(x.cpu().detach().numpy())

            # Now send the command to airsim
            if MAX_THROTTLE_ONLY:
                self.throttle = self.max_throttle
            else:
                self.throttle = self.out[0][1]
            self.__send_command((self.out[0][0], self.throttle))

        # Computation is now complete. Add to the counter.
        self.counter += 1

        # Determine then update direction
        if self._direction == 0:
            direction = "Forward"
        elif self._direction == -1:
            direction = "Left"
        else:
            direction = "Right"
        self.direction_text = "Direction: {0}".format(direction)

        # Update FPS
        if self._timer.elapsed_seconds_since_lap() > 0.3:
            # Determine FPS
            fps = float(self.counter) / self._timer.elapsed_seconds_since_lap()

            # Update the info
            self.fps_text = "FPS: {0}".format(int(fps))

            # Reset counters
            self.counter = 0
            self._timer.lap()

        pygame.display.update()  # Finally, update the display.

    def __on_render(self):
        """Renders the pygame window itself."""
        if self._main_image is not None and self._overlay_image is not None:
            # If there is an image in the pipeline, render it.
            img = self.__overlay_images(self._main_image, self._overlay_image)
            surface_main = pygame.surfarray.make_surface(
                img.swapaxes(0, 1))
            self._display.blit(surface_main, (0, 0))

        # Render white fill to "reset" the text
        self._display.fill((255, 255, 255),
                           rect=pygame.Rect(0, 64, WINDOW_WIDTH,
                                            WINDOW_HEIGHT - 64))

        # Create the text in the window
        surface_fps = self._text_font.render(self.fps_text, True,
                                             (0, 0, 0))
        surface_direction = self._text_font.render(self.direction_text, True,
                                                   (0, 0, 0))

        if self.out is None:
            self.out = (np.array([0], dtype="float32"),)
        surface_steering = self._text_font.render("Steering: %.2f"
                                                  % self.out[0][0], True,
                                                  (0, 0, 0))
        surface_throttle = self._text_font.render("Throttle: %.2f"
                                                  % self.throttle, True,
                                                  (0, 0, 0))

        # And now render that text
        self._display.blit(surface_fps, (6, 70))
        self._display.blit(surface_direction, (120, 70))
        self._display.blit(surface_steering, (6, 100))
        self._display.blit(surface_throttle, (6, 130))

    @staticmethod
    def __response_to_cv(r, channels):
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

    @staticmethod
    def __overlay_images(image1, image2):
        """Overlays using linear dodge.

        Args:
            image1 (np.ndarray): First image.
            image2 (np.ndarray): Second image. The one that will become the
            overlay.

        Returns:
            (np.ndarray) The overlayed image
        """
        shape = image1.shape
        output = np.ndarray([shape[0], shape[1], shape[2]])
        for i in range(shape[0]):
            for j in range(shape[1]):
                img2_color_array = [0, 0, 0]
                index_max = np.argmax(image2[i][j])
                if index_max == 1:
                    img2_color_array = np.array([0, 0, 255])
                elif index_max == 2:
                    img2_color_array = np.array([255, 0, 0])
                for k in range(shape[2]):
                    output[i][j][k] = max(image1[i][j][k], img2_color_array[k])

        return output

    def __parse_event(self, event):
        """Parses PyGame events.

        Args:
            event (pygame.Event): The PyGame event to be parsed.
        """
        if event.key == K_KP8:
            self._direction = 0
        elif event.key == K_KP4:
            self._direction = -1
        elif event.key == K_KP6:
            self._direction = 1
        elif event.key == K_SPACE:
            self.__on_reset()
        elif event.key == K_q:
            self._request_quit = True

    def __send_command(self, command):
        """Sends driving commands over the AirSim API.

        Args:
            command (tuple): A tuple in the form (steering, throttle).
        """
        car_control = airsim.CarControls()
        car_control.steering = float(command[0])
        car_control.throttle = command[1]
        self.client.setCarControls(car_control)

    @staticmethod
    def __to_tensor(image):
        """Turns an image into a tensor

        Args:
            image: The image to be converted as an array of uncompressed bits.

        Returns:
            (torch.Tensor) the image as a tensor.
        """
        image = image.transpose(2, 0, 1)

        return torch.from_numpy(image)

    @staticmethod
    def _num_flat_features(x):
        """Multiplies the number of features for flattening a convolution.

        References:
            https://pytorch.org/tutorials/beginner/blitz/neural_networks_
            tutorial.html
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
