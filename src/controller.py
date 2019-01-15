# -*- coding: utf-8 -*-
"""Controller.

This module is used to send commands to the AirSim simulation in Unreal Engine.

Authors:
    Maximilian Roth
    Nina Pant
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>

"""
class Controller:
    def __init__(self):
        """Acts as a controller object that sends and receives data from AirSim.

            This class acts as the interface between the network and the AirSim
            simulation inside of Unreal Engine. It is able to receive camera
            images from AirSim as well as send the driving commands to it.
        """

    def receive_image(self):
        """Receive images from the AirSim API.

        Returns:
            (torch.Tensor) A tensor of the image from the front camera of the
            vehicle.
        """
        raise NotImplementedError

    def send_command(self, command):
        """Sends driving commands over the AirSim API.

        Args:
            command (tuple): A tuple in the form (steering, throttle).
        """
        raise NotImplementedError
