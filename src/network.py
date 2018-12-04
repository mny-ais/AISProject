# -*- coding: utf-8 -*-
"""Network.

A neural-network based on conditional imitation learning that is capable of
self-driving. Based on the paper "End-to-end Driving via Conditional Imitation
Learning" by Codevilla et al.

Authors:
    Maximilian Roth
    Nina Pant
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>

References:
    End-to-end Driving via Conditional Imitation Learning
    arXiv:1710.02410v2 [cs.RO] 2 Mar 2018
"""

import torch

import torchvision


class Network:
    def __init__(self):
        """Neural Network for Self Driving."""