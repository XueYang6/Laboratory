import torch
import torch.nn as nn
from torch import Tensor

from torchvision.utils import _log_api_usage_once

"""
This is a modified version of torchvision's code for instantiating resnets. Here's a list of the changes made to the 
source code:
    - All convolutional layers now have bias set to True, where they were original set to False.
    - Removed the first maxpool layers so that input stays somewhat large.
    - Layer conv1 in the ResNet class has kernel size set to 3 and stride set to 1, where they were originally 7 and 2,
      respectively. 
    - Forward calls have a feature list argument to store the features of the network. This is only used for continual 
      backprop and doesn't affect the output of the network.

"""

class SequentialWithKeywordArguments(nn.Sequential):
    """
    Sequential module that allows the use of keyword arguments in the forward pass. 
    """

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input



