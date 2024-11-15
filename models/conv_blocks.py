import torch
import torch.nn as nn


def conv1x1(in_channels: int, out_channels: int, stride: int=1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)

def conv_nxn(in_channels: int, out_channels: int, kernel_size: int=3, stride=1, padding=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
    """
    nxn convolution with padding.
    """

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        bias=True,
    )

