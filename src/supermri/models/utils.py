import torch
import typing as t
from torch import nn
from math import floor


def cgetattr(obj, attr: str):
    """Case-insensitive getattr"""
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


def activation(name: str):
    """return activation layer with given name"""
    name = name.lower()
    if name == "elu":
        return nn.ELU
    elif name in ["leakyrelu", "lrelu", "leaky_relu"]:
        return nn.LeakyReLU
    elif name == "relu":
        return nn.ReLU
    elif name == "sigmoid":
        return nn.Sigmoid
    elif name == "tanh":
        return nn.Tanh
    elif name == "softmax":
        return nn.Softmax
    elif name == "gelu":
        return nn.GELU
    else:
        raise KeyError(f"activation layer {name} not found.")


def normalization(name: str):
    """return normalization layer with given name"""
    name = name.lower()
    if name in ["batchnorm", "batch_norm", "bn"]:
        return nn.BatchNorm2d
    elif name in ["instancenorm", "instance_norm", "in"]:
        return nn.InstanceNorm2d
    else:
        raise KeyError(f"normalization layer {name} not found.")


class Reshape(nn.Module):
    """Reshape layer"""

    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(self.target_shape)


def get_conv_shape(
    height: int,
    width: int,
    kernel_size: t.Union[int, t.Tuple[int, int]],
    stride: t.Union[int, t.Tuple[int, int]] = 1,
    padding: int = 0,
    dilation: int = 1,
) -> (int, int):
    """calculate Conv layer output shape"""
    if type(kernel_size) == int:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) == int:
        stride = (stride, stride)
    height = floor(
        ((height + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0])
        + 1
    )
    width = floor(
        ((width + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1])
        + 1
    )
    return height, width
