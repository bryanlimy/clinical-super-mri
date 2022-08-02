from .registry import register

import torch
from torch import nn


@register("identity")
def get_identity(args):
    return Identity(args)


class Identity(nn.Module):
    """
    Identity model
    """

    def __init__(self, args):
        super(Identity, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.weight + self.weight.detach()
