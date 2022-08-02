import numpy as np
import torch.nn as nn

from supermri.critic.critic import register


@register("dense")
class Dense(nn.Module):
    """
    Single dense layer discriminator
    """

    def __init__(self, args):
        super(Dense, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=int(np.prod(args.input_shape)), out_features=1
        )
        self.sigmoid = None
        if args.output_logits:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = self.flatten(x)
        outputs = self.linear(outputs)
        if self.sigmoid is not None:
            outputs = self.sigmoid(outputs)
        return outputs
