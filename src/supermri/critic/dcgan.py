import torch.nn as nn

from supermri.critic.critic import register
from supermri.models.utils import get_conv_shape


@register("dcgan")
class DCGAN(nn.Module):
    """
    DCGAN discriminator
    reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    def __init__(self, args):
        super(DCGAN, self).__init__()
        h, w = args.input_shape[1:]
        in_channels = args.input_shape[0]
        out_channels = (
            args.num_filters
            if args.critic_num_filters is None
            else args.critic_num_filters
        )
        num_blocks = args.critic_num_blocks
        dropout = args.critic_dropout

        kernel_size = (3, 3)
        stride = (2, 2)
        padding = 1

        h, w = get_conv_shape(h, w, kernel_size, stride, padding)
        self.input_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.input_activation = nn.LeakyReLU()
        self.input_dropout = nn.Dropout2d(dropout)

        conv_blocks = []
        for i in range(num_blocks):
            new_h, new_w = get_conv_shape(h, w, kernel_size, stride, padding)
            if h <= kernel_size[0] or w <= kernel_size[1]:
                break
            in_channels, out_channels = out_channels, out_channels * 2
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.LeakyReLU(),
                    nn.Dropout2d(dropout),
                )
            )
            h, w = new_h, new_w
        self.conv_blocks = nn.ModuleList(conv_blocks)

        h, w = get_conv_shape(h, w, kernel_size, 1, 0)
        self.conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.activation = nn.LeakyReLU()

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=h * w, out_features=1)

    def forward(self, x):
        outputs = self.input_conv(x)
        outputs = self.input_activation(outputs)
        outputs = self.input_dropout(outputs)
        for i in range(len(self.conv_blocks)):
            outputs = self.conv_blocks[i](outputs)
        outputs = self.conv(outputs)
        outputs = self.activation(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        return outputs
