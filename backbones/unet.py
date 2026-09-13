"""U-Net, the CNN arm of the factorial.

A PyTorch port of the architecture in the repository's TensorFlow unet.py, kept
deliberately identical -- same four encoder stages at 64/128/256/512, same 1024
bottleneck, same 2x2 transposed-convolution decoder with concatenated skips --
so the control-arm result can be compared against the TensorFlow baseline and
against published U-Net numbers on this dataset.

One difference, on purpose: the output layer emits logits. The TensorFlow
version ended in a sigmoid, which is a trap when paired with a loss that applies
its own.
"""

import torch
import torch.nn as nn


def check():
    return True, f"torch {torch.__version__}"


class ConvBlock(nn.Module):
    """Conv-BN-ReLU twice, the repeating unit of the original architecture."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, widths=(64, 128, 256, 512),
                 bottleneck=1024):
        super().__init__()

        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoders.append(EncoderBlock(channels, width))
            channels = width

        self.bottleneck = ConvBlock(widths[-1], bottleneck)

        self.decoders = nn.ModuleList()
        channels = bottleneck
        for width in reversed(widths):
            self.decoders.append(DecoderBlock(channels, width, width))
            channels = width

        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)          # logits


def build(in_channels=3, out_channels=1, img_size=None, **kwargs):
    """`img_size` is accepted and ignored.

    U-Net is fully convolutional and infers its spatial size from the input, but
    the experiment runner passes one config to every backbone and the
    transformer and Mamba arms both need the size declared up front. Rejecting
    it here would force the runner to special-case architectures, which is the
    thing this registry exists to avoid.
    """
    return UNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
