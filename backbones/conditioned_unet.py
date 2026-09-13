"""Type-conditioned U-Net -- the proposed method.

The same U-Net as backbones/unet.py, with two additions:

  * a type head on the encoder bottleneck, predicting meningioma / glioma /
    pituitary;
  * FiLM modulation of every decoder stage, driven by the type posterior.

Forward returns (segmentation_logits, type_logits), so the training loop can
apply a segmentation loss and a cross-entropy type loss together. Use
backbones.split_output() rather than unpacking by hand, so the runner treats
conditioned and unconditioned backbones the same way.

With condition="none" this is architecturally identical to the plain U-Net plus
an unused type head, and because FiLM initialises to the identity the two models
start from the same function. That makes "conditioned vs not" a clean
comparison rather than a comparison of two different initialisations.
"""

import torch
import torch.nn as nn

from conditioning import (CONDITION_MODES, NUM_TUMOR_TYPES, FiLM, TypeHead,
                          condition_vector)

from .unet import ConvBlock, EncoderBlock


def check():
    return True, f"torch {torch.__version__}"


class ConditionedDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.film = FiLM(cond_dim, out_channels)

    def forward(self, x, skip, cond):
        x = self.up(x)
        x = self.conv(torch.cat([x, skip], dim=1))
        return self.film(x, cond)


class ConditionedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, widths=(64, 128, 256, 512),
                 bottleneck=1024, num_types=NUM_TUMOR_TYPES,
                 condition="predicted", detach_condition=False):
        super().__init__()
        if condition not in CONDITION_MODES:
            raise ValueError(
                f"condition must be one of {CONDITION_MODES}, got {condition!r}.")

        self.condition = condition
        self.detach_condition = detach_condition
        self.num_types = num_types

        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoders.append(EncoderBlock(channels, width))
            channels = width

        self.bottleneck = ConvBlock(widths[-1], bottleneck)
        self.type_head = TypeHead(bottleneck, num_types)

        self.decoders = nn.ModuleList()
        channels = bottleneck
        for width in reversed(widths):
            self.decoders.append(
                ConditionedDecoderBlock(channels, width, width, num_types))
            channels = width

        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x, type_target=None):
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)
        type_logits = self.type_head(x)

        cond = condition_vector(self.condition, type_logits, type_target,
                                self.num_types, self.detach_condition)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip, cond)

        return self.head(x), type_logits          # seg logits, type logits


def build(in_channels=3, out_channels=1, img_size=None, **kwargs):
    """`img_size` is accepted and ignored -- see backbones/unet.py."""
    return ConditionedUNet(in_channels=in_channels, out_channels=out_channels,
                           **kwargs)
