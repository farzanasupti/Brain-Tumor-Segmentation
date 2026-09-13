"""Losses and metrics for the PyTorch training loop.

Everything here takes **logits**, never probabilities, and applies sigmoid
exactly once internally. Feeding it a model that already ends in a sigmoid
applies the squash twice, damps gradients toward 0.5 and quietly cripples
training -- the failure that already cost this project a run.

The joint loss adds a cross-entropy term on the tumor type when the backbone
returns one, so conditioned and unconditioned models can share a training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOOTH = 1.0


def dice_loss(logits, targets, smooth=SMOOTH):
    """Soft Dice over the batch, computed per item and averaged.

    Per item rather than over the flattened batch: a batch-level Dice lets one
    large tumor dominate several small ones, which is the same averaging mistake
    as reporting slice-averaged instead of patient-averaged scores.
    """
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(1)
    denominator = probs.sum(1) + targets.sum(1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Equal-weight Dice + BCE, the segmentation loss fixed by the protocol."""

    def __init__(self, dice_weight=0.5, smooth=SMOOTH, pos_weight=None):
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError(f"dice_weight must be in [0,1], got {dice_weight}.")
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32),
        )

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight)
        dsc = dice_loss(logits, targets, self.smooth)
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dsc


class JointLoss(nn.Module):
    """Segmentation loss, plus a tumor-type term when the model predicts one.

    `type_weight` scales the auxiliary term. At 0 the type head still trains on
    nothing and the model is effectively unconditioned in its supervision, which
    is a distinct ablation from condition="none".
    """

    def __init__(self, type_weight=0.2, dice_weight=0.5, smooth=SMOOTH):
        super().__init__()
        if type_weight < 0:
            raise ValueError(f"type_weight must be >= 0, got {type_weight}.")
        self.segmentation = DiceBCELoss(dice_weight=dice_weight, smooth=smooth)
        self.type_weight = type_weight

    def forward(self, seg_logits, seg_targets, type_logits=None, type_targets=None):
        seg = self.segmentation(seg_logits, seg_targets)
        parts = {"segmentation": seg.detach()}

        if type_logits is None or type_targets is None or self.type_weight == 0:
            parts["type"] = torch.zeros((), device=seg.device)
            return seg, parts

        type_loss = F.cross_entropy(type_logits, type_targets)
        parts["type"] = type_loss.detach()
        return seg + self.type_weight * type_loss, parts


@torch.no_grad()
def binary_scores(logits, targets, threshold=0.5, eps=1e-7):
    """Per-item Dice and IoU from logits. Returns tensors of shape (B,)."""
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = (targets > 0.5).float().flatten(1)

    intersection = (preds * targets).sum(1)
    union = preds.sum(1) + targets.sum(1) - intersection
    denominator = preds.sum(1) + targets.sum(1)

    # Both empty is a correct prediction, and scoring it 0 would punish the
    # model for agreeing that a slice has no tumor.
    both_empty = denominator == 0
    dice = torch.where(both_empty, torch.ones_like(denominator),
                       (2.0 * intersection) / (denominator + eps))
    iou = torch.where(both_empty, torch.ones_like(union),
                      intersection / (union + eps))
    return dice, iou


@torch.no_grad()
def type_accuracy(type_logits, type_targets):
    if type_logits is None or type_targets is None:
        return None
    return (type_logits.argmax(1) == type_targets).float().mean().item()
