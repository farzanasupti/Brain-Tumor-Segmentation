"""Tumor-type conditioning for segmentation decoders.

Meningioma, glioma and pituitary tumors differ sharply in location, margin and
enhancement pattern, and the Cheng dataset labels every slice with its type.
Existing work on this dataset either ignores those labels or predicts them from
a shared encoder alongside segmentation; neither uses them to *steer* the
segmentation decoder. That is what this module does.

Mechanism
---------
FiLM (feature-wise linear modulation): a small linear map turns the type
posterior into a per-channel scale and shift applied to decoder features. It is
cheap, it touches every decoder stage rather than only the output, and it starts
as the exact identity so a conditioned model is initially indistinguishable from
its unconditioned counterpart -- any difference in the result is attributable to
conditioning rather than to a different starting point.

Where the type comes from
-------------------------
This is the part that decides whether the method is credible.

    "predicted"  a type head reads the encoder bottleneck and its softmax
                 posterior conditions the decoder. Nothing outside the image is
                 needed at inference. This is the honest setting and the one to
                 report as the method.

    "oracle"     the ground-truth type conditions the decoder. This leaks a
                 label the model would not have at test time, so it is an upper
                 bound, not a result. Report it as such -- the gap between
                 oracle and predicted says how much of the achievable gain the
                 type head actually captures.

    "none"       conditioning disabled, for the ablation arm. FiLM still runs,
                 on a constant vector, so parameter count and compute stay
                 matched with the conditioned arms.

Conditioning on a soft posterior rather than a hard argmax keeps the path
differentiable, so the segmentation loss also shapes the type representation.
Pass detach_condition=True to cut that, which is worth an ablation of its own:
it isolates whether joint shaping helps or simply destabilises the type head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CONDITION_MODES = ("predicted", "oracle", "none")

#: Cheng dataset labels: 1 meningioma, 2 glioma, 3 pituitary.
NUM_TUMOR_TYPES = 3


class FiLM(nn.Module):
    """Per-channel scale and shift predicted from a conditioning vector.

    Initialised to the identity: the projection starts at zero, so gamma is 1
    and beta is 0 on the first forward pass.
    """

    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.project = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x, cond):
        gamma, beta = self.project(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class TypeHead(nn.Module):
    """Tumor-type logits from the encoder bottleneck."""

    def __init__(self, in_channels, num_types=NUM_TUMOR_TYPES, dropout=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_types)

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        return self.fc(self.drop(pooled))


def condition_vector(mode, type_logits, type_target=None, num_types=NUM_TUMOR_TYPES,
                     detach=False):
    """Build the vector the decoder is conditioned on.

    `type_target` holds zero-based class indices, so subtract 1 from the
    dataset's 1/2/3 labels before calling.
    """
    if mode not in CONDITION_MODES:
        raise ValueError(
            f"condition must be one of {CONDITION_MODES}, got {mode!r}.")

    if mode == "none":
        return torch.zeros(type_logits.shape[0], num_types,
                           device=type_logits.device, dtype=type_logits.dtype)

    if mode == "oracle":
        if type_target is None:
            raise ValueError(
                "condition='oracle' needs type_target. It leaks the label and is "
                "an upper bound only -- use condition='predicted' for a result "
                "you can report as the method."
            )
        if type_target.min() < 0 or type_target.max() >= num_types:
            raise ValueError(
                f"type_target must hold indices in [0, {num_types - 1}]; got "
                f"[{int(type_target.min())}, {int(type_target.max())}]. The "
                f"dataset labels tumor types 1..3, so subtract 1 first."
            )
        return F.one_hot(type_target.long(), num_types).to(type_logits.dtype)

    posterior = F.softmax(type_logits, dim=1)
    return posterior.detach() if detach else posterior
