"""SwinUNETR, the transformer arm of the factorial.

A thin adapter over MONAI's implementation, configured for 2D single-image
segmentation. MONAI's constructor signature has changed across releases -- older
versions require img_size, newer ones dropped it -- so the wanted arguments are
filtered against the installed signature rather than assumed.

MONAI's SwinUNETR emits logits, so no activation is appended here.

Input size must be divisible by 32 for the shifted-window downsampling path to
line up; the protocol's 256 x 256 satisfies that, and 192 x 192 (the fallback in
the compute budget) does too.
"""

import inspect

DEFAULT_FEATURE_SIZE = 24


def check():
    try:
        import monai
        import torch
    except ImportError as exc:
        return False, (f"{exc.name} not installed. Run: pip install monai einops")

    try:
        from monai.networks.nets import SwinUNETR  # noqa: F401
    except Exception as exc:                        # noqa: BLE001
        return False, f"monai {monai.__version__} cannot provide SwinUNETR: {exc}"

    return True, f"monai {monai.__version__}, torch {torch.__version__}"


def build(in_channels=3, out_channels=1, img_size=256,
          feature_size=DEFAULT_FEATURE_SIZE, **kwargs):
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:
        raise ImportError(
            "SwinUNETR needs MONAI. Run: pip install monai einops"
        ) from exc

    if img_size % 32:
        raise ValueError(
            f"img_size must be divisible by 32 for SwinUNETR's window "
            f"downsampling, got {img_size}."
        )

    wanted = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "spatial_dims": 2,
        "feature_size": feature_size,
        "img_size": img_size,
        **kwargs,
    }

    # Only pass what this MONAI version actually accepts; img_size in
    # particular was removed in later releases.
    accepted = inspect.signature(SwinUNETR.__init__).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        dropped = sorted(set(wanted) - set(accepted))
        wanted = {k: v for k, v in wanted.items() if k in accepted}
        if dropped:
            # Informational: the model is still correctly configured without them.
            print(f"[swin_unetr] this MONAI version ignores: {', '.join(dropped)}")

    return SwinUNETR(**wanted)
