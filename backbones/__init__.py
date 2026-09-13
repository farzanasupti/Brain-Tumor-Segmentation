"""Backbone registry for the backbone x augmentation factorial.

Every backbone is reached through one interface so the experiment runner never
special-cases an architecture:

    import backbones
    model = backbones.build("unet", in_channels=3, out_channels=1)
    logits = model(x)            # x: (B, C, H, W) -> (B, out_channels, H, W)

Some backbones return an auxiliary output alongside the segmentation map --
the type-conditioned model returns (segmentation_logits, type_logits). Pass the
forward result through split_output() instead of unpacking by hand, so the
runner handles every backbone identically.

All backbones return **logits**, not probabilities. Pair them with
BCEWithLogitsLoss or a Dice loss that applies its own sigmoid, and apply sigmoid
exactly once at inference. Applying it in both the model and the loss squashes
gradients toward 0.5 and silently cripples training.

Why PyTorch
-----------
The repository's original U-Net, metrics and training loop are TensorFlow, but
the factorial cannot run there. SwinUNETR ships in MONAI, Swin-UMamba exists
only as PyTorch, and mamba_ssm is a set of CUDA kernels bound to PyTorch with no
TensorFlow equivalent. Since the Mamba arm is the one the whole question is
framed around, PyTorch is the framework the experiment has to live in.

The framework-agnostic modules carry over untouched: prepare_dataset.py,
splits.py, folds.py and augment.py are numpy and stdlib only.

Checking what this environment can run:

    python -m backbones
"""

import importlib

#: name -> (module, family, human description)
REGISTRY = {
    "unet": ("backbones.unet", "cnn",
             "U-Net, 4 encoder blocks 64-512, 1024 bottleneck"),
    "swin_unetr": ("backbones.swin_unetr", "transformer",
                   "SwinUNETR (MONAI), 2D, shifted-window transformer encoder"),
    "swin_umamba": ("backbones.swin_umamba", "mamba",
                    "Swin-UMamba, selective state-space U-Net"),
    "cond_unet": ("backbones.conditioned_unet", "cnn",
                  "Type-conditioned U-Net: FiLM decoder driven by a tumor-type head"),
}

NAMES = tuple(REGISTRY)
FAMILIES = ("cnn", "transformer", "mamba")


def family_of(name):
    """Which model family a backbone belongs to -- the grouping factor in the
    analysis, so it comes from the registry rather than being retyped."""
    _require_known(name)
    return REGISTRY[name][1]


def _require_known(name):
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown backbone {name!r}. Available: {', '.join(NAMES)}."
        )


def build(name, in_channels=3, out_channels=1, **kwargs):
    """Instantiate a backbone by name.

    Raises ImportError with actionable text when a backbone's dependencies are
    missing, rather than failing deep inside a training run.
    """
    _require_known(name)
    module = importlib.import_module(REGISTRY[name][0])
    return module.build(in_channels=in_channels, out_channels=out_channels, **kwargs)


def available(name=None):
    """Report which backbones can actually be built here.

    Returns {name: (ok, detail)}. Import failures are captured, not raised, so
    a missing Mamba toolchain does not stop you inspecting the CNN arm.
    """
    names = (name,) if name else NAMES
    out = {}
    for n in names:
        _require_known(n)
        try:
            module = importlib.import_module(REGISTRY[n][0])
            ok, detail = module.check()
        except Exception as exc:                    # noqa: BLE001 - reported, not raised
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        out[n] = (ok, detail)
    return out


def split_output(output):
    """Normalise a backbone's forward result to (segmentation_logits, aux).

    `aux` is None for plain backbones and the type logits for the conditioned
    one, so the training loop can add an auxiliary loss when there is one
    without branching on the architecture.
    """
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError(
                f"expected (segmentation_logits, aux) from the backbone, got a "
                f"{type(output).__name__} of length {len(output)}.")
        return output[0], output[1]
    return output, None


def count_parameters(model, trainable_only=True):
    """Parameter count, for the matched-budget reporting in the protocol."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _main():
    status = available()
    width = max(len(n) for n in NAMES)

    print(f"{'backbone':<{width}}  {'family':<12} {'status':<12} detail")
    print("-" * (width + 60))
    for n in NAMES:
        ok, detail = status[n]
        _, family, _ = REGISTRY[n]
        mark = "ready" if ok else "unavailable"
        print(f"{n:<{width}}  {family:<12} {mark:<12} {detail}")

    usable = [n for n in NAMES if status[n][0]]
    families = {family_of(n) for n in usable}
    print(f"\n{len(usable)}/{len(NAMES)} backbones usable, "
          f"covering {len(families)}/{len(FAMILIES)} families.")
    if len(families) < len(FAMILIES):
        missing = [f for f in FAMILIES if f not in families]
        print(f"Missing {', '.join(missing)} -- the factorial cannot estimate a "
              f"backbone x augmentation interaction across families without them.")


if __name__ == "__main__":
    _main()
