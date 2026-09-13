"""Tests for the backbones package. Run directly: python test_backbones.py

Requires torch; the SwinUNETR tests additionally require monai and are skipped
with a note when it is absent. The Swin-UMamba tests check the *diagnostics*,
not the model, so they pass whether or not mamba_ssm is installed.
"""

import torch

import backbones
from backbones import (FAMILIES, NAMES, available, build, count_parameters,
                       family_of, split_output)

RESULTS = []
SKIPPED = []


def test(fn):
    RESULTS.append(fn)
    return fn


class Skip(Exception):
    pass


def need_monai():
    ok, detail = available("swin_unetr")["swin_unetr"]
    if not ok:
        raise Skip(detail)


# ------------------------------------------------------------- registry

@test
def registry_is_self_consistent():
    assert set(NAMES) == set(backbones.REGISTRY), "NAMES and REGISTRY disagree"
    for name in NAMES:
        assert family_of(name) in FAMILIES, (name, family_of(name))
    covered = {family_of(n) for n in NAMES}
    assert covered == set(FAMILIES), f"registry misses a family: {covered}"


@test
def unknown_backbone_is_rejected_clearly():
    for call in (lambda: build("resnet"), lambda: family_of("resnet"),
                 lambda: available("resnet")):
        try:
            call()
        except ValueError as exc:
            assert "resnet" in str(exc) and "Available" in str(exc), exc
            continue
        raise AssertionError("an unknown backbone name was accepted")


@test
def available_reports_without_raising():
    """A broken Mamba toolchain must not stop you inspecting the other arms."""
    status = available()
    assert set(status) == set(NAMES)
    for name, (ok, detail) in status.items():
        assert isinstance(ok, bool), name
        assert detail and isinstance(detail, str), f"{name} gave no detail"


# ----------------------------------------------------------------- unet

@test
def unet_preserves_spatial_size():
    model = build("unet", in_channels=3, out_channels=1).eval()
    for size in (192, 256):
        with torch.no_grad():
            out = model(torch.randn(2, 3, size, size))
        assert out.shape == (2, 1, size, size), (size, out.shape)


@test
def backbones_emit_logits_not_probabilities():
    """The TensorFlow original ended in a sigmoid; pairing that with a loss that
    applies its own is the bug that has already cost this project a run. Every
    backbone in the registry owes the same logits contract.

    Tested by amplifying the final convolution rather than by sampling output
    values: at random initialisation a genuine logit head still produces an
    all-positive output about 12% of the time, so a "some value is negative"
    assertion is flaky. An amplified logit head is unbounded; a sigmoid head
    saturates below 1 no matter how large its inputs get.
    """
    import torch.nn as nn

    for name in NAMES:
        ok, _ = available(name)[name]
        if not ok:
            continue
        size = 256
        torch.manual_seed(0)
        model = build(name, in_channels=3, out_channels=1, img_size=size).eval()

        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        assert convs, f"{name} has no Conv2d to amplify"
        head = convs[-1]

        with torch.no_grad():
            head.weight.mul_(1000.0)
            if head.bias is not None:
                head.bias.mul_(1000.0)
            out, _ = split_output(model(torch.randn(1, 3, size, size)))

        assert float(out.abs().max()) > 1.0, (
            f"{name} output saturated at {float(out.abs().max()):.4f} -- "
            f"looks like a bounded activation on the output")


@test
def unet_matches_the_tensorflow_parameter_count():
    """The TF model reported 31,043,521 trainable parameters. The port is meant
    to be the same architecture, so a mismatch means it drifted."""
    model = build("unet", in_channels=3, out_channels=1)
    trainable = count_parameters(model, trainable_only=True)
    assert trainable == 31_043_521, (
        f"expected 31,043,521 trainable parameters, got {trainable:,}")


@test
def every_backbone_accepts_the_shared_config():
    """The runner passes one config to all arms; none may reject a common key."""
    config = dict(in_channels=3, out_channels=1, img_size=256)
    for name in NAMES:
        ok, detail = available(name)[name]
        if not ok:
            continue
        model = build(name, **config)
        assert sum(p.numel() for p in model.parameters()) > 0, name


@test
def unet_channel_counts_are_configurable():
    model = build("unet", in_channels=1, out_channels=3).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64))
    assert out.shape == (1, 3, 64, 64), out.shape


@test
def unet_gradients_flow_to_every_parameter():
    model = build("unet")
    out = model(torch.randn(2, 3, 64, 64))
    out.mean().backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not missing, f"no gradient reached: {missing[:5]}"


# ----------------------------------------------------------- swin_unetr

@test
def swin_unetr_preserves_spatial_size():
    need_monai()
    model = build("swin_unetr", in_channels=3, out_channels=1, img_size=256).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1, 256, 256), out.shape


@test
def swin_unetr_rejects_sizes_it_cannot_downsample():
    need_monai()
    try:
        build("swin_unetr", img_size=250)
    except ValueError as exc:
        assert "divisible by 32" in str(exc), exc
        return
    raise AssertionError("img_size=250 was accepted")


@test
def swin_unetr_is_a_distinct_model_from_unet():
    need_monai()
    a = count_parameters(build("swin_unetr", img_size=256))
    b = count_parameters(build("unet"))
    assert a != b and a > 0, (a, b)


# ---------------------------------------------------------- swin_umamba

@test
def swin_umamba_diagnoses_rather_than_crashes():
    ok, detail = available("swin_umamba")["swin_umamba"]
    if ok:
        return
    lowered = detail.lower()
    assert any(k in lowered for k in ("mamba_ssm", "cuda", "clone", "torch")), detail
    assert len(detail) > 40, f"diagnostic too thin to act on: {detail}"


@test
def swin_umamba_build_fails_with_the_fix_attached():
    ok, _ = available("swin_umamba")["swin_umamba"]
    if ok:
        return
    try:
        build("swin_umamba")
    except ImportError as exc:
        assert "unavailable" in str(exc), exc
        assert "swin_umamba.py" in str(exc), "error should point at the setup notes"
        return
    raise AssertionError("build succeeded despite check() reporting unavailable")


# ------------------------------------------------------------- utilities

@test
def count_parameters_distinguishes_trainable_from_frozen():
    model = build("unet")
    total = count_parameters(model, trainable_only=False)
    for p in model.parameters():
        p.requires_grad_(False)
    assert count_parameters(model, trainable_only=True) == 0
    assert count_parameters(model, trainable_only=False) == total


if __name__ == "__main__":
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"  pass  {fn.__name__}")
        except Skip as exc:
            SKIPPED.append(fn.__name__)
            print(f"  skip  {fn.__name__}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    passed = len(RESULTS) - failed - len(SKIPPED)
    tail = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\n{passed}/{len(RESULTS)} passed{tail}")
    raise SystemExit(1 if failed else 0)
