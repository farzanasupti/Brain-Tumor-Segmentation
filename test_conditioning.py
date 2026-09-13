"""Tests for conditioning.py and the type-conditioned U-Net.

Run directly: python test_conditioning.py  (requires torch)

The properties tested here are the ones that decide whether the method is
credible rather than merely runnable: that conditioning starts as a no-op, that
it has real capacity once trained, and that the oracle setting cannot be used
by accident.
"""

import torch
import torch.nn as nn

import backbones
from backbones import build, count_parameters, split_output
from conditioning import (CONDITION_MODES, NUM_TUMOR_TYPES, FiLM, TypeHead,
                          condition_vector)

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def wake_film(model, scale=0.5, seed=0):
    """Give the FiLM projections non-zero weights, as training would.

    At initialisation they are exactly zero, so conditioning is a no-op and any
    test of its *effect* would trivially pass.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, FiLM):
                module.project.weight.normal_(0.0, scale)
                module.project.bias.normal_(0.0, scale)
    return model


# ------------------------------------------------------------------ FiLM

@test
def film_is_exactly_the_identity_at_initialisation():
    """So a conditioned model and its unconditioned twin start as the same
    function, and any measured gain comes from conditioning, not from a
    different starting point."""
    film = FiLM(NUM_TUMOR_TYPES, 16)
    x = torch.randn(4, 16, 8, 8)
    for cond in (torch.zeros(4, 3), torch.rand(4, 3), torch.eye(3)[[0, 1, 2, 0]]):
        assert torch.equal(film(x, cond), x), "FiLM perturbed its input at init"


@test
def film_modulates_once_its_weights_are_non_zero():
    film = FiLM(NUM_TUMOR_TYPES, 16)
    torch.manual_seed(0)
    with torch.no_grad():
        film.project.weight.normal_(0, 0.5)
        film.project.bias.normal_(0, 0.5)
    x = torch.randn(2, 16, 8, 8)
    a = film(x, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    b = film(x, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    assert not torch.allclose(a, b), "different types produced identical features"


@test
def film_applies_per_channel_not_per_pixel():
    """A scale/shift must be constant across space within a channel."""
    film = wake_film(nn.ModuleList([FiLM(3, 8)]))[0]
    x = torch.ones(1, 8, 5, 5)
    out = film(x, torch.tensor([[1.0, 0.0, 0.0]]))
    for c in range(8):
        channel = out[0, c]
        assert torch.allclose(channel, channel.flatten()[0].expand_as(channel)), \
            f"channel {c} varies across space for a constant input"


# ------------------------------------------------------- condition vector

@test
def oracle_without_a_target_is_refused_with_an_explanation():
    logits = torch.randn(2, 3)
    try:
        condition_vector("oracle", logits, None)
    except ValueError as exc:
        assert "leaks" in str(exc) and "predicted" in str(exc), exc
        return
    raise AssertionError("oracle conditioning ran without a target")


@test
def one_based_dataset_labels_are_caught():
    """The dataset labels types 1/2/3; one-hot needs 0/1/2. Silently accepting
    1-based labels would shift every condition by one class."""
    logits = torch.randn(3, 3)
    try:
        condition_vector("oracle", logits, torch.tensor([1, 2, 3]))
    except ValueError as exc:
        assert "subtract 1" in str(exc), exc
        return
    raise AssertionError("1-based labels were accepted as class indices")


@test
def condition_modes_produce_the_expected_vectors():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])

    none = condition_vector("none", logits)
    assert torch.equal(none, torch.zeros(2, 3)), none

    oracle = condition_vector("oracle", logits, torch.tensor([2, 0]))
    assert torch.equal(oracle, torch.tensor([[0., 0., 1.], [1., 0., 0.]])), oracle

    predicted = condition_vector("predicted", logits)
    assert torch.allclose(predicted.sum(1), torch.ones(2)), "posterior must sum to 1"
    assert predicted.argmax(1).tolist() == [0, 2], predicted


@test
def detach_controls_whether_gradient_reaches_the_type_head():
    logits = torch.randn(2, 3, requires_grad=True)
    condition_vector("predicted", logits, detach=False).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0

    logits2 = torch.randn(2, 3, requires_grad=True)
    out = condition_vector("predicted", logits2, detach=True)
    assert not out.requires_grad, "detach=True still carried gradient"


@test
def unknown_condition_mode_is_rejected():
    try:
        condition_vector("groundtruth", torch.randn(1, 3))
    except ValueError as exc:
        assert str(CONDITION_MODES) in str(exc) or "predicted" in str(exc), exc
        return
    raise AssertionError("an unknown condition mode was accepted")


# -------------------------------------------------------- conditioned net

@test
def conditioned_unet_returns_segmentation_and_type_logits():
    model = build("cond_unet", in_channels=3, out_channels=1).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    seg, aux = split_output(out)
    assert seg.shape == (2, 1, 64, 64), seg.shape
    assert aux.shape == (2, NUM_TUMOR_TYPES), aux.shape


@test
def split_output_handles_both_backbone_shapes():
    plain = build("unet").eval()
    cond = build("cond_unet").eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64)
        seg_a, aux_a = split_output(plain(x))
        seg_b, aux_b = split_output(cond(x))
    assert aux_a is None and aux_b is not None
    assert seg_a.shape == seg_b.shape

    try:
        split_output((1, 2, 3))
    except ValueError:
        return
    raise AssertionError("a 3-tuple was accepted as a backbone output")


@test
def at_init_the_predicted_type_cannot_change_the_segmentation():
    """Identity-initialised FiLM means conditioning is inert until trained."""
    torch.manual_seed(0)
    model = build("cond_unet", condition="predicted").eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        seg_pred, _ = split_output(model(x))
        model.condition = "none"
        seg_none, _ = split_output(model(x))
    assert torch.allclose(seg_pred, seg_none, atol=1e-6), \
        "conditioning altered the output before FiLM was trained"


@test
def a_trained_film_makes_the_type_change_the_segmentation():
    """The capacity has to exist, or the method cannot do anything."""
    torch.manual_seed(0)
    model = wake_film(build("cond_unet", condition="oracle").eval())
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a, _ = split_output(model(x, type_target=torch.tensor([0, 0])))
        b, _ = split_output(model(x, type_target=torch.tensor([2, 2])))
    rel = (a - b).abs().mean() / a.abs().mean().clamp_min(1e-8)
    assert rel > 0.01, f"tumor type barely moved the output (relative {rel:.4f})"


@test
def oracle_mode_refuses_to_run_without_a_target():
    model = build("cond_unet", condition="oracle").eval()
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 64))
    except ValueError as exc:
        assert "oracle" in str(exc), exc
        return
    raise AssertionError("oracle mode ran with no ground-truth type")


@test
def predicted_mode_needs_nothing_beyond_the_image():
    """The honest setting must work with no label at inference."""
    model = build("cond_unet", condition="predicted").eval()
    with torch.no_grad():
        seg, aux = split_output(model(torch.randn(1, 3, 64, 64)))
    assert seg.shape == (1, 1, 64, 64) and aux.shape == (1, 3)


@test
def segmentation_loss_shapes_the_type_head_unless_detached():
    """Joint shaping is the point of conditioning on a soft posterior; the
    detach flag has to actually switch it off."""
    torch.manual_seed(0)
    joint = wake_film(build("cond_unet", condition="predicted"))
    seg, _ = split_output(joint(torch.randn(2, 3, 64, 64)))
    seg.mean().backward()
    grad = joint.type_head.fc.weight.grad
    assert grad is not None and grad.abs().sum() > 0, \
        "segmentation loss did not reach the type head"

    torch.manual_seed(0)
    cut = wake_film(build("cond_unet", condition="predicted", detach_condition=True))
    seg2, _ = split_output(cut(torch.randn(2, 3, 64, 64)))
    seg2.mean().backward()
    grad2 = cut.type_head.fc.weight.grad
    assert grad2 is None or grad2.abs().sum() == 0, \
        "detach_condition=True still let the segmentation loss reach the type head"


@test
def conditioning_overhead_is_small():
    """FiLM plus a type head must not quietly turn this into a bigger model,
    or 'conditioned beats plain' becomes 'more parameters beat fewer'."""
    plain = count_parameters(build("unet"))
    cond = count_parameters(build("cond_unet"))
    overhead = (cond - plain) / plain
    assert 0 < overhead < 0.02, \
        f"conditioning added {overhead:.2%} of parameters ({cond - plain:,})"


@test
def invalid_condition_mode_is_rejected_at_construction():
    try:
        build("cond_unet", condition="oracle_soft")
    except ValueError as exc:
        assert "condition must be one of" in str(exc), exc
        return
    raise AssertionError("an invalid condition mode was accepted")


@test
def type_head_reads_the_bottleneck_shape():
    head = TypeHead(1024, NUM_TUMOR_TYPES)
    out = head(torch.randn(3, 1024, 16, 16))
    assert out.shape == (3, NUM_TUMOR_TYPES), out.shape


if __name__ == "__main__":
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"  pass  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    raise SystemExit(1 if failed else 0)
