"""Tests for the boundary-step measure in shortcut_figure.py. Run directly:
python test_shortcut_figure.py

The measure backs Table 3 and Fig. 2 of the paper: a wrong value would
misstate how strongly an arm writes the label outline into the image. It is
checked against a step worked out by hand, against images with no step, and
against the augmentation arms themselves on a synthetic slice.
"""

import numpy as np

from augment import Augmenter
from shortcut_figure import boundary_step

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def square_mask(size=64, top=20, side=20):
    m = np.zeros((size, size), np.float32)
    m[top:top + side, top:top + side] = 1.0
    return m


@test
def test_step_matches_hand_value():
    # Image 0 outside, d inside a square covering fraction f of the pixels:
    # ring means are d and 0, the image sd is d * sqrt(f (1 - f)).
    m = square_mask()
    d = 3.0
    img = d * m
    f = m.mean()
    expected = 1.0 / np.sqrt(f * (1 - f))
    got = boundary_step(img, m)
    assert abs(got - expected) < 1e-4, (got, expected)


@test
def test_no_boundary_gives_none():
    img = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
    assert boundary_step(img, np.zeros((32, 32), np.float32)) is None
    assert boundary_step(img, np.ones((32, 32), np.float32)) is None


@test
def test_smooth_image_has_small_step():
    # A ramp across the whole image: the rings straddle the boundary 1 px apart.
    img = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    got = boundary_step(img, square_mask())
    assert got < 0.1, got


@test
def test_channels_and_mask_shape_do_not_matter():
    m = square_mask()
    img = np.random.default_rng(1).normal(size=(64, 64)).astype(np.float32) + 2 * m
    one = boundary_step(img, m)
    three = boundary_step(np.repeat(img[..., None], 3, -1), m[..., None])
    assert abs(one - three) < 1e-5, (one, three)


@test
def test_region_arm_marks_the_boundary_and_bounded_does_not():
    # A textured slice with no intensity difference at the mask: only an arm
    # that remaps the two regions separately can create a step there.
    rng = np.random.default_rng(2)
    base = rng.uniform(0.2, 0.8, size=(96, 96)).astype(np.float32)
    img = np.repeat(base[..., None], 3, -1)
    mask = np.zeros((96, 96, 1), np.float32)
    mask[30:66, 30:66] = 1.0
    med = {}
    for arm in ("standard", "global", "region", "region-bounded"):
        steps = []
        for seed in range(40):
            a, m = Augmenter(arm, seed=seed)(img, mask)
            s = boundary_step(a, m)
            if s is not None:
                steps.append(s)
        med[arm] = float(np.median(steps))
    assert med["region"] > 3 * max(med["standard"], med["global"]), med
    assert med["region-bounded"] < med["region"] / 2, med


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
