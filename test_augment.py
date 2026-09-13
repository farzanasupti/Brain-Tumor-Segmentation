"""Tests for augment.py. Run directly: python test_augment.py

No pytest dependency; numpy is required, OpenCV only for the geometric tests.
"""

import numpy as np

import augment
from augment import (ARMS, Augmenter, apply_transfer, bezier_global,
                     bezier_region_wise, sample_bezier_curve, standard_augment)

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def synthetic_slice(h=64, w=64, seed=0):
    """An image with a known structure: dark background, bright disc tumor."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((yy - h * 0.4) ** 2 + (xx - w * 0.55) ** 2) < (min(h, w) * 0.18) ** 2

    image = np.full((h, w), 0.25, dtype=np.float32)
    image[disc] = 0.75
    image += rng.normal(0, 0.02, size=(h, w)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    mask = disc.astype(np.float32)[..., None]
    return image[..., None], mask


# --------------------------------------------------------------- curve shape

@test
def curve_endpoints_are_pinned():
    """Monotonic curves run (0,0)->(1,1); reversed ones run (0,1)->(1,0)."""
    for seed in range(200):
        xs, ys = sample_bezier_curve(seed, monotonic=True)
        assert (xs[0], ys[0]) == (0.0, 0.0), (xs[0], ys[0])
        assert (xs[-1], ys[-1]) == (1.0, 1.0), (xs[-1], ys[-1])

        xs, ys = sample_bezier_curve(seed, monotonic=False)
        assert (xs[0], ys[0]) == (0.0, 1.0), (xs[0], ys[0])
        assert (xs[-1], ys[-1]) == (1.0, 0.0), (xs[-1], ys[-1])


@test
def curve_x_is_strictly_increasing():
    """np.interp gives undefined results otherwise."""
    for seed in range(300):
        for monotonic in (True, False):
            xs, _ = sample_bezier_curve(seed, monotonic=monotonic)
            assert np.all(np.diff(xs) > 0), f"seed={seed} has a tie or reversal"


@test
def curve_stays_in_unit_range():
    """Convex hull property: no clipping should ever be needed."""
    for seed in range(300):
        for monotonic in (True, False):
            xs, ys = sample_bezier_curve(seed, monotonic=monotonic)
            assert ys.min() >= -1e-9 and ys.max() <= 1 + 1e-9, (ys.min(), ys.max())
            assert xs.min() >= -1e-9 and xs.max() <= 1 + 1e-9


@test
def monotonic_regime_preserves_order():
    probe = np.linspace(0, 1, 512)
    for seed in range(300):
        xs, ys = sample_bezier_curve(seed, monotonic=True)
        out = apply_transfer(probe, xs, ys)
        assert np.all(np.diff(out) >= -1e-6), f"seed={seed} inverted an ordering"


@test
def non_monotonic_regime_reverses_order():
    """The second regime must genuinely differ from the first, or the two
    intensity regimes are the same experiment."""
    probe = np.linspace(0, 1, 512)
    for seed in range(300):
        xs, ys = sample_bezier_curve(seed, monotonic=False)
        out = apply_transfer(probe, xs, ys)
        assert np.all(np.diff(out) <= 1e-6), f"seed={seed} was not order-reversing"
        assert out[0] > out[-1], "reversed curve did not invert the range"


@test
def transfer_clamps_out_of_range_input():
    xs, ys = sample_bezier_curve(0, monotonic=True)
    out = apply_transfer(np.array([-5.0, 0.0, 1.0, 5.0]), xs, ys)
    assert out[0] == 0.0 and out[-1] == 1.0, out


@test
def constant_image_maps_to_a_constant():
    image = np.full((16, 16, 1), 0.3, dtype=np.float32)
    out = bezier_global(image, rng=7)
    assert np.allclose(out, out.flat[0]), "a flat image gained structure"


# ------------------------------------------------------------- region-wise

@test
def region_wise_leaves_mask_untouched():
    image, mask = synthetic_slice()
    before = mask.copy()
    bezier_region_wise(image, mask, rng=3)
    assert np.array_equal(mask, before), "the mask was modified in place"


@test
def region_wise_treats_tumor_and_background_differently():
    """Same input intensity inside vs outside the mask should usually diverge."""
    h = w = 64
    image = np.full((h, w, 1), 0.5, dtype=np.float32)   # uniform intensity
    mask = np.zeros((h, w, 1), dtype=np.float32)
    mask[:, : w // 2] = 1.0

    differing = 0
    for seed in range(60):
        out = bezier_region_wise(image, mask, rng=seed)
        inside = out[:, : w // 2].flat[0]
        outside = out[:, w // 2:].flat[0]
        if abs(inside - outside) > 1e-3:
            differing += 1
    assert differing > 50, f"only {differing}/60 draws separated the regions"


@test
def empty_mask_falls_back_to_the_background_curve():
    image, _ = synthetic_slice()
    empty = np.zeros_like(image)

    out = bezier_region_wise(image, empty, rng=11)

    rng = np.random.default_rng(11)
    sample_bezier_curve(rng, monotonic=rng.random() < 0.5)          # tumor draw
    xs_b, ys_b = sample_bezier_curve(rng, monotonic=rng.random() < 0.5)
    expected = apply_transfer(image, xs_b, ys_b)

    assert np.allclose(out, expected), "empty mask did not use the background curve"


@test
def region_wise_broadcasts_a_single_channel_mask_over_rgb():
    image = np.repeat(synthetic_slice()[0], 3, axis=-1)
    _, mask = synthetic_slice()
    out = bezier_region_wise(image, mask, rng=5)
    assert out.shape == image.shape, (out.shape, image.shape)
    assert np.allclose(out[..., 0], out[..., 1]), "channels diverged"


@test
def region_wise_output_stays_in_range():
    image, mask = synthetic_slice()
    for seed in range(100):
        out = bezier_region_wise(image, mask, rng=seed)
        assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6, (out.min(), out.max())


# ------------------------------------------------------------------- arms

@test
def standard_arm_only_resamples_intensities():
    """Bilinear warping is a convex combination, so it can never push a value
    outside the input range. A Bezier arm routinely does."""
    image, mask = synthetic_slice()
    lo, hi = float(image.min()), float(image.max())

    plain = Augmenter("standard", seed=1)
    for _ in range(30):
        out, _ = plain(image, mask)
        assert out.min() >= -1e-6 and out.max() <= hi + 1e-6, \
            f"standard arm left the input range [{lo}, {hi}]"

    region = Augmenter("region", seed=1)
    escaped = sum(float(region(image, mask)[0].max()) > hi + 1e-3 for _ in range(30))
    assert escaped > 5, f"region arm escaped the input range only {escaped}/30 times"


@test
def arms_share_the_geometric_stream():
    """Same seed, different arm => identical flips, rotations and scales, so the
    arms are paired within fold and differ only in intensity."""
    image, mask = synthetic_slice()
    a = Augmenter("standard", seed=77)
    b = Augmenter("region", seed=77)
    c = Augmenter("global", seed=77)
    for _ in range(10):
        _, ma = a(image, mask)
        _, mb = b(image, mask)
        _, mc = c(image, mask)
        assert np.array_equal(ma, mb) and np.array_equal(ma, mc), \
            "geometric draws diverged between arms"


@test
def training_false_is_the_identity():
    image, mask = synthetic_slice()
    for arm in ARMS:
        aug = Augmenter(arm, seed=2)
        out_img, out_mask = aug(image, mask, training=False)
        assert out_img is image and out_mask is mask, f"{arm} altered eval input"


@test
def same_seed_reproduces_the_same_output():
    image, mask = synthetic_slice()
    for arm in ARMS:
        a = Augmenter(arm, seed=99)
        b = Augmenter(arm, seed=99)
        for _ in range(5):
            ia, ma = a(image, mask)
            ib, mb = b(image, mask)
            assert np.array_equal(ia, ib) and np.array_equal(ma, mb), arm


@test
def different_seeds_give_different_output():
    image, mask = synthetic_slice()
    for arm in ("global", "region"):
        a, _ = Augmenter(arm, seed=1)(image, mask)
        b, _ = Augmenter(arm, seed=2)(image, mask)
        assert not np.array_equal(a, b), arm


@test
def arms_preserve_shape_and_mask_binarity():
    image, mask = synthetic_slice()
    for arm in ARMS:
        aug = Augmenter(arm, seed=4)
        for _ in range(10):
            out_img, out_mask = aug(image, mask)
            assert out_img.shape == image.shape, (arm, out_img.shape)
            assert out_mask.shape == mask.shape, (arm, out_mask.shape)
            assert set(np.unique(out_mask)) <= {0.0, 1.0}, \
                f"{arm} produced fractional mask values"


@test
def p_bezier_zero_disables_the_intensity_transform():
    image, mask = synthetic_slice()
    plain = Augmenter("standard", seed=8)
    off = Augmenter("region", seed=8, p_bezier=0.0)
    for _ in range(5):
        a, _ = plain(image, mask)
        b, _ = off(image, mask)
        assert np.array_equal(a, b), "p_bezier=0 still transformed intensities"


@test
def invalid_configuration_is_rejected():
    for bad in (lambda: Augmenter("bezier"),
                lambda: Augmenter("region", p_bezier=1.5),
                lambda: Augmenter("region", p_monotonic=-0.1)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid configuration was accepted")


# ------------------------------------------------------------- geometric

@test
def standard_augment_keeps_mask_binary_and_aligned():
    image, mask = synthetic_slice()
    rng = np.random.default_rng(0)
    for _ in range(50):
        out_img, out_mask = standard_augment(image, mask, rng)
        assert out_img.shape == image.shape and out_mask.shape == mask.shape
        assert set(np.unique(out_mask)) <= {0.0, 1.0}, "mask was interpolated"
        if out_mask.sum() > 0:
            # tumor is brighter than background, so the mask should still sit
            # on the bright pixels after the shared warp
            inside = out_img[out_mask > 0.5].mean()
            outside = out_img[(out_mask < 0.5) & (out_img > 0.05)].mean()
            assert inside > outside, "mask drifted off the tumor"


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
