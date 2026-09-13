"""Tests for segmetrics.py. Run directly: python test_segmetrics.py

Boundary metrics are checked against distances worked out by hand, because a
plausible-looking but wrong HD95 would quietly change which model the paper
says is better.
"""

import numpy as np

import segmetrics
from segmetrics import assd, batch_hd95, boundary, hd95, summarise, surface_distances

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def square(size=64, top=20, left=20, height=10, width=10):
    m = np.zeros((size, size), dtype=np.float32)
    m[top:top + height, left:left + width] = 1.0
    return m


# ------------------------------------------------------------------- basics

@test
def scipy_is_available():
    ok, detail = segmetrics.check()
    assert ok, detail


@test
def a_perfect_prediction_scores_zero():
    m = square()
    assert hd95(m, m) == 0.0
    assert assd(m, m) == 0.0


@test
def boundary_is_one_pixel_wide():
    m = square(height=10, width=10)
    b = boundary(m)
    assert b.sum() == 36, f"a 10x10 square has 36 border pixels, got {int(b.sum())}"
    assert not boundary(np.zeros((8, 8))).any(), "empty mask has no boundary"


@test
def a_known_shift_gives_the_known_distance():
    """Shifting a square 3 px right moves most of the boundary by 3."""
    a = square(left=20)
    b = square(left=23)
    d = hd95(a, b)
    assert 2.5 <= d <= 3.6, f"expected about 3 px, got {d:.3f}"


@test
def hd95_grows_with_displacement():
    a = square(left=20)
    previous = -1.0
    for shift in (1, 3, 6, 12):
        d = hd95(a, square(left=20 + shift))
        assert d > previous, f"shift {shift} did not increase HD95 ({d:.2f})"
        previous = d


@test
def hd95_ignores_a_single_stray_pixel():
    """That is the point of the 95th percentile rather than the maximum."""
    truth = square()
    noisy = square()
    noisy[60, 60] = 1.0                         # one pixel far away
    assert hd95(noisy, truth) < 10.0, hd95(noisy, truth)


@test
def assd_is_smaller_than_hd95_for_a_ragged_edge():
    rng = np.random.default_rng(0)
    truth = square(height=20, width=20)
    pred = truth.copy()
    edge = np.argwhere(boundary(truth.astype(bool)))
    for y, x in edge[rng.integers(0, len(edge), 8)]:
        pred[y, x] = 0.0
    assert assd(pred, truth) <= hd95(pred, truth), "ASSD exceeded HD95"


@test
def the_metric_is_symmetric():
    a, b = square(left=20), square(left=26, height=14)
    assert abs(hd95(a, b) - hd95(b, a)) < 1e-9
    assert abs(assd(a, b) - assd(b, a)) < 1e-9


# ------------------------------------------------------- the empty convention

@test
def both_empty_is_a_correct_answer():
    empty = np.zeros((32, 32))
    assert hd95(empty, empty) == 0.0
    assert assd(empty, empty) == 0.0


@test
def one_empty_is_undefined_by_default():
    empty, full = np.zeros((32, 32)), square(size=32, top=5, left=5)
    assert np.isnan(hd95(empty, full)), "a missed tumour silently scored a number"
    assert np.isnan(hd95(full, empty))
    assert np.isnan(assd(empty, full))


@test
def the_diagonal_convention_penalises_instead_of_skipping():
    empty, full = np.zeros((32, 32)), square(size=32, top=5, left=5)
    d = hd95(empty, full, empty_value="diagonal")
    assert abs(d - np.hypot(32, 32)) < 1e-6, d


@test
def a_custom_empty_value_is_honoured():
    empty, full = np.zeros((16, 16)), square(size=16, top=2, left=2, height=4, width=4)
    assert hd95(empty, full, empty_value=99.0) == 99.0


# ---------------------------------------------------------------- aggregation

@test
def summarise_counts_the_undefined_rather_than_hiding_them():
    """Dropping missed slices silently would inflate the reported score."""
    values = [1.0, 3.0, float("nan"), 5.0, float("nan")]
    out = summarise(values, "hd95")
    assert out["hd95"] == 3.0, out
    assert out["hd95_n"] == 3 and out["hd95_undefined"] == 2, out
    assert out["hd95_median"] == 3.0, out


@test
def summarise_survives_an_all_undefined_batch():
    out = summarise([float("nan")] * 4, "hd95")
    assert np.isnan(out["hd95"]) and out["hd95_n"] == 0 and out["hd95_undefined"] == 4


@test
def batch_hd95_matches_the_single_slice_version():
    a = np.stack([square(left=20), square(left=25)])[:, None]
    b = np.stack([square(left=23), square(left=25)])[:, None]
    batch = batch_hd95(a, b)
    assert len(batch) == 2
    assert abs(batch[0] - hd95(a[0], b[0])) < 1e-9
    assert batch[1] == 0.0


# ------------------------------------------------------------------- shapes

@test
def channel_dimensions_are_accepted():
    m = square()
    assert hd95(m[None], m[None, ..., None][0]) == 0.0
    assert hd95(m.reshape(1, 64, 64), m.reshape(64, 64, 1)) == 0.0


@test
def mismatched_shapes_are_rejected():
    try:
        surface_distances(square(size=32), square(size=64))
    except ValueError as exc:
        assert "shape mismatch" in str(exc), exc
        return
    raise AssertionError("mismatched masks were compared")


@test
def spacing_scales_the_distance():
    """Reporting in millimetres instead of pixels is a spacing argument away;
    this dataset is 0.49 mm isotropic in plane."""
    a, b = square(left=20), square(left=24)
    pixels = hd95(a, b)
    mm = hd95(a, b, spacing=0.49)
    assert abs(mm - pixels * 0.49) < 1e-6, (pixels, mm)


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
