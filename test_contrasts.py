"""Tests for contrasts.py. Run directly: python test_contrasts.py

Checked against values worked out by hand on small synthetic runs.
"""

import csv
import os
import shutil
import tempfile

import numpy as np

import contrasts
from contrasts import family, fold_scores, paired_contrast, slice_averaged

RESULTS = []
TMP = None


def test(fn):
    RESULTS.append(fn)
    return fn


def write_slices(runs_dir, run, cell, fold, rows):
    """rows: list of (pid, dice)."""
    d = os.path.join(runs_dir, run, f"{cell}__f{fold}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "test_slices.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slice_id", "pid", "type", "predicted_type", "dice"])
        for i, (pid, dice) in enumerate(rows):
            w.writerow([i, pid, 0, "", dice])


@test
def slice_and_patient_averages_differ_when_slice_counts_differ():
    root = os.path.join(TMP, "avg")
    # patient a: three slices at 0.9; patient b: one slice at 0.5
    write_slices(root, "r", "c", 0, [("a", 0.9), ("a", 0.9), ("a", 0.9), ("b", 0.5)])
    s = fold_scores(root, "r", "c", "slice-averaged", folds=(0,))
    p = fold_scores(root, "r", "c", "patient-averaged", folds=(0,))
    assert abs(s[0] - 0.8) < 1e-12, s          # (0.9*3 + 0.5) / 4
    assert abs(p[0] - 0.7) < 1e-12, p          # (0.9 + 0.5) / 2


@test
def unparseable_and_nan_scores_are_skipped():
    assert slice_averaged([{"dice": "0.6"}, {"dice": "nan"}, {"dice": ""}, {}]) == 0.6


@test
def paired_contrast_matches_hand_values():
    a = {f: v for f, v in enumerate([0.80, 0.82, 0.78, 0.81, 0.79])}
    b = {f: v for f, v in enumerate([0.78, 0.80, 0.77, 0.80, 0.77])}
    r = paired_contrast(a, b)
    d = np.array([0.02, 0.02, 0.01, 0.01, 0.02])
    assert abs(r["mean"] - 0.016) < 1e-12
    se = d.std(ddof=1) / np.sqrt(5)
    q = 2.7764451051977987                      # t(0.975, 4)
    assert abs(r["lo"] - (0.016 - q * se)) < 1e-9
    assert abs(r["hi"] - (0.016 + q * se)) < 1e-9
    assert r["p"] < 0.01


@test
def only_folds_present_on_both_sides_are_paired():
    r = paired_contrast({0: 0.8, 1: 0.7, 2: 0.9}, {1: 0.6, 2: 0.8, 3: 0.5})
    assert r["folds"] == [1, 2]
    assert abs(r["mean"] - 0.1) < 1e-12


@test
def holm_applies_within_the_family_only():
    root = os.path.join(TMP, "fam")
    rng = np.random.default_rng(0)
    for f in range(5):
        base = 0.75 + 0.01 * f
        write_slices(root, "main", "std", f, [("p", base)])
        write_slices(root, "main", "big", f, [("p", base + 0.10 + rng.normal(0, 0.005))])
        write_slices(root, "main", "small", f, [("p", base + rng.normal(0, 0.005))])
    spec = {"big": (("main", "big"), ("main", "std")),
            "small": (("main", "small"), ("main", "std"))}
    corrected = family(root, spec, "slice-averaged", correct=True)
    raw = family(root, spec, "slice-averaged", correct=False)
    ps = [corrected[n]["p"] for n in spec]
    lo = min(ps)
    assert abs(min(corrected[n]["p_holm"] for n in spec) - 2 * lo) < 1e-12
    assert all("p_holm" not in r for r in raw.values())


@test
def missing_run_gives_an_empty_contrast_not_a_crash():
    r = family(os.path.join(TMP, "none"), contrasts.EXPLORATORY, "slice-averaged")
    assert all(v["folds"] == [] for v in r.values())


@test
def registered_h4_is_region_minus_global():
    (treat, comp) = contrasts.REGISTERED["H4 region Bezier - global Bezier"]
    assert treat[1] == "unet__n-a__region" and comp[1] == "unet__n-a__global"


if __name__ == "__main__":
    TMP = tempfile.mkdtemp(prefix="bts-contrasts-tests-")
    failed = 0
    try:
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
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    raise SystemExit(1 if failed else 0)
