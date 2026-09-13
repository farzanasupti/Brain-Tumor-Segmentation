"""Tests for analyze.py. Run directly: python test_analyze.py

These are the statistics the paper's claims rest on, so they are checked against
values worked out by hand rather than against whatever the code returns.
"""

import csv
import os
import shutil
import tempfile

import numpy as np

import analyze
from analyze import (aggregate, by_tumour_type, contrast, contrast_table, holm,
                     interaction_table, load_results, paired_differences,
                     patient_averaged, summary_table, tost)

RESULTS = []
TMP = None


def test(fn):
    RESULTS.append(fn)
    return fn


def write_results(root, rows):
    os.makedirs(root, exist_ok=True)
    fields = ["cell", "backbone", "family", "condition", "arm", "fold",
              "test_dice", "test_hd95"]
    with open(os.path.join(root, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return root


def grid(dice_by_cell, folds=(0, 1, 2, 3, 4)):
    """dice_by_cell: {(backbone, condition, arm): [one value per fold]}"""
    rows = []
    for (backbone, condition, arm), values in dice_by_cell.items():
        for fold, value in zip(folds, values):
            rows.append({
                "cell": f"{backbone}__{condition}__{arm}__f{fold}",
                "backbone": backbone, "family": "cnn", "condition": condition,
                "arm": arm, "fold": fold, "test_dice": value,
                "test_hd95": 20.0 - 10 * value,
            })
    return rows


# ------------------------------------------------------------------- loading

@test
def results_load_and_coerce():
    root = write_results(os.path.join(TMP, "a1"),
                         grid({("unet", "n/a", "standard"): [0.80, 0.81, 0.79, 0.82, 0.80]}))
    rows = load_results(root)
    assert len(rows) == 5
    assert isinstance(rows[0]["fold"], int)
    assert isinstance(rows[0]["test_dice"], float)


@test
def a_missing_results_file_is_reported_clearly():
    try:
        load_results(os.path.join(TMP, "nope"))
    except SystemExit as exc:
        assert "results.csv" in str(exc) and "experiment.py" in str(exc), exc
        return
    raise AssertionError("a missing results file was tolerated")


# --------------------------------------------------------------- aggregation

@test
def aggregate_summarises_each_cell_across_folds():
    values = [0.80, 0.82, 0.84, 0.86, 0.88]
    root = write_results(os.path.join(TMP, "a2"),
                         grid({("unet", "n/a", "standard"): values}))
    agg = aggregate(load_results(root), "test_dice")
    s = agg[("unet", "n/a", "standard")]
    assert s["n"] == 5
    assert abs(s["mean"] - np.mean(values)) < 1e-12
    assert abs(s["std"] - np.std(values, ddof=1)) < 1e-12


@test
def cells_are_paired_by_fold_not_by_position():
    """Pairing by order would silently compare fold 0 against fold 3."""
    rows = grid({("unet", "n/a", "standard"): [0.10, 0.20, 0.30, 0.40, 0.50]})
    rows += grid({("unet", "n/a", "region"): [0.15, 0.25, 0.35, 0.45, 0.55]})
    # shuffle the file order so position and fold disagree
    rows = rows[::-1]
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a3"), rows)), "test_dice")
    diffs, folds = paired_differences(agg, ("unet", "n/a", "region"),
                                      ("unet", "n/a", "standard"))
    assert folds == [0, 1, 2, 3, 4], folds
    assert np.allclose(diffs, 0.05), diffs


@test
def only_shared_folds_are_compared():
    rows = grid({("unet", "n/a", "standard"): [0.8, 0.8, 0.8, 0.8, 0.8]})
    rows += grid({("unet", "n/a", "region"): [0.9, 0.9, 0.9]}, folds=(0, 1, 2))
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a4"), rows)), "test_dice")
    diffs, folds = paired_differences(agg, ("unet", "n/a", "region"),
                                      ("unet", "n/a", "standard"))
    assert folds == [0, 1, 2] and diffs.size == 3, (folds, diffs)


# ----------------------------------------------------------------- contrasts

@test
def a_constant_improvement_gives_a_known_contrast():
    rows = grid({("unet", "n/a", "standard"): [0.80, 0.81, 0.82, 0.83, 0.84]})
    rows += grid({("unet", "n/a", "region"): [0.82, 0.83, 0.84, 0.85, 0.86]})
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a5"), rows)), "test_dice")
    c = contrast(agg, ("unet", "n/a", "region"), ("unet", "n/a", "standard"))
    assert abs(c["mean"] - 0.02) < 1e-9, c["mean"]
    assert c["sd"] < 1e-9, "a constant difference should have zero spread"
    assert c["n"] == 5


@test
def a_noisy_difference_has_a_ci_that_spans_zero():
    rows = grid({("unet", "n/a", "standard"): [0.80, 0.90, 0.70, 0.85, 0.75]})
    rows += grid({("unet", "n/a", "region"): [0.81, 0.85, 0.76, 0.80, 0.79]})
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a6"), rows)), "test_dice")
    c = contrast(agg, ("unet", "n/a", "region"), ("unet", "n/a", "standard"))
    assert c["ci_low"] < 0 < c["ci_high"], (c["ci_low"], c["ci_high"])
    assert c["p"] > 0.05, c["p"]


@test
def cohens_dz_matches_the_definition():
    rows = grid({("unet", "n/a", "standard"): [0.80, 0.82, 0.84, 0.86, 0.88]})
    rows += grid({("unet", "n/a", "region"): [0.83, 0.84, 0.88, 0.89, 0.90]})
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a7"), rows)), "test_dice")
    c = contrast(agg, ("unet", "n/a", "region"), ("unet", "n/a", "standard"))
    diffs = np.array([0.03, 0.02, 0.04, 0.03, 0.02])
    assert abs(c["cohen_dz"] - diffs.mean() / diffs.std(ddof=1)) < 1e-9


@test
def contrasts_only_compare_cells_differing_in_one_factor():
    rows = grid({("unet", "n/a", "standard"): [0.80] * 5})
    rows += grid({("unet", "n/a", "region"): [0.82] * 5})
    rows += grid({("cond_unet", "predicted", "standard"): [0.84] * 5})
    rows += grid({("cond_unet", "predicted", "region"): [0.86] * 5})
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a8"), rows)), "test_dice")
    text = contrast_table(agg)
    # differs in two factors (backbone and arm) -- not a clean contrast
    assert "cond_unet / predicted / region  vs  unet / n/a / standard" not in text
    # differs in backbone alone, once "n/a" is understood as "no conditioning"
    assert ("cond_unet / predicted / standard  vs  unet / n/a / standard" in text
            or "unet / n/a / standard  vs  cond_unet / predicted / standard" in text), \
        "the headline unet vs cond_unet contrast was excluded"
    assert "4 contrast(s)" in text, text.splitlines()[-1]


# ------------------------------------------------------------ multiplicity

@test
def holm_matches_a_hand_computed_example():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


@test
def holm_is_monotone_and_never_shrinks_a_p_value():
    ps = [0.001, 0.02, 0.03, 0.5]
    adjusted = holm(ps)
    assert all(a >= p for a, p in zip(adjusted, ps)), (ps, adjusted)
    assert all(a <= 1.0 for a in adjusted)


# ------------------------------------------------------------- equivalence

@test
def tost_confirms_equivalence_for_a_tight_difference():
    diffs = np.array([0.001, -0.002, 0.000, 0.001, -0.001])
    assert tost(diffs, margin=0.01) < 0.05, "a tiny difference was not called equivalent"


@test
def tost_refuses_equivalence_for_a_real_difference():
    diffs = np.array([0.05, 0.06, 0.04, 0.05, 0.055])
    assert tost(diffs, margin=0.01) > 0.05, "a clear difference was called equivalent"


@test
def a_non_significant_difference_is_not_equivalence():
    """The whole reason TOST is here: noisy data fails both tests."""
    diffs = np.array([0.05, -0.06, 0.04, -0.03, 0.02])
    from scipy import stats
    _, p = stats.ttest_rel(diffs, np.zeros_like(diffs))
    assert p > 0.05, "expected a non-significant difference"
    assert tost(diffs, margin=0.01) > 0.05, (
        "noisy data must fail the equivalence test too, not pass it by default")


# ------------------------------------------------------ per-slice breakdowns

@test
def patient_averaging_stops_one_patient_dominating():
    """This dataset has patients with 1 slice and patients with 38."""
    rows = [{"pid": "P1", "type": "0", "dice": "0.0"} for _ in range(10)]
    rows += [{"pid": f"P{i}", "type": "0", "dice": "1.0"} for i in range(2, 12)]
    slice_mean = np.mean([float(r["dice"]) for r in rows])
    patient_mean, n_patients = patient_averaged(rows)
    assert abs(slice_mean - 0.5) < 1e-9, slice_mean
    assert n_patients == 11
    assert abs(patient_mean - 10 / 11) < 1e-9, patient_mean
    assert patient_mean > slice_mean


@test
def tumour_types_are_named_not_numbered():
    rows = [{"pid": "P1", "type": "0", "dice": "0.8"},
            {"pid": "P2", "type": "1", "dice": "0.6"},
            {"pid": "P3", "type": "2", "dice": "0.9"}]
    out = by_tumour_type(rows)
    assert set(out) == {"meningioma", "glioma", "pituitary"}, out
    assert abs(out["glioma"][0] - 0.6) < 1e-9


@test
def unparseable_values_are_skipped_not_crashed_on():
    rows = [{"pid": "P1", "type": "0", "dice": "0.8"},
            {"pid": "P2", "type": "0", "dice": ""},
            {"pid": "P3", "type": "0", "dice": "nan"}]
    mean, n = patient_averaged(rows)
    assert n == 1 and abs(mean - 0.8) < 1e-9, (mean, n)


# --------------------------------------------------------------- rendering

@test
def the_summary_orders_by_the_right_end_of_the_metric():
    rows = grid({("unet", "n/a", "standard"): [0.70] * 5})
    rows += grid({("unet", "n/a", "region"): [0.90] * 5})
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a9"), rows)), "test_dice")

    higher = summary_table(agg, "test_dice", lower_is_better=False)
    assert higher.index("region") < higher.index("standard"), "best should lead"

    lower = summary_table(agg, "test_dice", lower_is_better=True)
    assert lower.index("standard") < lower.index("region"), "lowest should lead"


@test
def the_interaction_table_lays_out_both_factors():
    rows = grid({("unet", "n/a", "standard"): [0.80] * 5})
    rows += grid({("unet", "n/a", "region"): [0.82] * 5})
    rows += grid({("cond_unet", "predicted", "standard"): [0.84] * 5})
    rows += grid({("cond_unet", "predicted", "region"): [0.86] * 5})
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a10"), rows)), "test_dice")
    table = interaction_table(agg)
    assert table is not None
    for token in ("unet", "cond_unet", "region", "standard", "0.8600"):
        assert token in table, token


@test
def a_single_cell_yields_no_interaction_table():
    rows = grid({("unet", "n/a", "standard"): [0.80] * 5})
    agg = aggregate(load_results(write_results(os.path.join(TMP, "a11"), rows)), "test_dice")
    assert interaction_table(agg) is None


if __name__ == "__main__":
    TMP = tempfile.mkdtemp(prefix="bts-analyze-tests-")
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
