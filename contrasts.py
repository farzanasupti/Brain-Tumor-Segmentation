"""The contrasts the paper reports, computed from per-slice test scores.

    python contrasts.py                     # everything below, runs/ as root
    python contrasts.py --runs-dir runs --only registered

analyze.py works from results.csv, so it sees one pooled test Dice per fold.
The paper reports two fold-level outcomes instead, both rebuilt here from each
cell's test_slices.csv:

  * slice-averaged Dice: the mean over a fold's test slices;
  * patient-averaged Dice: the mean per patient first, then over patients, so a
    patient with 38 slices does not outweigh one with 3.

Each contrast is a paired comparison over the five fold-level differences, with
a t-based 95% CI, a one-sample t-test, and TOST against the registered margin.

Two families, kept apart on purpose:

  * registered (H1-H4, PREREGISTRATION.md): Holm-corrected across the four;
  * exploratory (weight decay 1e-4, bounded region arm): registered as
    exploratory before they ran, reported without correction and never pooled
    into the registered family.

Note that H4 is region - global, as registered, not global - standard.
"""

import argparse
import os

import numpy as np

from analyze import (HAVE_SCIPY, REGISTERED_MARGIN, holm, patient_averaged,
                     per_slice_rows, tost)

if HAVE_SCIPY:
    from scipy import stats

FOLDS = (0, 1, 2, 3, 4)

#: name -> ((run, cell) of the treatment, (run, cell) of the comparator)
REGISTERED = {
    "H1 predicted-type FiLM - unconditioned": (
        ("main", "cond_unet__predicted__standard"), ("main", "unet__n-a__standard")),
    "H2 oracle-type FiLM - predicted-type FiLM": (
        ("main", "cond_unet__oracle__standard"), ("main", "cond_unet__predicted__standard")),
    "H3 region Bezier - standard": (
        ("main", "unet__n-a__region"), ("main", "unet__n-a__standard")),
    "H4 region Bezier - global Bezier": (
        ("main", "unet__n-a__region"), ("main", "unet__n-a__global")),
}

EXPLORATORY = {
    "weight decay 1e-4 - 1e-2 (standard U-Net)": (
        ("wd1e-4", "unet__n-a__standard"), ("main", "unet__n-a__standard")),
    "bounded region Bezier - standard": (
        ("region-bounded", "unet__n-a__region-bounded"), ("main", "unet__n-a__standard")),
}


def slice_averaged(rows, metric="dice"):
    values = []
    for r in rows:
        try:
            v = float(r[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isnan(v):
            values.append(v)
    return float(np.mean(values)) if values else float("nan")


OUTCOMES = {
    "slice-averaged": slice_averaged,
    "patient-averaged": lambda rows: patient_averaged(rows)[0],
}


def fold_scores(runs_dir, run, cell, outcome, folds=FOLDS):
    """{fold: score} for the folds that have a test_slices.csv."""
    scores = {}
    for f in folds:
        rows = per_slice_rows(os.path.join(runs_dir, run), f"{cell}__f{f}")
        if rows:
            scores[f] = OUTCOMES[outcome](rows)
    return scores


def paired_contrast(a_scores, b_scores, margin=REGISTERED_MARGIN):
    """Paired statistics over the folds both sides have."""
    folds = sorted(set(a_scores) & set(b_scores))
    a = np.array([a_scores[f] for f in folds])
    b = np.array([b_scores[f] for f in folds])
    d = a - b
    out = {"folds": folds, "a": a, "b": b, "diffs": d,
           "mean": float(d.mean()) if d.size else float("nan"),
           "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "tost": None}
    if d.size >= 2 and HAVE_SCIPY:
        se = d.std(ddof=1) / np.sqrt(d.size)
        q = stats.t.ppf(0.975, d.size - 1)
        out["lo"], out["hi"] = out["mean"] - q * se, out["mean"] + q * se
        out["p"] = float(stats.ttest_1samp(d, 0).pvalue) if se > 0 else float("nan")
        out["tost"] = tost(d, margin)
    return out


def family(runs_dir, contrasts, outcome, margin=REGISTERED_MARGIN, correct=False):
    results = {}
    for name, ((run_a, cell_a), (run_b, cell_b)) in contrasts.items():
        results[name] = paired_contrast(fold_scores(runs_dir, run_a, cell_a, outcome),
                                        fold_scores(runs_dir, run_b, cell_b, outcome),
                                        margin)
    if correct:
        testable = [n for n, r in results.items() if not np.isnan(r["p"])]
        for n, p in zip(testable, holm([results[n]["p"] for n in testable])):
            results[n]["p_holm"] = p
    return results


def report(title, results, correct):
    print(f"\n{title}")
    for name, r in results.items():
        if not r["folds"]:
            print(f"  {name}: no folds with per-slice scores on both sides")
            continue
        partial = "" if len(r["folds"]) == len(FOLDS) else f"  [folds {r['folds']} only]"
        holm_part = f"  p_holm {r['p_holm']:.4f}" if correct and "p_holm" in r else ""
        tost_part = f"  TOST {r['tost']:.3f}" if r["tost"] is not None else ""
        print(f"  {name}{partial}")
        print(f"    {r['a'].mean():.4f} vs {r['b'].mean():.4f}  diff {r['mean']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  p {r['p']:.4f}{holm_part}{tost_part}")
        print(f"    per fold {np.round(r['diffs'], 4).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs",
                        help="directory holding main/, wd1e-4/ and region-bounded/")
    parser.add_argument("--only", choices=("registered", "exploratory"), default=None)
    parser.add_argument("--margin", type=float, default=REGISTERED_MARGIN)
    args = parser.parse_args()

    if not HAVE_SCIPY:
        raise SystemExit("contrasts.py needs scipy")
    if args.margin != REGISTERED_MARGIN:
        print(f"NOTE: margin {args.margin} differs from the registered {REGISTERED_MARGIN}")

    groups = [("registered", REGISTERED, True), ("exploratory", EXPLORATORY, False)]
    for outcome in OUTCOMES:
        for kind, contrasts, correct in groups:
            if args.only and args.only != kind:
                continue
            title = (f"{outcome} Dice, {kind}"
                     + (f" (Holm over {len(contrasts)})" if correct else " (uncorrected)"))
            report(title, family(args.runs_dir, contrasts, outcome, args.margin, correct),
                   correct)


if __name__ == "__main__":
    main()
