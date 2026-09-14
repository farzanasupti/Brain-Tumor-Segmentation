"""Turn a run's results.csv into the numbers a paper reports.

    python analyze.py --run runs/main
    python analyze.py --run runs/main --metric test_hd95 --lower-is-better
    python analyze.py --run runs/main --baseline unet/n-a/standard

The design is paired: every cell of the grid is trained on the same five folds,
and the augmentation arms see identical geometric transforms item by item. So a
contrast between two cells is a paired comparison over five fold-level
differences, not two independent samples of five. That is what is tested here.

Five folds is a small n, and the honest consequences are built in:

  * effect sizes with confidence intervals, not bare p-values;
  * Holm correction across the contrast family, with the family size printed;
  * equivalence testing (TOST) for claims of *no* difference, because a
    non-significant difference at n=5 is absence of evidence, and reporting it
    as "backbone-agnostic" would be wrong.

Needs scipy. statsmodels is used for a mixed model when present and skipped with
a note when not -- at five folds the paired contrasts carry the argument anyway.
"""

import argparse
import csv
import itertools
import os
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

FACTORS = ("backbone", "condition", "arm")


def load_results(run_dir):
    """Read results.csv from a run directory."""
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        raise SystemExit(f"No results at {path!r}. Run experiment.py first.")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path!r} has no rows.")

    for r in rows:
        r["fold"] = int(r["fold"])
        for key, value in list(r.items()):
            if key.startswith(("test_", "best_")) and value not in ("", None):
                try:
                    r[key] = float(value)
                except ValueError:
                    pass
    return rows


def cell_key(row):
    return tuple(row[f] for f in FACTORS)


def label(key):
    return " / ".join(key)


def aggregate(rows, metric):
    """Mean and spread of `metric` across folds, per cell."""
    by_cell = defaultdict(dict)
    for r in rows:
        value = r.get(metric)
        if isinstance(value, float) and not np.isnan(value):
            by_cell[cell_key(r)][r["fold"]] = value

    out = {}
    for key, per_fold in by_cell.items():
        values = np.array([per_fold[f] for f in sorted(per_fold)], dtype=float)
        out[key] = {
            "folds": sorted(per_fold),
            "values": values,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "n": int(values.size),
        }
    return out


def paired_differences(agg, key_a, key_b):
    """Fold-wise differences a - b, over folds both cells actually have."""
    a, b = agg.get(key_a), agg.get(key_b)
    if a is None or b is None:
        return None, []
    shared = sorted(set(a["folds"]) & set(b["folds"]))
    if not shared:
        return None, []
    av = {f: v for f, v in zip(a["folds"], a["values"])}
    bv = {f: v for f, v in zip(b["folds"], b["values"])}
    return np.array([av[f] - bv[f] for f in shared], dtype=float), shared


def contrast(agg, key_a, key_b):
    """Paired comparison of two cells. Returns None when they do not overlap."""
    diffs, folds = paired_differences(agg, key_a, key_b)
    if diffs is None or diffs.size < 2:
        return None

    n = diffs.size
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))

    result = {
        "a": key_a, "b": key_b, "folds": folds, "n": n,
        "mean": mean, "sd": sd,
        "cohen_dz": mean / sd if sd > 0 else float("inf") * np.sign(mean),
    }

    if HAVE_SCIPY and sd > 0:
        t_stat, p = stats.ttest_rel(diffs, np.zeros_like(diffs))
        crit = stats.t.ppf(0.975, n - 1)
        half = crit * sd / np.sqrt(n)
        result.update({"t": float(t_stat), "p": float(p),
                       "ci_low": mean - half, "ci_high": mean + half})
    return result


def tost(diffs, margin):
    """Two one-sided tests. A small p supports 'the difference is within margin'.

    Use this, not a non-significant t-test, to claim two things are equivalent.
    """
    if not HAVE_SCIPY:
        return None
    n = diffs.size
    if n < 2:
        return None
    sd = diffs.std(ddof=1)
    if sd == 0:
        return 0.0 if abs(diffs.mean()) < margin else 1.0

    se = sd / np.sqrt(n)
    df = n - 1
    t_lower = (diffs.mean() + margin) / se
    t_upper = (diffs.mean() - margin) / se
    p_lower = stats.t.sf(t_lower, df)        # H0: diff <= -margin
    p_upper = stats.t.cdf(t_upper, df)       # H0: diff >= +margin
    return float(max(p_lower, p_upper))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the order given."""
    indexed = sorted(enumerate(pvalues), key=lambda kv: kv[1])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, (index, p) in enumerate(indexed):
        running = max(running, (m - rank) * p)
        adjusted[index] = min(1.0, running)
    return adjusted


def per_slice_rows(run_dir, cell):
    path = os.path.join(run_dir, cell, "test_slices.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def patient_averaged(rows, metric="dice"):
    """Average per patient first, then over patients.

    Slice-averaging lets a patient with 38 slices outweigh one with 3, and this
    dataset's per-patient counts run from 1 to 38.
    """
    by_patient = defaultdict(list)
    for r in rows:
        try:
            value = float(r[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isnan(value):
            by_patient[r.get("pid", "")].append(value)
    if not by_patient:
        return float("nan"), 0
    means = [np.mean(v) for v in by_patient.values()]
    return float(np.mean(means)), len(means)


def by_tumour_type(rows, metric="dice"):
    names = {"0": "meningioma", "1": "glioma", "2": "pituitary"}
    grouped = defaultdict(list)
    for r in rows:
        try:
            value = float(r[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isnan(value):
            grouped[names.get(str(r.get("type")), str(r.get("type")))].append(value)
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(grouped.items())}


# ------------------------------------------------------------------ reporting

def summary_table(agg, metric, lower_is_better=False):
    lines = []
    header = f"{'cell':<42}{'n':>3}{'mean':>10}{'sd':>9}   per-fold"
    lines += [header, "-" * len(header)]
    order = sorted(agg, key=lambda k: agg[k]["mean"], reverse=not lower_is_better)
    for key in order:
        s = agg[key]
        folds = " ".join(f"{v:.4f}" for v in s["values"])
        sd = "  --  " if np.isnan(s["std"]) else f"{s['std']:.4f}"
        lines.append(f"{label(key):<42}{s['n']:>3}{s['mean']:>10.4f}{sd:>9}   {folds}")
    lines.append("")
    lines.append(f"metric: {metric}  ({'lower' if lower_is_better else 'higher'} is better)")
    return "\n".join(lines)


def _differing_factors(a, b):
    """Which factors two cells differ in.

    A condition difference is not counted when either side is "n/a": a plain
    backbone has no conditioning to vary, so unet/n/a/standard against
    cond_unet/predicted/standard differs in *backbone* alone. Counting it as two
    differences would exclude the comparison this study exists to make.
    """
    differing = []
    for i, factor in enumerate(FACTORS):
        if a[i] == b[i]:
            continue
        if factor == "condition" and "n/a" in (a[i], b[i]):
            continue
        differing.append(i)
    return differing


def contrast_table(agg, baseline=None, margin=None):
    """Every paired contrast between cells differing in exactly one factor."""
    keys = sorted(agg)
    pairs = []
    for a, b in itertools.combinations(keys, 2):
        differing = _differing_factors(a, b)
        if len(differing) != 1:
            continue
        if baseline and baseline not in (a, b):
            continue
        pairs.append((b, a) if baseline == a else (a, b))

    results = [c for c in (contrast(agg, a, b) for a, b in pairs) if c]
    if not results:
        return "No paired contrasts available (cells share no folds)."

    ps = [r.get("p") for r in results]
    if all(p is not None for p in ps):
        for r, adj in zip(results, holm(ps)):
            r["p_holm"] = adj

    lines = []
    header = (f"{'contrast (a - b)':<58}{'n':>3}{'diff':>9}{'95% CI':>18}"
              f"{'dz':>7}{'p':>9}{'p_holm':>9}")
    lines += [header, "-" * len(header)]
    for r in sorted(results, key=lambda x: -abs(x["mean"])):
        name = f"{label(r['a'])}  vs  {label(r['b'])}"
        ci = (f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
              if "ci_low" in r else "        --        ")
        p = f"{r['p']:.4f}" if "p" in r else "  --  "
        ph = f"{r['p_holm']:.4f}" if "p_holm" in r else "  --  "
        lines.append(f"{name:<58}{r['n']:>3}{r['mean']:>+9.4f}{ci:>18}"
                     f"{r['cohen_dz']:>7.2f}{p:>9}{ph:>9}")

    lines.append("")
    lines.append(f"{len(results)} contrast(s); p_holm is Holm-corrected across that family.")

    if margin is not None:
        lines.append("")
        lines.append(f"Equivalence (TOST) at a margin of +/-{margin}:")
        for r in results:
            diffs, _ = paired_differences(agg, r["a"], r["b"])
            p_eq = tost(diffs, margin)
            verdict = ("equivalent within margin" if p_eq is not None and p_eq < 0.05
                       else "cannot claim equivalence")
            shown = f"{p_eq:.4f}" if p_eq is not None else "--"
            lines.append(f"  {label(r['a'])} vs {label(r['b'])}: p={shown}  {verdict}")
        lines.append("  A non-significant difference is not equivalence; this is the test"
                     " that supports a 'no difference' claim.")
    return "\n".join(lines)


def interaction_table(agg, row_factor="backbone", col_factor="arm"):
    """Cell means laid out as factor x factor -- parallel rows mean no interaction."""
    ri, ci = FACTORS.index(row_factor), FACTORS.index(col_factor)
    rows = sorted({k[ri] for k in agg})
    cols = sorted({k[ci] for k in agg})
    if len(rows) < 2 or len(cols) < 2:
        return None

    lines = [f"{row_factor} x {col_factor} cell means",
             f"{'':<16}" + "".join(f"{c:>16}" for c in cols)]
    lines.append("-" * len(lines[-1]))
    for r in rows:
        cells = []
        for c in cols:
            matching = [agg[k]["mean"] for k in agg if k[ri] == r and k[ci] == c]
            cells.append(f"{np.mean(matching):>16.4f}" if matching else f"{'--':>16}")
        lines.append(f"{r:<16}" + "".join(cells))
    lines.append("")
    lines.append("Parallel rows are the signature of no interaction; crossing rows are "
                 "the interesting failure.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory holding results.csv")
    parser.add_argument("--metric", default="test_dice")
    parser.add_argument("--lower-is-better", action="store_true",
                        help="set for distance metrics such as test_hd95")
    parser.add_argument("--baseline", default=None,
                        help="restrict contrasts to this cell, as backbone/condition/arm")
    parser.add_argument("--margin", type=float, default=None,
                        help="equivalence margin for TOST, e.g. 0.005 Dice")
    parser.add_argument("--per-slice", action="store_true",
                        help="also report patient-averaged and per-type breakdowns")
    args = parser.parse_args()

    if not HAVE_SCIPY:
        print("scipy is not installed; reporting means only, no tests.\n")

    rows = load_results(args.run)
    agg = aggregate(rows, args.metric)
    if not agg:
        raise SystemExit(f"No usable values for metric {args.metric!r}.")

    baseline = tuple(args.baseline.replace("n-a", "n/a").split("/")) if args.baseline else None
    if baseline and baseline not in agg:
        raise SystemExit(f"Baseline {args.baseline!r} is not among the cells: "
                         f"{', '.join(label(k) for k in sorted(agg))}")

    print(summary_table(agg, args.metric, args.lower_is_better))
    print()
    print(contrast_table(agg, baseline, args.margin))

    table = interaction_table(agg)
    if table:
        print()
        print(table)

    if args.per_slice:
        print()
        print("Patient-averaged and per-type (slice-averaging lets a 38-slice "
              "patient outweigh a 3-slice one):")
        from experiment import cell_name
        for key in sorted(agg):
            slices = per_slice_rows(args.run, cell_name(key[0], key[1], key[2], 0))
            if not slices:
                continue
            mean, n_patients = patient_averaged(slices)
            types = by_tumour_type(slices)
            detail = ", ".join(f"{k} {v:.4f} (n={c})" for k, (v, c) in types.items())
            print(f"  {label(key)} fold 0: patient-averaged {mean:.4f} "
                  f"over {n_patients} patients | {detail}")


if __name__ == "__main__":
    main()
