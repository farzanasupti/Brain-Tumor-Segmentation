"""How much of a published Dice score is protocol rather than model.

    python leakage.py --patient runs/baseline --slice-level runs/diag-leaky \
        --target 0.841 --out figures/

Two choices sit between a segmentation model and the number a paper prints, and
neither is usually stated:

  * the **split**: patient-disjoint, or slice-level, where slices from one
    patient land on both sides of the split;
  * the **aggregation**: mean over slices, or pooled (dataset-level) Dice, which
    is the same thing weighted by tumour size.

This module measures both on runs that already exist, so nothing here needs a
GPU. It reads each cell's test_slices.csv and reports the same five folds under
every combination, then decomposes the distance to a published target into the
part each choice explains.

The variance result matters more than the mean shift and is easy to miss: a
slice-level split stops sampling patients, so the between-fold standard
deviation a paper reports beside its mean is no longer measuring patient-level
variation. That claim does not depend on which aggregation convention the paper
used.

The comparison is **not paired**. A slice-level split reshuffles, so each fold's
test set is a near-identical but not identical set of slices; differences are
reported as differences of means, without a paired test.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

TYPE_NAMES = {"0": "meningioma", "1": "glioma", "2": "pituitary"}
CONVENTIONS = ("slice-averaged", "pooled", "patient-averaged")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def cells_in(run_dir):
    """Cell directories holding per-slice scores, in fold order."""
    if not os.path.isdir(run_dir):
        return []
    names = [n for n in sorted(os.listdir(run_dir))
             if os.path.exists(os.path.join(run_dir, n, "test_slices.csv"))]
    return sorted(names, key=lambda n: n.rsplit("__f", 1)[-1])


def slice_scores(rows, metric="dice"):
    out = []
    for r in rows:
        try:
            value = float(r[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isnan(value):
            out.append(value)
    return np.array(out, dtype=float)


def pooled_dice(rows):
    """Dataset-level Dice: 2*sum(intersection) / sum(pred + true).

    Recovered exactly from what each cell saved. Per slice
    dice = 2*I/(P+T), so I = dice*(P+T)/2 and the pooled figure is the
    size-weighted mean of the per-slice scores. This is an identity, not an
    approximation -- which is the point: "pooled Dice" is slice-averaged Dice
    with big tumours counted more, and on this dataset that is worth over a
    point on its own.
    """
    dice, weight = [], []
    for r in rows:
        try:
            d = float(r["dice"])
            w = float(r["pred_pixels"]) + float(r["true_pixels"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isnan(d) and w > 0:
            dice.append(d)
            weight.append(w)
    if not weight:
        return float("nan")
    return float(np.average(dice, weights=weight))


def patient_averaged(rows, metric="dice"):
    """Average within patient, then across patients.

    Under a slice-level split every slice is its own pseudo-patient, so this
    collapses onto the slice-averaged figure. That the two come out equal is a
    consistency check on the split, not a result.
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
    return float(np.mean([np.mean(v) for v in by_patient.values()])), len(by_patient)


def fold_conventions(rows):
    """The three conventions for one fold's per-slice scores."""
    dice = slice_scores(rows)
    patient, n_patients = patient_averaged(rows)
    return {
        "slice-averaged": float(dice.mean()) if dice.size else float("nan"),
        "pooled": pooled_dice(rows),
        "patient-averaged": patient,
        "n_slices": dice.size,
        "n_patients": n_patients,
        "n_zero": int((dice == 0).sum()),
    }


def run_conventions(run_dir):
    """Per-fold and summary figures for a whole run."""
    folds = []
    for cell in cells_in(run_dir):
        rows = read_csv(os.path.join(run_dir, cell, "test_slices.csv"))
        entry = fold_conventions(rows)
        entry["cell"] = cell
        folds.append(entry)
    summary = {}
    for key in CONVENTIONS:
        values = np.array([f[key] for f in folds], dtype=float)
        summary[key] = (float(values.mean()),
                        float(values.std(ddof=1)) if values.size > 1 else float("nan"))
    return {"folds": folds, "summary": summary}


def by_type(run_dir):
    """Per-tumour-type mean Dice and zero-Dice rate, pooled over folds."""
    grouped = defaultdict(list)
    for cell in cells_in(run_dir):
        for r in read_csv(os.path.join(run_dir, cell, "test_slices.csv")):
            try:
                value = float(r["dice"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isnan(value):
                grouped[TYPE_NAMES.get(str(r.get("type")), str(r.get("type")))].append(value)
    out = {}
    for name, values in sorted(grouped.items()):
        arr = np.array(values, dtype=float)
        out[name] = {"mean": float(arr.mean()), "n": arr.size,
                     "zero": int((arr == 0).sum()),
                     "zero_rate": float((arr == 0).mean())}
    return out


def decomposition(patient_run, slice_run, target):
    """Split the distance from the registered figure to `target` into parts.

    The registered figure is patient-disjoint and slice-averaged. Walking to the
    published convention one choice at a time -- first aggregation, then split --
    gives each an attributable share. The order is stated because the two are not
    strictly additive; taking them the other way round moves a little between the
    two rows without changing the total or the residual.
    """
    start = patient_run["summary"]["slice-averaged"][0]
    mid = patient_run["summary"]["pooled"][0]
    end = slice_run["summary"]["pooled"][0]
    return [
        ("registered: patient-disjoint, slice-averaged", start, None),
        ("+ pooled aggregation", mid, mid - start),
        ("+ slice-level split", end, end - mid),
        ("published target", target, target - end),
    ]


def variance_note(patient_run, slice_run):
    """Between-fold standard deviations, both protocols, every convention."""
    return {key: (patient_run["summary"][key][1], slice_run["summary"][key][1])
            for key in CONVENTIONS}


# ------------------------------------------------------------------ reporting

def _fmt(mean, sd):
    return f"{mean:.4f} ± {sd:.4f}" if not np.isnan(sd) else f"{mean:.4f}"


def convention_table(patient_run, slice_run):
    lines = ["| convention | patient-disjoint (registered) | slice-level (leaking) | difference |",
             "|---|---|---|---|"]
    for key in CONVENTIONS:
        a = patient_run["summary"][key]
        b = slice_run["summary"][key]
        lines.append(f"| {key} | {_fmt(*a)} | {_fmt(*b)} | {b[0] - a[0]:+.4f} |")
    return "\n".join(lines)


def decomposition_table(rows):
    lines = ["| step | Dice | attributable |", "|---|---|---|"]
    for name, value, delta in rows:
        lines.append(f"| {name} | {value:.4f} | "
                     + ("—" if delta is None else f"{delta:+.4f}") + " |")
    lines.append("")
    lines.append(f"Residual after both conventions: {rows[-1][2]:+.4f} "
                 f"({abs(rows[-1][2]) * 100:.1f} Dice points).")
    return "\n".join(lines)


def type_table(patient_types, slice_types):
    lines = ["| tumour type | patient-disjoint | slice-level | difference | zero-Dice (pd → sl) |",
             "|---|---|---|---|---|"]
    for name in sorted(patient_types):
        a, b = patient_types[name], slice_types.get(name)
        if b is None:
            continue
        lines.append(f"| {name} (n={a['n']}) | {a['mean']:.4f} | {b['mean']:.4f} | "
                     f"{b['mean'] - a['mean']:+.4f} | "
                     f"{100 * a['zero_rate']:.1f}% → {100 * b['zero_rate']:.1f}% |")
    return "\n".join(lines)


def variance_table(patient_run, slice_run):
    lines = ["| convention | sd, patient-disjoint | sd, slice-level | ratio |",
             "|---|---|---|---|"]
    for key, (a, b) in variance_note(patient_run, slice_run).items():
        lines.append(f"| {key} | {a:.4f} | {b:.4f} | {a / b:.1f}× |")
    return "\n".join(lines)


# -------------------------------------------------------------------- figures

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })


def protocol_figure(patient_run, slice_run, out_path, target=None,
                    convention="pooled"):
    """Per-fold scores under both protocols: the shift and the variance collapse.

    Points, not bars. These scores sit near 0.8, so a bar from zero renders the
    entire effect as two bars of the same apparent length. The spread is half the
    result here, and only a scaled axis shows it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    series = [("patient-disjoint\n(registered)", patient_run, "#2c7fb8"),
              ("slice-level\n(leaking)", slice_run, "#e6550d")]

    for i, (label, run, colour) in enumerate(series):
        values = np.array([f[convention] for f in run["folds"]], dtype=float)
        ax.scatter(values, np.full(values.size, i), s=34, color=colour, zorder=3,
                   label=f"{label.splitlines()[0]} folds")
        ax.scatter([values.mean()], [i], s=120, marker="|", linewidths=2.4,
                   color="#37474f", zorder=4)
        ax.hlines(i, values.min(), values.max(), color=colour, alpha=0.3,
                  linewidth=7, zorder=2)
        ax.annotate(f"sd {values.std(ddof=1):.4f}",
                    (values.max(), i), textcoords="offset points",
                    xytext=(10, -3), fontsize=8, color=colour)

    if target is not None:
        ax.axvline(target, color="#616161", linestyle="--", linewidth=1,
                   zorder=1)
        ax.annotate(f"published {target:.3f}", (target, -0.62),
                    textcoords="offset points", xytext=(-4, 0), fontsize=8,
                    color="#616161", ha="right")

    ax.set_yticks([0, 1])
    ax.set_yticklabels([s[0] for s in series], fontsize=8)
    ax.set_ylim(-0.75, 1.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"{convention} Dice (higher is better)")
    ax.set_title("same model, same folds, same data — only the split differs",
                 fontsize=9)
    ax.legend(loc="center left", frameon=False, fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def type_gain_figure(patient_types, slice_types, out_path):
    """Where the leak pays: per-type Dice under both protocols."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    names = [n for n in sorted(patient_types) if n in slice_types]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    y = np.arange(len(names))
    for i, name in enumerate(names):
        a, b = patient_types[name]["mean"], slice_types[name]["mean"]
        ax.annotate("", xy=(b, i), xytext=(a, i),
                    arrowprops=dict(arrowstyle="-|>", color="#e6550d",
                                    linewidth=1.6, shrinkA=0, shrinkB=0))
        ax.scatter([a], [i], s=34, color="#2c7fb8", zorder=3,
                   label="patient-disjoint" if i == 0 else None)
        ax.scatter([b], [i], s=34, color="#e6550d", zorder=3,
                   label="slice-level" if i == 0 else None)
        ax.annotate(f"{b - a:+.4f}", ((a + b) / 2, i), textcoords="offset points",
                    xytext=(0, 7), fontsize=8, ha="center", color="#e6550d")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n(n={patient_types[n]['n']})" for n in names],
                       fontsize=8)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("mean Dice, pooled over folds")
    ax.set_title("the leak pays where cases are multi-slice and variable",
                 fontsize=9)
    ax.legend(loc="center left", frameon=False, fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def epochs_gain_figure(patient_results, slice_results, slice_run, patient_run,
                       out_path, convention="pooled"):
    """Gain against epochs trained, to rule out the extra-training confound.

    A slice-level split leaks into validation too, so the leaking arm can stop
    later and train longer. If the extra epochs were doing the work, gain would
    rise with them. It does not: the correlation is weak and negative. At n=5
    that rules the explanation out only loosely, which the title states rather
    than hides.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, gains, labels = [], [], []
    for i, (a, b) in enumerate(zip(patient_run["folds"], slice_run["folds"])):
        row = slice_results.get(i)
        if row is None:
            continue
        epochs.append(float(row["epochs_run"]))
        gains.append(b[convention] - a[convention])
        labels.append(f"f{i}")

    if len(epochs) < 2:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.scatter(epochs, gains, s=42, color="#e6550d", zorder=3)
    for x, y, name in zip(epochs, gains, labels):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 2),
                    fontsize=8, color="#546e7a")
    ax.set_xlabel("epochs trained by the leaking arm")
    ax.set_ylabel(f"gain in {convention} Dice")
    r = np.corrcoef(epochs, gains)[0, 1]
    ax.set_title(f"extra training does not explain the gain "
                 f"(r = {r:+.2f}, n={len(epochs)})", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def results_rows(run_dir):
    """results.csv keyed by fold, for metadata such as epochs_run."""
    out = {}
    for r in read_csv(os.path.join(run_dir, "results.csv")):
        try:
            out[int(r["fold"])] = r
        except (KeyError, TypeError, ValueError):
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", default="runs/baseline",
                        help="run under the registered patient-disjoint split")
    parser.add_argument("--slice-level", default="runs/diag-leaky",
                        help="run under the slice-level (leaking) split")
    parser.add_argument("--target", type=float, default=0.841,
                        help="published figure to decompose the distance to")
    parser.add_argument("--out", default=None,
                        help="directory for figures; omit to print tables only")
    args = parser.parse_args()

    patient = run_conventions(args.patient)
    leaking = run_conventions(args.slice_level)
    if not patient["folds"] or not leaking["folds"]:
        raise SystemExit("Both runs need cells with test_slices.csv.")

    n_p, n_l = len(patient["folds"]), len(leaking["folds"])
    print(f"# Protocol, not model\n")
    print(f"{n_p} folds patient-disjoint ({args.patient}), "
          f"{n_l} folds slice-level ({args.slice_level}).")
    if n_p != n_l:
        print(f"WARNING: fold counts differ ({n_p} vs {n_l}); "
              f"the comparison of means is not like-for-like.")
    print("\n## Both conventions, both splits\n")
    print(convention_table(patient, leaking))
    print("\n## Distance to the published figure\n")
    print(decomposition_table(decomposition(patient, leaking, args.target)))
    print("\n## Between-fold spread\n")
    print(variance_table(patient, leaking))
    print("\nA slice-level split stops sampling patients, so this standard "
          "deviation\nno longer measures patient-level variation.")
    print("\n## Where the gain lands\n")
    p_types, l_types = by_type(args.patient), by_type(args.slice_level)
    print(type_table(p_types, l_types))
    print("\nThe comparison is not paired: a slice-level split reshuffles, so "
          "each fold's\ntest set is near-identical but not identical. No paired "
          "test is reported.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        made = [
            protocol_figure(patient, leaking,
                            os.path.join(args.out, "protocol.png"),
                            target=args.target),
            type_gain_figure(p_types, l_types,
                             os.path.join(args.out, "protocol_by_type.png")),
            epochs_gain_figure(results_rows(args.patient),
                               results_rows(args.slice_level),
                               leaking, patient,
                               os.path.join(args.out, "protocol_epochs.png")),
        ]
        print("\nFigures: " + ", ".join(p for p in made if p))


if __name__ == "__main__":
    main()
