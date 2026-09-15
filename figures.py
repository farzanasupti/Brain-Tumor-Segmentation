"""Figures for the paper, built from what a run already saved.

    python figures.py --run runs/baseline --manifest data/manifest.csv --out figures/

Three figures, each answering a question a reader will ask:

  qualitative.png   what do the predictions actually look like, at the best,
                    median and worst of the test split -- picked by score rather
                    than by eye, so the reader is not shown a curated best case;
  folds.png         every fold plotted with the cell mean, on an axis scaled
                    to the data -- a bar from zero would render the whole effect
                    as two identical-looking bars;
  by_type.png       per-tumour-type scores, since meningioma, glioma and
                    pituitary differ in location and margin.

Nothing is recomputed: predictions are read from each cell's predictions/ and
scores from test_slices.csv, so figures can be regenerated without a GPU.

Figures are greyscale-safe. The prediction overlay is a contour rather than a
fill, so the underlying anatomy stays visible -- a filled mask hides exactly the
boundary the reader is trying to judge.
"""

import argparse
import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # no display on a training box
import matplotlib.pyplot as plt            # noqa: E402

TRUTH_COLOUR = "#2c7fb8"
PRED_COLOUR = "#e6550d"
TYPE_NAMES = {"0": "meningioma", "1": "glioma", "2": "pituitary"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def cells_in(run_dir):
    return sorted(d for d in os.listdir(run_dir)
                  if os.path.isdir(os.path.join(run_dir, d))
                  and os.path.exists(os.path.join(run_dir, d, "test_slices.csv")))


def _load_gray(path, size=None):
    import cv2
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    if size and image.shape != size:
        image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return image


def pick_examples(rows, n=3):
    """Best, median and worst by Dice -- chosen by score, not by eye."""
    scored = [r for r in rows if r.get("dice") not in (None, "")]
    scored.sort(key=lambda r: float(r["dice"]))
    if not scored:
        return []
    if len(scored) <= n:
        return scored
    picks = [("worst", scored[0]),
             ("median", scored[len(scored) // 2]),
             ("best", scored[-1])]
    return picks[:n]


def qualitative(run_dir, cell, manifest_dir, out_path, n=3):
    """A row per example: image, ground truth, prediction over the image."""
    rows = read_csv(os.path.join(run_dir, cell, "test_slices.csv"))
    picks = pick_examples(rows, n)
    if not picks:
        return None

    manifest = {r["slice_id"]: r for r in read_csv(
        os.path.join(manifest_dir, "manifest.csv"))}

    _style()
    fig, axes = plt.subplots(len(picks), 3, figsize=(6.4, 2.2 * len(picks)),
                             squeeze=False)

    for row_index, (tag, row) in enumerate(picks):
        record = manifest.get(row["slice_id"])
        if record is None:
            continue
        image = _load_gray(os.path.join(manifest_dir, record["image"]))
        truth = _load_gray(os.path.join(manifest_dir, record["mask"]), image.shape)
        pred = _load_gray(os.path.join(run_dir, cell, "predictions",
                                       f"{row['slice_id']}.png"), image.shape)

        for col, (title, overlay, colour) in enumerate((
                ("MRI", None, None),
                ("ground truth", truth, TRUTH_COLOUR),
                ("prediction", pred, PRED_COLOUR))):
            ax = axes[row_index][col]
            ax.imshow(image, cmap="gray", interpolation="nearest")
            if overlay is not None and (overlay > 127).any():
                # a contour, not a fill: the boundary is what is being judged
                ax.contour(overlay > 127, levels=[0.5], colors=[colour],
                           linewidths=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            if row_index == 0:
                ax.set_title(title, fontsize=9)

        kind = TYPE_NAMES.get(str(row.get("type")), "")
        axes[row_index][0].set_ylabel(
            f"{tag}\nDice {float(row['dice']):.3f}\n{kind}", fontsize=8)

    fig.suptitle(f"{cell}  -  test split, chosen by score", fontsize=9, y=1.0)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def folds_figure(run_dir, out_path, metric="test_dice", lower_is_better=False):
    """Per-fold points with the cell mean, on an axis scaled to the data.

    Deliberately not a bar chart. These scores cluster near 0.8, and a bar from
    zero renders a 0.05 difference -- the entire result -- as two bars of
    visually identical length. Dots on a truncated axis show the effect and the
    fold-to-fold spread that decides whether it is real.
    """
    rows = read_csv(os.path.join(run_dir, "results.csv"))
    if not rows:
        return None

    by_cell = {}
    for r in rows:
        try:
            value = float(r[metric])
        except (KeyError, TypeError, ValueError):
            continue
        by_cell.setdefault(r["cell"].rsplit("__f", 1)[0], []).append(value)
    if not by_cell:
        return None

    labels = sorted(by_cell, key=lambda k: np.mean(by_cell[k]),
                    reverse=not lower_is_better)

    _style()
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(labels) + 1.8))

    for i, key in enumerate(labels):
        values = np.array(by_cell[key], dtype=float)
        mean = values.mean()
        ax.scatter(values, np.full(values.size, i), s=26, color="#90a4ae",
                   zorder=3, label="individual fold" if i == 0 else None)
        ax.scatter([mean], [i], s=90, marker="|", linewidths=2.2,
                   color="#c62828", zorder=4, label="mean" if i == 0 else None)
        if values.size > 1:
            ax.hlines(i, values.min(), values.max(), color="#cfd8dc",
                      linewidth=6, zorder=2)

    everything = np.concatenate([np.asarray(v, dtype=float) for v in by_cell.values()])
    span = everything.max() - everything.min()
    pad = max(span * 0.25, 0.01)
    ax.set_xlim(everything.min() - pad, everything.max() + pad)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([k.replace("__", " / ") for k in labels], fontsize=8)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric}  ({'lower' if lower_is_better else 'higher'} is better)")
    ax.legend(loc="best", frameon=False, fontsize=8)
    n_folds = max(len(v) for v in by_cell.values())
    ax.set_title(f"per-fold scores and cell mean  (n={n_folds} folds; "
                 f"axis scaled to the data, not to zero)", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def by_type_figure(run_dir, out_path, metric="dice"):
    """Scores split by tumour type, which differ in location and margin."""
    cells = cells_in(run_dir)
    if not cells:
        return None

    grouped = {}
    for cell in cells:
        base = cell.rsplit("__f", 1)[0]
        for r in read_csv(os.path.join(run_dir, cell, "test_slices.csv")):
            try:
                value = float(r[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isnan(value):
                continue
            kind = TYPE_NAMES.get(str(r.get("type")), str(r.get("type")))
            grouped.setdefault(base, {}).setdefault(kind, []).append(value)
    if not grouped:
        return None

    bases = sorted(grouped)
    kinds = sorted({k for g in grouped.values() for k in g})

    _style()
    fig, ax = plt.subplots(figsize=(1.6 * len(kinds) + 2.4, 3.2))
    width = 0.8 / max(len(bases), 1)
    x = np.arange(len(kinds))

    for i, base in enumerate(bases):
        values = [np.mean(grouped[base].get(k, [np.nan])) for k in kinds]
        counts = [len(grouped[base].get(k, [])) for k in kinds]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width * 0.9,
                      label=base.replace("__", " / "), zorder=2)
        for bar, count in zip(bars, counts):
            if count:
                ax.annotate(f"n={count}", (bar.get_x() + bar.get_width() / 2,
                                           bar.get_height()),
                            ha="center", va="bottom", fontsize=6.5,
                            xytext=(0, 1), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels(kinds)
    ax.set_ylabel(metric)
    ax.set_title("by tumour type", fontsize=9)
    if len(bases) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--metric", default="test_dice")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    written = []

    path = folds_figure(args.run, os.path.join(args.out, "folds.png"),
                        args.metric, args.lower_is_better)
    if path:
        written.append(path)

    path = by_type_figure(args.run, os.path.join(args.out, "by_type.png"))
    if path:
        written.append(path)

    for cell in cells_in(args.run):
        target = os.path.join(args.out, f"qualitative_{cell}.png")
        path = qualitative(args.run, cell, manifest_dir, target, args.examples)
        if path:
            written.append(path)

    if not written:
        raise SystemExit(
            f"Nothing to plot in {args.run!r} yet -- no finished cells. "
            f"Figures read saved predictions and scores, so run experiment.py first.")
    for path in written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
