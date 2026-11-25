"""Does an augmentation arm write the label outline into the training image?

    CUDA_VISIBLE_DEVICES="" python shortcut_figure.py --manifest data/manifest.csv --out figures/

Region-wise Bezier augmentation remaps tumour and background through separate
curves. With a hard composite, as registered, that puts an intensity step on
the ground-truth boundary of every training image and on none at test time, so
the network can segment by following the step. This measures the step for each
arm on the same 100 training slices as PREREGISTRATION.md (2026-11-17 and
2026-11-18 entries): every 30th of the first 3,000 manifest rows, augmented in
training mode with the default seed, epoch 0.

The step of one slice is |mean of a 1-px ring just inside the mask - mean of a
1-px ring just outside| / image standard deviation, on the channel mean.

Writes
  shortcut.png        (a) one slice under each arm with the true outline,
                      (b) the step on all 100 slices per arm, 1 sd marked;
  shortcut_steps.csv  the per-slice steps behind (b) and Table 3;
and prints median, mean and share above 1 sd per arm.
"""

import argparse
import csv
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figures import TRUTH_COLOUR, _style  # noqa: E402

ARMS = ("standard", "global", "region", "region-bounded")
LABELS = {"standard": "standard", "global": "global", "region": "region",
          "region-bounded": "bounded\nregion"}


def boundary_step(image, mask):
    """|inner-ring mean - outer-ring mean| / image sd, or None without a boundary.

    image: H x W x C float array; mask: H x W (x 1) array, tumour > 0.5.
    """
    m = (mask[..., 0] if mask.ndim == 3 else mask) > 0.5
    g = image.mean(-1) if image.ndim == 3 else image
    k = np.ones((3, 3), np.uint8)
    inner = m & ~cv2.erode(m.astype(np.uint8), k).astype(bool)
    outer = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m
    if inner.sum() == 0 or outer.sum() == 0:
        return None
    return float(abs(g[inner].mean() - g[outer].mean()) / (g.std() + 1e-6))


def sample_rows(manifest, start=0, stop=3000, every=30):
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))[start:stop:every]
    root = os.path.dirname(os.path.abspath(manifest))
    for r in rows:
        for k in ("image", "mask"):
            r[k] = os.path.join(root, r[k])
    return rows


def measure(rows, arms=ARMS, image_size=192):
    """{arm: [(slice_id, step, image, mask), ...]} for slices that have a boundary."""
    from dataloading import SliceDataset
    out = {}
    for arm in arms:
        ds = SliceDataset(rows, arm=arm, training=True, image_size=image_size)
        items = []
        for i, r in enumerate(rows):
            it = ds[i]
            img = it["image"].numpy().transpose(1, 2, 0)
            msk = it["mask"].numpy().transpose(1, 2, 0)
            s = boundary_step(img, msk)
            if s is not None:
                items.append((r["slice_id"], s, img, msk[..., 0]))
        out[arm] = items
    return out


def pick_example(results):
    """The slice whose registered-region step is closest to that arm's median."""
    steps = {sid: s for sid, s, _, _ in results["region"]}
    med = float(np.median(list(steps.values())))
    return min(steps, key=lambda sid: abs(steps[sid] - med))


def draw(results, out_path):
    _style()
    example = pick_example(results)
    # Drawn at the LNCS text width (12.2 cm), so font sizes are the printed sizes.
    fig = plt.figure(figsize=(4.8, 2.45))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.38, 1.75], wspace=0.04, hspace=0.2,
                          left=0.0, right=0.995, top=0.85, bottom=0.17)
    for k, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        sid, s, img, msk = next(x for x in results[arm] if x[0] == example)
        g = img.mean(-1)
        lo, hi = np.percentile(g, [1, 99])
        ax.imshow(g, cmap="gray", vmin=lo, vmax=hi)
        ax.contour(msk > 0.5, levels=[0.5], colors=[TRUTH_COLOUR], linewidths=0.6)
        ax.set_title(f"{LABELS[arm].replace(chr(10), ' ')}: {s:.2f}", fontsize=6.5, pad=1.5)
        if k == 0:
            ax.text(0.0, 1.17, "a", transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax = fig.add_subplot(gs[:, 3])
    rng = np.random.default_rng(0)
    for k, arm in enumerate(ARMS):
        v = np.array([s for _, s, _, _ in results[arm]])
        x = k + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v, s=3, alpha=0.55, color="0.35" if arm != "region" else "#e6550d", linewidths=0)
        med = np.median(v)
        ax.plot([k - 0.28, k + 0.28], [med, med], color="black", linewidth=1.2)
        ax.text(k, 1.01, f"{med:.2f}\n{(v > 1).mean():.0%}", ha="center", va="bottom",
                fontsize=6, transform=ax.get_xaxis_transform())
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=0.7)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LABELS[a] for a in ARMS], fontsize=6)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("boundary step (image sd)", fontsize=7)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.text(-0.3, 1.06, "b", transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
    ax.text(-0.02, 1.01, "median\n> 1 sd", ha="right", va="bottom", fontsize=6, color="0.35",
            transform=ax.transAxes)
    fig.savefig(out_path)
    plt.close(fig)
    return example


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = measure(sample_rows(args.manifest))
    with open(os.path.join(args.out, "shortcut_steps.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "slice_id", "boundary_step_sd"])
        for arm in ARMS:
            for sid, s, _, _ in results[arm]:
                w.writerow([arm, sid, f"{s:.4f}"])
    for arm in ARMS:
        v = np.array([s for _, s, _, _ in results[arm]])
        print(f"{arm:15s} median {np.median(v):.2f}  mean {v.mean():.2f}  frac>1: {(v > 1).mean():.2f}  n={len(v)}")
    example = draw(results, os.path.join(args.out, "shortcut.png"))
    print(f"example slice {example}; wrote {os.path.join(args.out, 'shortcut.png')}")


if __name__ == "__main__":
    main()
