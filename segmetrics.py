"""Boundary-distance metrics: HD95 and average symmetric surface distance.

Dice alone will not separate the models in this study. The published spread
across fourteen backbones on this dataset is 5.5 Dice points, while HD95 across
the same models ranges 9.2 to 5.9 pixels -- boundary distance is the more
sensitive measure, and it is the one an intensity augmentation should move.

Computed from the boundary pixels of each mask via Euclidean distance
transforms, which needs only scipy, not medpy.

The empty-mask convention
-------------------------
Boundary distance is undefined when either mask is empty, and this dataset
contains slices where the model predicts nothing. Silently dropping those
inflates the score, because the easiest failures disappear from the average. So
the convention is explicit and reported alongside the number:

    both empty          0.0    the model correctly found no tumour
    one empty           `empty_value`, default NaN

With NaN, aggregate with `summarise()`, which reports how many slices were
undefined next to the mean. Pass empty_value="diagonal" instead to substitute
the image diagonal -- the largest distance possible in the frame -- which keeps
every slice in the average at the cost of a harsh, arbitrary penalty. State
which you used when reporting.
"""

import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    HAVE_SCIPY = True
except ImportError:                                  # reported, not raised
    HAVE_SCIPY = False


def check():
    if not HAVE_SCIPY:
        return False, "scipy not installed. Run: pip install scipy"
    import scipy
    return True, f"scipy {scipy.__version__}"


def _as_bool(mask):
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {np.shape(mask)}.")
    return mask > 0.5


def boundary(mask):
    """Pixels of `mask` that touch the background -- its one-pixel-wide border.

    Accepts float or boolean masks; anything above 0.5 counts as foreground.
    """
    mask = _as_bool(mask)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def surface_distances(pred, target, spacing=1.0):
    """Distances from each predicted boundary pixel to the nearest target
    boundary pixel, and vice versa. Returns (pred_to_target, target_to_pred)."""
    if not HAVE_SCIPY:
        raise ImportError("boundary metrics need scipy. Run: pip install scipy")

    pred, target = _as_bool(pred), _as_bool(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}.")

    pred_border, target_border = boundary(pred), boundary(target)
    if not pred_border.any() or not target_border.any():
        return np.array([]), np.array([])

    # distance_transform_edt measures distance to the nearest zero, so invert
    to_target = distance_transform_edt(~target_border, sampling=spacing)
    to_pred = distance_transform_edt(~pred_border, sampling=spacing)
    return to_target[pred_border], to_pred[target_border]


def _empty_case(pred, target, empty_value, shape):
    """Score for a pair where at least one mask is empty, or None if both are full."""
    pred_empty, target_empty = not pred.any(), not target.any()
    if not (pred_empty or target_empty):
        return None
    if pred_empty and target_empty:
        return 0.0
    if empty_value == "diagonal":
        return float(np.hypot(*shape))
    return float("nan") if empty_value is None else float(empty_value)


def hd95(pred, target, spacing=1.0, empty_value=None):
    """95th-percentile symmetric Hausdorff distance, in pixels.

    The 95th percentile rather than the maximum, because a single stray pixel
    should not define the score.
    """
    pred_b, target_b = _as_bool(pred), _as_bool(target)
    special = _empty_case(pred_b, target_b, empty_value, pred_b.shape)
    if special is not None:
        return special

    forward, backward = surface_distances(pred_b, target_b, spacing)
    if forward.size == 0 or backward.size == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def assd(pred, target, spacing=1.0, empty_value=None):
    """Average symmetric surface distance, in pixels."""
    pred_b, target_b = _as_bool(pred), _as_bool(target)
    special = _empty_case(pred_b, target_b, empty_value, pred_b.shape)
    if special is not None:
        return special

    forward, backward = surface_distances(pred_b, target_b, spacing)
    if forward.size == 0 or backward.size == 0:
        return float("nan")
    return float(np.concatenate([forward, backward]).mean())


def batch_hd95(preds, targets, spacing=1.0, empty_value=None):
    """HD95 for a batch of masks, shaped (B, 1, H, W) or (B, H, W)."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return [hd95(preds[i], targets[i], spacing, empty_value)
            for i in range(len(preds))]


def summarise(values, name="hd95"):
    """Mean over the defined values, with the undefined ones counted, not hidden."""
    values = np.asarray(values, dtype=float)
    defined = values[~np.isnan(values)]
    return {
        f"{name}": float(defined.mean()) if defined.size else float("nan"),
        f"{name}_median": float(np.median(defined)) if defined.size else float("nan"),
        f"{name}_n": int(defined.size),
        f"{name}_undefined": int(values.size - defined.size),
    }
