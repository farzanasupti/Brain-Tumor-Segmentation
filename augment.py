"""Augmentation arms for the backbone x augmentation factorial.

Three arms, matching the experiment protocol. Every arm applies the same
geometric augmentation; the Bezier arms add an intensity transform on top.

    "standard"  flip / rotate / scale only                  (control)
    "global"    standard + one Bezier curve over the image  (ablation)
    "region"    standard + separate curves for tumor and background

The control arm deliberately includes geometric augmentation. A control with no
augmentation at all would measure "augmentation vs none" and say nothing about
Bezier specifically.

Intensity transform
-------------------
A cubic Bezier curve with endpoints pinned at (0,0) and (1,1) and two free
control points defines a transfer function on normalized intensities. The curve
is sampled parametrically and evaluated by interpolation, which needs x(t) to be
single-valued, so the x-coordinates of the control points are kept ordered:

    x0 = 0 <= x1 <= x2 <= x3 = 1

which makes x'(t) a sum of non-negative terms, hence x(t) non-decreasing.

Two regimes, drawn with probability p_monotonic:

    monotonic      order-preserving: a smooth contrast-like remap, 0 -> 0 and
                   1 -> 1.
    non-monotonic  order-reversing: the same curve read against inverted
                   intensities, so 0 -> 1 and 1 -> 0. Bright and dark swap
                   while the curve's shape still controls the midtones. This is
                   the stronger perturbation used for domain randomization.

On the word "non-monotonic". A cubic Bezier with endpoints pinned at (0,0) and
(1,1) and both y-control points inside [0,1] can never fold. Writing
y'(t)/3 = a*u^2 + 2b*u*v + c*v^2 with u = 1-t, v = t, a = y1, b = y2-y1 and
c = 1-y2, a fold requires b < 0 and b^2 > a*c; maximising (y1-y2)^2 - y1(1-y2)
over 0 <= y2 < y1 <= 1 gives 0, attained only at the corners. So a genuine fold
is impossible in the unit square, and admitting y-controls outside it produces
folds that are both rare and shallow -- measured at roughly 4% of draws and
0.001 in depth for y in [-0.6, 1.6], with heavy clipping before the rate becomes
useful. Order reversal is what this augmentation family has always meant by
"non-monotonic", and it is what is implemented here.

Output stays in [0,1] without clipping: by the convex hull property a Bezier
curve lies within the hull of its control points, and all four y-coordinates are
drawn from [0,1].

Region-wise application perturbs lesion-to-background contrast while leaving
mask geometry untouched -- the mask never passes through the intensity path.

Reference: Region-Wise Bezier Intensity Augmentation for Domain-Generalized
Brain Tumor Segmentation with a Mamba U-Net, J. Clin. Med. 15(14):5508, 2026.

Augmentation is training-only. Call with training=False at validation and test
time and the input is returned untouched.
"""

import numpy as np

ARMS = ("standard", "global", "region")

#: Samples along the curve. 256 is ample for 8-bit-derived data; the cost is a
#: one-off array build per curve, not per pixel.
CURVE_SAMPLES = 256


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def sample_bezier_curve(rng=None, monotonic=True, n_samples=CURVE_SAMPLES):
    """Sample a cubic Bezier transfer function.

    Returns (xs, ys) with xs strictly increasing, ready for np.interp.
    Monotonic curves run (0,0) -> (1,1); non-monotonic ones run (0,1) -> (1,0).
    """
    rng = _as_rng(rng)

    # Ordered x controls keep x(t) non-decreasing, so the curve stays a
    # single-valued function of intensity. Ordered y controls keep the
    # monotonic regime genuinely order-preserving.
    x1, x2 = np.sort(rng.uniform(0.0, 1.0, size=2))
    y1, y2 = np.sort(rng.uniform(0.0, 1.0, size=2))

    t = np.linspace(0.0, 1.0, n_samples)
    mt = 1.0 - t
    b1, b2, b3 = 3 * mt ** 2 * t, 3 * mt * t ** 2, t ** 3

    xs = b1 * x1 + b2 * x2 + b3          # x0 = 0, x3 = 1
    ys = b1 * y1 + b2 * y2 + b3          # y0 = 0, y3 = 1

    # x(t) is non-decreasing analytically but can tie at float precision near
    # the endpoints; np.interp wants strictly increasing xs.
    if np.any(np.diff(xs) <= 0):
        xs = np.maximum.accumulate(xs + np.arange(n_samples) * 1e-12)
    xs[0], xs[-1] = 0.0, 1.0
    ys[0], ys[-1] = 0.0, 1.0

    if not monotonic:
        ys = ys[::-1].copy()             # order-reversing: 0 -> 1, 1 -> 0

    return xs, ys


def apply_transfer(image, xs, ys):
    """Map intensities through a sampled transfer function.

    `image` is expected in [0,1]; values outside are clamped to the endpoint
    values by np.interp, which is the intended behaviour.
    """
    return np.interp(image, xs, ys).astype(np.float32, copy=False)


def bezier_global(image, rng=None, p_monotonic=0.5):
    """One curve applied to the whole image."""
    rng = _as_rng(rng)
    xs, ys = sample_bezier_curve(rng, monotonic=rng.random() < p_monotonic)
    return apply_transfer(image, xs, ys)


def bezier_region_wise(image, mask, rng=None, p_monotonic=0.5):
    """Independent curves for tumor and background.

    `mask` is treated as binary at 0.5 and broadcast against `image`, so a
    (H, W, 1) mask pairs with an (H, W, 3) image. The mask itself is neither
    modified nor transformed.
    """
    rng = _as_rng(rng)

    xs_t, ys_t = sample_bezier_curve(rng, monotonic=rng.random() < p_monotonic)
    xs_b, ys_b = sample_bezier_curve(rng, monotonic=rng.random() < p_monotonic)

    tumor = apply_transfer(image, xs_t, ys_t)
    background = apply_transfer(image, xs_b, ys_b)

    return np.where(np.asarray(mask) > 0.5, tumor, background).astype(
        np.float32, copy=False)


def standard_augment(image, mask, rng=None, max_rotation=15.0, max_scale=0.10,
                     p_flip=0.5):
    """Horizontal flip, rotation and scale, applied identically to both.

    Bilinear for the image, nearest for the mask -- interpolating a binary mask
    invents fractional labels at the tumor boundary.
    """
    import cv2  # local, so the Bezier path stays importable without OpenCV

    rng = _as_rng(rng)

    if rng.random() < p_flip:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    angle = float(rng.uniform(-max_rotation, max_rotation))
    scale = float(1.0 + rng.uniform(-max_scale, max_scale))

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), angle, scale)

    warped_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # warpAffine drops a trailing axis of size 1; restore the caller's shape.
    if warped_image.ndim == image.ndim - 1:
        warped_image = warped_image[..., None]
    if warped_mask.ndim == mask.ndim - 1:
        warped_mask = warped_mask[..., None]

    return warped_image, warped_mask


class Augmenter:
    """One augmentation arm of the factorial.

        aug = Augmenter("region", seed=42)
        x, y = aug(image, mask, training=True)

    Order is geometric first, then intensity, so the intensity transform acts on
    final pixel values rather than on values that resampling will then blend.

    Geometric and intensity draws come from separate streams spawned off `seed`.
    Two arms built with the same seed therefore see an identical sequence of
    flips, rotations and scales, and differ only in the intensity transform --
    which makes the arms paired within fold and is what the analysis plan's
    within-fold contrasts assume.
    """

    def __init__(self, arm="standard", p_monotonic=0.5, p_bezier=1.0, seed=None):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}.")
        if not 0.0 <= p_bezier <= 1.0:
            raise ValueError(f"p_bezier must be in [0,1], got {p_bezier}.")
        if not 0.0 <= p_monotonic <= 1.0:
            raise ValueError(f"p_monotonic must be in [0,1], got {p_monotonic}.")

        self.arm = arm
        self.p_monotonic = p_monotonic
        self.p_bezier = p_bezier

        geom_seed, intensity_seed = np.random.SeedSequence(seed).spawn(2)
        self.geom_rng = np.random.default_rng(geom_seed)
        self.intensity_rng = np.random.default_rng(intensity_seed)

    def __call__(self, image, mask, training=True):
        if not training:
            return image, mask

        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        image, mask = standard_augment(image, mask, self.geom_rng)

        if self.arm != "standard" and self.intensity_rng.random() < self.p_bezier:
            if self.arm == "global":
                image = bezier_global(image, self.intensity_rng, self.p_monotonic)
            else:
                image = bezier_region_wise(image, mask, self.intensity_rng,
                                           self.p_monotonic)

        return image, mask

    def __repr__(self):
        return (f"Augmenter(arm={self.arm!r}, p_monotonic={self.p_monotonic}, "
                f"p_bezier={self.p_bezier})")
