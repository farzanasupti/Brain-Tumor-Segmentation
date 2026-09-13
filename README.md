# Brain-Tumor-Segmentation

Tumour-type-conditioned brain tumour segmentation on T1-weighted contrast-enhanced
MRI, evaluated **patient-disjoint**.

## What this is

Meningioma, glioma and pituitary tumours differ in location, margin and enhancement,
and the Cheng dataset labels every slice with its type. Existing work on this dataset
either ignores those labels or predicts them from a shared encoder alongside
segmentation; none use them to *steer* the segmentation decoder. This repository does,
via FiLM modulation of every decoder stage driven by a predicted type posterior.

Evaluation is patient-disjoint throughout. On this dataset that matters more than
usual: 3064 slices come from only 233 patients — about 13 slices each at 6 mm
thickness — so neighbouring slices are near-duplicates. Under a random slice-level
split, **195 of the 233 patients (83.7%) land in more than one split**, and the test
set measures interpolation between memorised slices rather than generalization.

## Data

Cheng et al., 3064 T1-CE slices, 233 patients, meningioma 708 / glioma 1426 /
pituitary 930. [figshare 1512427 v5](https://doi.org/10.6084/m9.figshare.1512427.v5),
CC BY 4.0.

Use the **original `.mat` files**, not the Kaggle PNG mirror — the mirror drops
`cjdata.PID` and `cjdata.label`, so it supports neither patient-disjoint folds nor
type conditioning.

```bash
pip install torch torchvision monai einops opencv-python h5py numpy pandas tqdm

python fetch_dataset.py --out data/raw            # 4 zips + cvind.mat, 839 MB
python prepare_dataset.py --src data/raw --out data/   # -> images/, masks/, manifest.csv
python folds.py --manifest data/manifest.csv --out data/folds.csv
```

`data/manifest.csv` and `data/folds.csv` are committed, so the splits are reproducible
without re-downloading. The archives read directly — no need to unzip.

## Running an experiment

```bash
# pre-flight: enumerate the grid, check backbones, print split sizes
python experiment.py --manifest data/manifest.csv --folds data/folds.csv \
    --backbones unet cond_unet --arms standard region --dry-run

# run it; --limit lets a session stop cleanly and resume later
python experiment.py --manifest data/manifest.csv --folds data/folds.csv \
    --backbones unet cond_unet --arms standard region --limit 4
```

One cell is a (backbone, condition, augmentation arm, fold) triple. Finished cells are
recorded in `runs/<name>/results.csv` and skipped on the next invocation, so an
interrupted run costs only the cell in flight. Each cell leaves `best.pt`, `last.pt`,
a per-epoch log, and per-slice test scores carrying patient ID and tumour type.

## Layout

| File | Purpose |
|---|---|
| `fetch_dataset.py` | download the figshare archives, ids resolved from the API |
| `prepare_dataset.py` | `.mat` → PNG pairs + `manifest.csv` with patient ID and tumour type |
| `splits.py` | patient-disjoint and slice-level splitting; leakage reporting |
| `folds.py` | stratified group k-fold at patient level, persisted to CSV |
| `augment.py` | augmentation arms: standard, global Bézier, region-wise Bézier |
| `conditioning.py` | FiLM modulation and the tumour-type head |
| `backbones/` | model registry — `unet`, `cond_unet`, `swin_unetr`, `swin_umamba` |
| `dataloading.py` | dataset and loaders |
| `losses.py` | Dice+BCE, joint segmentation + type loss, scoring |
| `engine.py` | training loop: early stopping, checkpointing, resume |
| `experiment.py` | grid runner over backbone × condition × arm × fold |
| `legacy/` | the original TensorFlow U-Net scripts, superseded |

`python -m backbones` reports which backbones this environment can actually build.

## Conventions worth knowing

**Every backbone returns logits.** Pair them with a loss that applies its own sigmoid,
and apply sigmoid exactly once at inference.

**Conditioning has three modes.** `predicted` — a type head reads the bottleneck and
its posterior conditions the decoder; nothing outside the image is needed at inference.
This is the method. `oracle` — ground-truth type conditions the decoder; it leaks a
label and is an upper bound, not a result. `none` — disabled, for the ablation.

**FiLM starts as the identity,** so a conditioned model and a plain one begin as the
same function and any gain is attributable to conditioning rather than to a different
initialisation. Conditioning costs +10,755 parameters, 0.03%.

**Augmentation arms are paired.** Geometric and intensity draws come from separate
seeded streams, so two arms see identical flips and rotations on every item and differ
only in intensity — which is what within-fold contrasts assume.

**Validation is carved from the training folds**, not given a fold of its own: 70/10/20
rather than 60/20/20. A model trained on 60% cannot be compared against a published
number obtained with 80%.

## Tests

```bash
for t in test_augment test_folds test_backbones test_conditioning test_training test_experiment; do
    python $t.py
done
```

113 tests, no pytest dependency. `test_augment`, `test_backbones`, `test_conditioning`,
`test_training` and `test_experiment` need torch; `test_folds` is stdlib only.

## Preprocessing note for reporting

`prepare_dataset.py` scales each 16-bit slice to 8-bit by per-slice min–max, because
window/level varies between scans. Splits are seeded at 42.
