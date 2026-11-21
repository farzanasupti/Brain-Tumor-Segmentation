# Brain-Tumor-Segmentation

[![tests](https://github.com/farzanasupti/Brain-Tumor-Segmentation/actions/workflows/tests.yml/badge.svg)](https://github.com/farzanasupti/Brain-Tumor-Segmentation/actions/workflows/tests.yml)

This is my brain tumour segmentation project on T1-weighted contrast-enhanced MRI. I test whether telling a U-Net what kind of tumour it is looking at helps it draw the tumour better, and I evaluate everything with patients kept apart between training and testing.

## The idea

Meningiomas, gliomas and pituitary tumours look different. They sit in different places, have different edges and enhance differently, and the Cheng dataset labels every slice with its type. As far as I could find, earlier work on this dataset either ignores those labels or predicts them from a shared encoder next to the segmentation. None of it uses the type to *steer* the decoder. That's what I try here: a small head predicts the tumour type, and that prediction modulates every decoder stage through FiLM.

I keep patients separate across splits throughout, and on this dataset that matters a lot. The 3,064 slices come from only 233 patients, about 13 slices each at 6 mm thickness, so neighbouring slices are nearly identical. If you split slices at random, **195 of the 233 patients (83.7%) end up in more than one split**. The test set then mostly measures how well the model fills in between slices it has already memorised, not how it does on a new patient.

I wrote down the hypotheses and the analysis plan in [PREREGISTRATION.md](PREREGISTRATION.md) before I had any results.

## Data

Cheng et al., 3064 T1-CE slices, 233 patients, meningioma 708 / glioma 1426 /
pituitary 930. [figshare 1512427 v5](https://doi.org/10.6084/m9.figshare.1512427.v5),
CC BY 4.0.

Use the **original `.mat` files**, not the Kaggle PNG mirror. The mirror drops
`cjdata.PID` and `cjdata.label`, so you can't build patient-disjoint folds or
condition on type from it.

```bash
pip install -r requirements.txt      # on a CPU-only box, install torch first:
                                     #   pip install torch --index-url https://download.pytorch.org/whl/cpu

python fetch_dataset.py --out data/raw            # 4 zips + cvind.mat, 839 MB
python prepare_dataset.py --src data/raw --out data/   # -> images/, masks/, manifest.csv
python folds.py --manifest data/manifest.csv --out data/folds.csv
```

I've committed `data/manifest.csv` and `data/folds.csv`, so you get the same splits
without downloading anything. The scripts read the archives directly, so there's no
need to unzip them.

## Running an experiment

```bash
# pre-flight: enumerate the grid, check backbones, print split sizes
python experiment.py --manifest data/manifest.csv --folds data/folds.csv \
    --backbones unet cond_unet --arms standard region --dry-run

# run it; --limit lets a session stop cleanly and resume later
python experiment.py --manifest data/manifest.csv --folds data/folds.csv \
    --backbones unet cond_unet --arms standard region --limit 4
```

A cell is one (backbone, condition, augmentation arm, fold) combination. I ran the
whole grid on a 4 GB laptop GPU, so it had to survive interruptions: finished cells go
into `runs/<name>/results.csv` and are skipped next time, and a crash only costs the
cell that was running. Each cell leaves a weights-only `best.pt`, a per-epoch log, per-slice test scores
carrying patient ID and tumour type, and its predicted masks. `last.pt` holds the
optimizer state needed to resume and is deleted once the cell's result is recorded;
`experiment.compact_run(run_dir)` applies the same cleanup to cells that finished
before this behaviour existed.

## Layout

| File | Purpose |
|---|---|
| `fetch_dataset.py` | download the figshare archives, ids resolved from the API |
| `prepare_dataset.py` | `.mat` → PNG pairs + `manifest.csv` with patient ID and tumour type |
| `splits.py` | patient-disjoint and slice-level splitting; leakage reporting |
| `folds.py` | stratified group k-fold at patient level, persisted to CSV |
| `augment.py` | augmentation arms: standard, global Bézier, region-wise Bézier |
| `conditioning.py` | FiLM modulation and the tumour-type head |
| `backbones/` | model registry: `unet`, `cond_unet`, `swin_unetr`, `swin_umamba` |
| `dataloading.py` | dataset and loaders |
| `losses.py` | Dice+BCE, joint segmentation + type loss, Dice/IoU scoring |
| `segmetrics.py` | HD95 and average symmetric surface distance |
| `engine.py` | training loop: early stopping, checkpointing, resume |
| `experiment.py` | grid runner over backbone × condition × arm × fold |
| `analyze.py` | paired contrasts, effect sizes, Holm correction, TOST |
| `contrasts.py` | the paper's contrasts on slice- and patient-averaged Dice |
| `figures.py` | qualitative strips, per-fold plot, per-type breakdown |
| `PREREGISTRATION.md` | the analysis plan, fixed before any result existed |
| `run_tests.py` | one entry point for the test suites, used by CI |
| `legacy/` | the original TensorFlow U-Net scripts, superseded |

`python -m backbones` reports which backbones this environment can actually build.

## Things I decided along the way

**Every backbone returns logits.** Pair them with a loss that applies its own sigmoid,
and apply sigmoid exactly once at inference.

**Conditioning has three modes.** In `predicted`, a type head reads the bottleneck and
its prediction conditions the decoder, so nothing beyond the image is needed at
inference. That's the actual method. In `oracle`, the true type conditions the decoder.
It uses a label the model wouldn't have in practice, so I treat it as an upper bound,
not a result. `none` turns conditioning off for the ablation.

**FiLM starts as the identity.** A conditioned model and a plain one start out as the
same function, so any difference comes from the conditioning and not from a different
starting point. Conditioning adds 10,755 parameters, about 0.03%.

**Augmentation arms are paired.** Geometric and intensity randomness come from separate
seeded streams, so two arms see the same flips and rotations on every image and differ
only in intensity. The within-fold comparisons rely on that.

**Validation comes out of the training folds** instead of getting a fold of its own, so
the split is 70/10/20 rather than 60/20/20. I wanted to compare against a published
number that was trained on 80% of the data, and a model trained on 60% wouldn't be a
fair comparison.

**Boundary distance isn't defined when a mask is empty**, so by default HD95 returns
NaN for those slices. `segmetrics.summarise()` prints how many were undefined next to
the mean, because quietly dropping missed slices makes the score look better than it
is. If you'd rather penalise them, pass `empty_value="diagonal"`, and say which one you
used.

**Micro-batch size is something to report.** Different models fit different batch
sizes in the same memory. On my 4 GB card at 256 px, U-Net fits 8, the conditioned
U-Net 4, and SwinUNETR 8 or more. `--micro-batch` sets what actually fits, and gradient
accumulation makes up the rest of `--batch-size`. That isn't quite the same as a real
larger batch, because U-Net's BatchNorm only sees the micro-batch. So keep
`--micro-batch` the same across cells you want to compare, and report it.

**I keep the predicted masks** under each cell's `predictions/`. If I think of a new
metric after 45 cells have run, I can compute it without retraining anything.

## Tests

```bash
python run_tests.py            # everything this environment supports
python run_tests.py --list     # what would run, and what is missing
python run_tests.py test_folds # one suite
```

There are 181 tests and they don't need pytest. If a suite's dependencies are missing,
it's **skipped and reported**, never quietly counted as passing. `test_folds` runs on
plain Python, and the model suites need torch. Individual files still run directly (`python test_augment.py`).

CI runs on every push and on pull requests to `main`: a fast stdlib job across Python
3.10 and 3.12 that byte-compiles everything and re-derives the committed splits, and a
full job that installs CPU torch and runs all nine suites.

## Analysis

```bash
python analyze.py --run runs/main                         # Dice
python analyze.py --run runs/main --metric test_hd95 --lower-is-better
python analyze.py --run runs/main --margin 0.005          # equivalence testing
python contrasts.py                                       # the numbers in the paper
```

The design is paired. Every cell trains on the same five folds, and the arms see the
same geometric transforms image by image, so each comparison is over five fold-level
*differences*, not two independent samples. With only five, I report effect sizes with
confidence intervals instead of bare p-values, apply Holm correction across the
family of comparisons, and use **TOST before claiming two things are the same**. A
non-significant result from five folds doesn't show there's no difference, so I don't
report it as if it did.

The equivalence margin (±0.005 Dice), the hypotheses, the set of comparisons, the
decision rules and the baseline check are all fixed in
[PREREGISTRATION.md](PREREGISTRATION.md), which I committed before any results existed.
`analyze.py` uses that margin by default and flags the output if you pass a different
one.

`analyze.py` works from `results.csv`, so it sees one pooled test Dice per fold. The paper
reports slice-averaged and patient-averaged Dice instead, rebuilt from each cell's
`test_slices.csv`. `contrasts.py` computes those for the four registered hypotheses
(Holm-corrected together) and, kept apart and uncorrected, for the two exploratory
runs: weight decay $10^{-4}$ and the bounded region arm.

The score tables those numbers come from are committed under `runs/`: each run's
`results.csv` and each cell's per-slice `test_slices.csv` (patient ID, tumour type, predicted
type, Dice, IoU, HD95, ASSD). Checkpoints and predicted masks stay local. So
`python contrasts.py` reproduces the paper's contrasts from a fresh clone without
retraining anything. The runs are the ones named in
[PREREGISTRATION.md](PREREGISTRATION.md): `main` (the registered grid), `baseline`,
`baseline-frozen-aug`, `diag-256`, `diag-leaky`, `wd1e-4` and `region-bounded`.

## Preprocessing note for reporting

`prepare_dataset.py` scales each 16-bit slice to 8-bit by its own min and max, because
window and level vary between scans. Splits use seed 42, and model initialisation is
seeded per fold. Every arm and backbone in a fold starts from the same random draw, so
comparing two arms doesn't also mean comparing two different initialisations.
