# What a slice-level split does to a brain-tumour segmentation benchmark: a five-point gain, and an error bar that stops measuring patients

**Status:** draft v2, 2026-11-17. Reframed from v1 (2026-09-19), whose central
claim did not survive a check of its source: v1 explained our gap to a published
84.1 Dice as leakage, but that figure (Yurdakul et al. 2026) was measured under a
patient-disjoint split, so leakage cannot explain it. The measurements below are
unchanged and reproducible from committed run artefacts without a GPU
(`python leakage.py --out figures/`). Items marked `[VERIFY]` must be checked
against the primary source before submission.

---

## Abstract

On the Cheng T1-CE brain-tumour dataset, every patient contributes several
near-identical slices, so a split drawn over slices rather than patients places
the same tumour on both sides of the evaluation boundary. Several published
segmentation studies on this dataset split at slice level `[VERIFY]`. We measure
what that choice is worth, holding everything else fixed: one U-Net, the same
data, five folds under a pre-registered patient-disjoint protocol and five under a
slice-level split. The slice-level split raises Dice by **+4.3 points**
slice-averaged (0.7719 → 0.8152) and **+5.1 points** pooled (0.7846 → 0.8358).

Two findings matter more than the mean shift. First, leakage **collapses
between-fold variance** by 3–8× depending on aggregation: once folds stop sampling
patients, the standard deviation reported beside a cross-validated mean no longer
measures patient-level variation, and comparisons judged against it overstate
significance. Second, the gain is **concentrated in glioma** (+7.2 points; slices
scoring zero fall from 133 to 88 of 1426), while meningioma and pituitary move by
under 2 points. We also show that a single fold is not enough to size the effect:
fold 0 alone understated it by three points. We recommend reporting the split
unit and aggregation explicitly, and we note that the reported error bar is itself
weak evidence of which split a paper used.

## 1. Introduction

The Cheng dataset `[CITE: Cheng et al. 2015/2016; figshare 1512427]` (3064 T1-CE
slices from 233 patients; meningioma 708, glioma 1426, pituitary 930) is a
standard testbed for brain-tumour segmentation and classification. It is
distributed as slices, and per-patient slice counts run from 1 to 38. That
organisation makes one mistake easy: a split drawn with an ordinary
`train_test_split` over slices puts neighbouring slices of one tumour into both
training and test, and neighbouring slices of one tumour are nearly the same image.

The literature on this benchmark mixes both kinds of protocol. The related-work
table of Yurdakul et al. (2026, Table 1) lists several prior segmentation studies that
split at slice level (Gupta et al. 2023; Avazov et al. 2024; Rai et al. 2024)
`[VERIFY each against the original paper]`, while their own benchmark uses a
patient-grouped split. Numbers from the two kinds of study are then read on the
same scale. How far apart are they, for an identical model?

We did not set out to answer this. It came out of a pre-registered conditioning
study whose baseline fell short of a published figure, which led us to measure
each reporting convention directly (Section 5 explains how that gap now stands).

**Contributions.**

1. A like-for-like measurement, same model and fold machinery throughout, of what
   a slice-level split is worth on this benchmark: **+4.3** points slice-averaged,
   **+5.1** pooled, **+4.5** patient-averaged.
2. The observation that leakage **collapses between-fold variance** by 3–8×, so a
   leaking cross-validated error bar does not measure what it is reported to
   measure. This holds under every aggregation convention we tested.
3. A per-class decomposition showing the gain is almost entirely glioma and mostly
   the recovery of complete failures, which says where leakage pays and predicts
   where it will pay on other datasets.
4. Evidence that a single fold misjudges a protocol effect: our own fold-0
   prediction, recorded before folds 1–4 ran, was three points low.

## 2. Protocol

**Data.** Cheng et al., figshare 1512427 v5 (CC BY 4.0) `[CITE]`. 3064 slices,
233 patients.

**Patient-disjoint protocol (registered).** Stratified group five-fold at patient
ID, seed 42, committed before any run. Validation is carved from the training
folds as whole patients (70/10/20). U-Net (31.0M parameters), input 192 × 192,
micro-batch 8 with effective batch 16 by accumulation, AdamW (learning rate 1e-4,
weight decay 1e-2), cosine schedule, Dice+BCE with equal weights, at most 150
epochs, early stopping on validation Dice with patience 20, standard geometric
augmentation. Model initialisation is seeded per fold.

**Slice-level protocol.** Identical in every respect except that the five-fold
split is drawn over slices instead of patients. 224 of 233 patients then span more
than one fold, and 194 of 196 test patients are also seen in training. This is
what a naive slice-indexed split produces.

**Aggregation.** *Slice-averaged* Dice is the mean of per-slice scores.
*Pooled* (dataset-level) Dice is `2·ΣI / Σ(P+T)` over all test slices, equal to
the per-slice mean weighted by tumour size. *Patient-averaged* Dice averages within
patient, then across patients. Under a slice-level split every slice is effectively
its own pseudo-patient, so patient-averaged collapses onto slice-averaged; this is
a consistency check on the split rather than a result.

**Scope.** One architecture, one dataset, one seed per fold. We measure protocol,
not a method; the effects apply to whatever is trained under these conventions.

## 3. Results

### 3.1 The split effect under each aggregation

*(Figure: `figures/protocol.png`: per-fold pooled Dice under both protocols.)*

| aggregation | patient-disjoint | slice-level | difference |
|---|---|---|---|
| slice-averaged | 0.7719 ± 0.0164 | 0.8152 ± 0.0035 | +0.0433 |
| pooled | 0.7846 ± 0.0165 | 0.8358 ± 0.0054 | +0.0511 |
| patient-averaged | 0.7699 ± 0.0286 | 0.8152 ± 0.0035 | +0.0453 |

Five folds each side. The comparison is **not paired**: a slice-level split
reshuffles, so each fold's test set is a near-identical but not identical set of
slices. We therefore report differences of means and no paired test.

Aggregation and split compound: moving from the registered convention
(patient-disjoint, slice-averaged) to the most favourable one (slice-level,
pooled) adds 6.4 points to the same model, more than either choice alone. A paper
that does not state both choices leaves its reader unable to place it within that
range.

### 3.2 Leakage collapses between-fold variance

| aggregation | sd, patient-disjoint | sd, slice-level | ratio |
|---|---|---|---|
| slice-averaged | 0.0164 | 0.0035 | 4.8× |
| pooled | 0.0165 | 0.0054 | 3.0× |
| patient-averaged | 0.0286 | 0.0035 | 8.3× |

We consider this the most consequential finding, and it is the one least visible
in a results table. A cross-validated mean is normally reported as `μ ± σ`, and σ
is read as *how much this result would move on a different sample of patients*.
Under a slice-level split the folds no longer sample patients (nearly every patient
is in every fold), so σ measures only which slices of the same tumours landed
where. It shrinks by 3–8×, and a method comparison judged against it will find
differences significant that a patient-level protocol would not. Which aggregation
a paper meant can be argued about; this effect holds under all three.

The collapse also runs in reverse, as weak evidence. On this dataset, at this
scale, a patient-disjoint five-fold U-Net shows σ ≈ 0.016, and a leaking one
σ ≈ 0.004–0.005. A reported σ near the lower value is consistent with a leaking
split; one near the higher value is consistent with a patient-level split. For
example, Yurdakul et al. report 84.1 ± 1.6 for U-Net under a patient-grouped split
`[VERIFY: Table 2]`, in line with our patient-disjoint spread. σ also depends on
the model, the fold count and the number of patients, so this is a consistency
check, not a test, and we do not use it to classify any specific paper.

### 3.3 Where the gain lands

*(Figure: `figures/protocol_by_type.png`.)*

| tumour type | patient-disjoint | slice-level | difference | zero-Dice slices |
|---|---|---|---|---|
| glioma (n=1426) | 0.6627 | 0.7344 | +0.0717 | 9.3% → 6.2% |
| meningioma (n=708) | 0.8940 | 0.9139 | +0.0199 | 1.4% → 1.0% |
| pituitary (n=930) | 0.8464 | 0.8638 | +0.0174 | 0.5% → 0.5% |

Leakage pays where cases are multi-slice, irregular and variable in appearance,
which here means gliomas. It pays almost nothing on the two classes already near
their ceiling. Most of the effect is the recovery of complete failures: glioma
slices scoring exactly zero fall from 133 to 88 of 1426. This predicts that the
size of a leakage effect on another dataset depends on how much within-patient
appearance variation the hard class has, not on dataset size.

### 3.4 Controls

**Extra training (not supported).** Leakage reaches validation as well as test,
so the leaking arm could stop later and simply train more. Across the five folds,
gain and epoch count are weakly negatively correlated (r = −0.21, n = 5). The fold
that trained longest (108 epochs) gained least (+0.0207), and the fold that stopped
soonest (20 epochs, against that baseline fold's 60) gained most (+0.0638). At
n = 5 this rules the explanation out only loosely, and the two extreme folds carry
it; what can be said is that gain does not increase with training.
*(Figure: `figures/protocol_epochs.png`.)*

**One fold is not enough.** We first measured the split effect on fold 0 alone
and recorded a prediction, before running folds 1–4, that the five-fold leaking
figure would be near 0.805 pooled. It was 0.8358. Fold 0 is the *best*
patient-disjoint fold and the *worst* leaking fold, so it understated the protocol
effect by three points. Single-fold diagnostics of protocol effects should not be
trusted, including ours.

## 4. Discussion

**Scores from the two protocols are not on the same scale.** A 4–5 point
difference from the split alone, and up to 6.4 points once aggregation is
included, is as large as many of the improvements reported between architectures
on this benchmark `[VERIFY: cite a representative spread with its source]`.
Rankings that mix slice-level and patient-level studies are therefore uninformative
about the methods being ranked.

**What we are not claiming.** We have not audited any study's code. Where we
describe a published protocol, we rely on that paper's own description or on a
another paper's reading of it, and each such statement is marked for verification. We
claim that the conventions we measured are enough to move an ordinary U-Net by
the amounts shown, that they are often unstated, and that a reader cannot always
tell which was used.

**Recommendation.** State the split unit (patient or slice) and the aggregation
convention. Report patient-averaged Dice alongside slice-averaged, since
per-patient slice counts here run from 1 to 38. Treat a cross-validated σ as
uninformative about generalisation unless the folds sample patients.

## 5. An open discrepancy with a patient-disjoint benchmark

Under the same split unit as Yurdakul et al. (patient-grouped five-fold), our
U-Net reaches 0.7719 slice-averaged against their 84.1, a gap of 6.9 points
(5.6 if their figure is pooled `[VERIFY: their aggregation]`). Leakage cannot
explain this gap, and we do not claim it does. The parameter counts match (31.0M),
as do learning rate, batch size, epoch budget, patience, loss weighting and
schedule. The remaining documented differences are:

| | ours | Yurdakul et al. | status |
|---|---|---|---|
| weight decay | 1e-2 | 1e-4 | **untested** |
| input size | 192 | 256 | tested on fold 0 only: −0.0013 |
| augmentation for the baseline | geometric only | `[VERIFY]` | untested |

Section 3.4 is a reason not to over-read the fold-0 resolution test. A five-fold
run at weight decay 1e-4 is the next experiment and needs roughly 12–15 GPU-hours.
Until it is done, this gap is an open question, reported as such.

## 6. Limitations

One dataset, one architecture, one seed per fold, five folds. The two protocols
are compared unpaired, for the reason given in §3.1. The variance argument in §3.2
compares five standard deviations against five; its direction is large and
consistent across aggregations, but the ratios themselves are imprecise. Section 5
remains unexplained.

## 7. Conclusion

A slice-level split adds four to five Dice points to an identical U-Net on the
Cheng dataset, almost all of it on gliomas, and shrinks the cross-validated error
bar three- to eightfold. The second effect should change practice: a leaking
cross-validated error bar is not a smaller error bar, it is a different quantity.

---

## Before submission

1. **Citations.** Cheng et al. (dataset). Yurdakul, Ersoy, Özger, Pacal, J. Clin.
   Med. 2026, 15:5508, doi 10.3390/jcm15145508 (84.1 ± 1.6, Table 2;
   prior protocols, Table 1). Gupta 2023, Avazov 2024, Rai 2024: confirm from each original
   paper that the split is at slice level, quoting their wording.
2. **The weight-decay run (§5).** Five folds, patient-disjoint, weight decay 1e-4.
   Runs after the registered grid. Either result is reportable: if it closes the
   gap, §5 becomes a short note on sensitivity to weight decay; if not, the gap
   stays open.
3. **Related work.** Not drafted. Needs the leakage and data-hygiene literature in
   medical imaging.
4. **Venue and length.** Not chosen. Drafted at about 4 pages; a longer version
   would add a second architecture or a second dataset.
