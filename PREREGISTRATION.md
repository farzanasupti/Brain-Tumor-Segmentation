# Analysis plan

Registered **2026-09-14**, before any training run completed. The repository
contains no results at this commit — `runs/` is empty and no `results.csv`
exists — so the commit that adds this file timestamps the plan against the data.

Nothing here may be changed after results exist. Anything decided later goes in
"Deviations" at the bottom, with a date and a reason, and is reported as
exploratory rather than confirmatory.

## Question

Does conditioning a segmentation decoder on tumour type improve segmentation of
brain tumours, under patient-disjoint evaluation?

## Data and protocol

Cheng et al., 3064 T1-CE slices, 233 patients (meningioma 708, glioma 1426,
pituitary 930). Stratified group 5-fold at patient ID, `seed=42`, committed as
`data/folds.csv`. Validation is carved from the training folds as whole
patients: 70 / 10 / 20. Input 192 × 192, micro-batch 8, effective batch 16 by
accumulation, AdamW at 1e-4, Dice+BCE, max 150 epochs, early stopping on
validation Dice with patience 20. Model initialisation is seeded per fold, so
every arm within a fold starts from the same draw.

## Outcomes

**Co-primary**, both registered so neither can be chosen after seeing results:

1. **Slice-averaged Dice** on the held-out fold — comparable with the published
   literature on this dataset.
2. **Patient-averaged Dice** — averaged within patient, then across patients.
   More defensible here, because per-patient slice counts run from 1 to 38.

**Secondary:** IoU, HD95, ASSD, and tumour-type classification accuracy.

HD95 is undefined when exactly one mask is empty. Those slices are reported as
`NaN` and counted, never dropped silently; the count is reported beside every
HD95 figure. The `diagonal` substitution is **not** used.

## Hypotheses

| | Claim | Test |
|---|---|---|
| **H1** | Type conditioning on the *predicted* posterior improves Dice over an unconditioned U-Net. | Paired contrast `cond_unet/predicted/standard` − `unet/n-a/standard` |
| **H2** | Conditioning on the *oracle* type gives a further improvement, bounding what a perfect type head could buy. | Paired contrast `oracle` − `predicted` |
| **H3** | Region-wise Bézier augmentation improves over standard augmentation. | Paired contrast `region` − `standard`, within backbone |
| **H4** | Region-wise beats a single global Bézier curve. | Paired contrast `region` − `global`, within backbone |

H1 is the paper's claim. H2–H4 are supporting.

## Statistics

The design is paired: every cell trains on the same five folds, from the same
initialisation, with arms seeing identical geometric transforms item by item. So
each contrast is a paired comparison over five fold-level differences.

- Paired t-test on the five differences, with Cohen's dz and a 95% CI.
- **Holm correction across a family of exactly the four contrasts above** (m=4).
  Any further contrast is exploratory and labelled as such.
- Effect sizes and CIs are reported whether or not p clears 0.05. With n=5 the
  intervals will be wide, and that is reported rather than hidden.

## Equivalence margin

**±0.005 Dice (0.5 Dice points).**

Justification, fixed in advance: the published spread across fourteen backbones
on this dataset is 5.5 Dice points, so 0.5 is about a tenth of the range between
the weakest and strongest model — below what anyone would change a method over.

A claim that two configurations do **not** differ requires TOST against this
margin at p < 0.05. A non-significant t-test alone is absence of evidence and
will not be reported as equivalence.

## Decision rules

Stated now so no outcome can be reinterpreted later.

- **H1 supported** (positive difference, Holm-adjusted p < 0.05): report type
  conditioning as beneficial, with the effect size.
- **H1 equivalent** (TOST p < 0.05 within ±0.005): report that type conditioning
  does **not** help on this dataset. This is a publishable result and will be
  written up as such, not buried.
- **H1 inconclusive** (neither test clears): report as underpowered at n=5, give
  the CI, and draw no directional conclusion.

## Stopping rule

All five folds of a configuration run to completion before its numbers are
looked at. No stopping early because an interim fold looks good, and no adding
folds or seeds to chase significance.

## Baseline credibility gate

Before any hypothesis is tested, the unconditioned U-Net baseline must land
within about **1 Dice point** of the published 84.1 under the same
patient-level protocol. If it does not, the discrepancy is investigated and
reported, and no downstream claim is made until it is explained.

## What is not pre-registered

Cross-domain evaluation and the Mamba backbone are not yet possible here
(`mamba_ssm` does not build, and the model does not fit in 4 GB). If either
happens later it is exploratory, and will say so.

## Deviations

Each entry: date, what changed, why, and whether the affected result becomes
exploratory.

**2026-09-15 — interim folds inspected before all five completed.** The baseline
(`unet`, standard arm) was interrupted after folds 0 and 1 finished so the laptop
could cool. Their test scores were printed to the run log and were then analysed
(per-slice, pooled and per-type Dice) before folds 2–4 ran. The stopping rule above
says all five folds complete before numbers are looked at. What did **not** change:
no fold was added or dropped, no hyperparameter was altered, and folds 2–4 resumed
under the identical command. The baseline's status is unchanged, since the
credibility gate applies to the complete five-fold mean; the interim look is recorded
here so it is not hidden.

**2026-09-15 — credibility gate likely to fail; diagnosis begun, exploratory.**
Interim folds 0–1 sit well below the published 84.1: per-slice Dice 0.760 / 0.718,
pooled (dataset-level) Dice 0.786 / 0.757. Zero-Dice slices concentrate in glioma
(26 of 32; 33 of 51) and in small tumours (median 286 px and 194 px against 546 px and
450 px overall, at 192 × 192 input). Per the gate, no downstream claim is made until
this is explained. Two explanations are under test, both **exploratory**: the Dice
aggregation convention, which accounts for roughly 3–4 points of the gap, and input
resolution, to be checked on a single fold at 256 × 256 once the baseline finishes.
That resolution run departs from the registered 192 × 192 and will be reported as a
diagnostic, not as a result.
