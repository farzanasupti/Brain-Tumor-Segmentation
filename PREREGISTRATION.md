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

**2026-09-15 — augmentation was frozen in every baseline fold; baseline rerun.**
`make_loader` kept its worker processes alive between epochs. Each worker holds
its own copy of the dataset, so `set_epoch()` never reached them, and every epoch
replayed the augmentation drawn at the epoch the workers started. Folds 0–3
therefore trained on one fixed augmented copy of the training set for their whole
run. Fold 4 used epoch 0's copy until a lid-close suspend hung the process at
epoch 28, and epoch 29's copy after the restart. The protocol assumes augmentation
is redrawn each epoch, so none of these five folds follows it. The bug surfaced
because training loss doubled on resume while validation Dice held. Unit tests
had all used `num_workers=0`, where the copy is shared, so they could not show it.

The fix redraws augmentation every epoch, and a regression test covers it with
real worker processes. The baseline is rerun from scratch under the identical
command, seeds and folds, into `runs/baseline`. The frozen run is kept as
`runs/baseline-frozen-aug` and reported only as exploratory. All interim numbers
in the two entries above come from that frozen run. No hyperparameter changed.
Its test scores have already been seen, so the rerun is not blind; this is recorded
here rather than hidden.

**2026-09-16 — the credibility gate failed; two exploratory diagnostics opened.**
The corrected five-fold baseline scored **0.7719 ± 0.0164** slice-averaged Dice and
**0.7699 ± 0.0286** patient-averaged, with pooled Dice 0.7846. Against the published
84.1 that is 6.9 points short (5.6 on the pooled convention), so the gate fails and
no hypothesis is tested until the gap is explained. Fixing the frozen augmentation
was worth +0.031 Dice per fold (95% CI +0.023 to +0.039, n=4), which closes part of
the earlier gap but not this one.

The loss is concentrated in glioma: mean Dice 0.663 with 9.3% zero-Dice slices,
against 0.894 meningioma and 0.846 pituitary. Tumour size explains less than first
suspected — the smallest size quartile would add only about 1.1 points if it scored
like the rest — so resolution is unlikely to be the main cause.

Two diagnostics, both **exploratory**, neither a claim about any method:

1. **Input resolution**, fold 0 at 256 × 256 (`runs/diag-256/`), departing from the
   registered 192 × 192.
2. **Split protocol**, fold 0 under a slice-level split that puts slices from the
   same patient on both sides (`runs/diag-leaky/`, built by
   `make_slice_level_split.py`; 224 of 233 patients span more than one fold, and
   194 of 196 test patients are also seen in training). Published figures on this
   dataset are generally obtained this way, so this measures how much of the gap is
   protocol rather than model. A leaking number is never reported as our result.

**2026-09-16 — diagnostic 1 (resolution) came back negative.** Fold 0 at 256 × 256
scored 0.7901 against 0.7914 at the registered 192 × 192: a difference of −0.0013 on
the same 613 test slices. It did not help where it was supposed to. Small gliomas,
the hypothesised victims of low resolution, got worse (−0.0287, with zero-Dice
slices rising from 19 to 27 of 143), while pituitary gained (+0.0138). Resolution is
therefore ruled out as the explanation for the 6.9-point gap, and the registered
192 × 192 stands for all confirmatory runs.

**2026-09-16 — diagnostic 2 (split protocol) came back positive: it accounts for
about 2.7 of the 6.9 points, and not more.** Fold 0 under the slice-level split
scored 0.8185 against 0.7914 under the registered patient-disjoint split, a
difference of +0.0271. The gain is entirely in glioma: 0.6908 → 0.7518 (+0.0610,
zero-Dice slices 20 → 9 of 286), while meningioma (0.9066 → 0.9016) and pituitary
(0.8585 → 0.8573) do not move. Boundary metrics shift with it — HD95 10.51 → 7.30 px,
undefined 9 → 4. This is the signature leakage would be expected to produce: gliomas
are the multi-slice, most variable cases, where a neighbouring slice from the same
patient is worth most, and the two types that gain nothing are the two already near
their ceiling.

Two caveats, recorded so the number is not over-read. The comparison is **not
paired**: the slice-level split reshuffles, so the 613 test slices are a
near-identical but not identical set (142 / 285 / 186 by type against 142 / 286 / 185).
And the leak reaches validation as well — best validation Dice 0.8301 against
0.7686 — so early stopping ran 108 epochs instead of 62. The leaking arm therefore
got both an easier test set and more training, and +0.0271 is an upper bound on what
the protocol alone is worth.

This does not close the gap. Carrying the fold-0 shift onto the five-fold mean of
0.7719 would reach about 0.799, still roughly 4 points short of 84.1; fold 0's own
leaking score is 2.3 points short. Protocol explains part of the discrepancy,
resolution none, and about 4 points remain unexplained. **The credibility gate stays
failed and no hypothesis is tested.** No confirmatory run adopts the slice-level
split: the registered patient-disjoint protocol stands, and this number is reported
only as a measurement of how published figures on this dataset are obtained.

**2026-09-17 — the two reporting conventions compound; exploratory re-analysis of
saved predictions.** No new training: slice-averaged, pooled and patient-averaged
Dice were recomputed from the per-slice test scores already on disk.

| run | slice-avg | pooled | patient-avg |
|---|---|---|---|
| baseline, five-fold mean | 0.7719 | 0.7846 | 0.7699 |
| baseline, fold 0 | 0.7914 | 0.8088 | 0.8033 |
| slice-level split, fold 0 | 0.8185 | **0.8295** | 0.8185 |

The leaking split *under the pooled convention* reaches 0.8295, 1.15 points from the
published 84.1. Neither choice alone comes close; together they nearly close the gap.
Fold 0 flatters, though — it is the best of the five, +0.0242 above the pooled mean —
so a five-fold leaking pooled number should be expected nearer **0.805, about 3.6
points short**, and the folds 1–4 runs started 2026-09-17 21:40 will settle it.

That the slice-level split's patient-averaged Dice equals its slice-averaged exactly
(613 pseudo-patients for 613 slices) is a consistency check on
`make_slice_level_split.py`, not a result.

**This casts doubt on the gate as written.** It fixes a single published scalar as the
target, but that scalar's split protocol is now known to differ from ours and its
aggregation convention is not stated. Depending on which pair of choices 84.1 encodes,
this baseline is between 1.2 and 6.9 points short — a range wider than the 5.5-point
spread across fourteen backbones that justifies the equivalence margin above. The gate
is not being quietly relaxed: it stays failed, and nothing downstream is claimed. But
the honest form of the remaining question is whether a like-for-like target exists at
all, and that will be reported rather than resolved by picking the convention that
flatters.

**2026-09-17 — bookkeeping defect in `results.csv`, no effect on any score.** Baseline
fold 4 resumed from `last.pt` at epoch 21. Its `epochs_run` (39) and `seconds` (5408)
count only the post-resume session, not the cell: the log holds 60 epochs totalling
8535 s. The other four folds are unaffected, having never resumed. `best_valid_dice`,
the test metrics and `best.pt` all correspond to the true best epoch, so no reported
Dice changes; only those two metadata columns are wrong, and they must not be used for
a training-cost table until `experiment.py` accumulates them across resumes.

**2026-09-19 — diagnostic 2 completed on all five folds: the gap is fully accounted
for, and the prediction on record was wrong.** The slice-level split was run on folds
1–4 under the identical command (`runs/diag-leaky/`, five folds, seed 42, 192 × 192).

| convention | patient-disjoint (registered) | slice-level (leaking) | difference |
|---|---|---|---|
| slice-averaged | 0.7719 ± 0.0164 | 0.8152 ± 0.0035 | +0.0433 |
| pooled | 0.7846 ± 0.0165 | **0.8358 ± 0.0054** | +0.0511 |
| patient-averaged | 0.7699 ± 0.0286 | 0.8152 ± 0.0035 | +0.0453 |

The 2026-09-17 entry predicted about 0.805 pooled and 3.6 points still missing. The
five-fold leaking pooled figure is **0.8358, 0.52 points from the published 84.1** —
inside the gate's 1-point tolerance, though not under the gate's protocol. The
prediction failed because fold 0 flatters the baseline and penalises the leaking arm:
it is the best baseline fold (+0.0242 pooled above the baseline mean) and the *worst*
leaking fold (−0.0063 below the leaking mean). Its +0.0207 pooled gain is the smallest
of the five; folds 1, 2 and 4 gain +0.062, +0.062 and +0.064. Extrapolating a
protocol effect from one fold understated it by three points, which is the general
lesson, not a detail about this dataset.

Taking the two conventions in order, the 6.9-point shortfall decomposes as
**+1.3 points of aggregation** (slice-averaged → pooled, on the registered split)
and **+5.1 points of split protocol** (patient-disjoint → slice-level, pooled),
leaving 0.5. Nothing beyond the two reporting choices needs to be invoked.

Two further observations, both from the completed folds:

* **The extra-training confound is dead.** The 2026-09-16 entry warned that the
  leaking arm also trained longer, because the leak reaches validation. Across five
  folds the gain runs *opposite* to epoch count: fold 0 trained longest (108 epochs)
  and gained least (+0.0207), while fold 4 stopped at 20 epochs — against that
  baseline fold's 60 — and gained most (+0.0638). The gain is the leak, not the
  epochs.
* **Leakage collapses between-fold variance**, from ±0.0165 pooled to ±0.0054, and
  ±0.0164 slice-averaged to ±0.0035. With slices from the same patient on both sides,
  the fold split stops sampling patients, so the patient-level variation that a
  five-fold standard deviation is meant to capture disappears. A leaking five-fold
  result is therefore not merely optimistic in the mean; its error bar is not
  measuring what it is reported to measure. That is a stronger statement than the
  mean shift and it does not depend on which aggregation convention a paper uses.

The gain remains concentrated in glioma (0.6627 → 0.7344, zero-Dice slices 9.3% →
6.2% of 1426), with meningioma (0.8940 → 0.9139) and pituitary (0.8464 → 0.8638)
moving little — the fold-0 signature, now confirmed at five folds. The comparison is
still **not paired**: the slice-level split reshuffles, so each fold's test set is a
near-identical but not identical set of slices.

**Gate status.** As written the gate is not met: under the registered patient-disjoint
protocol the baseline is 0.7719 slice-averaged, 6.9 points short, and no rerun changes
that. What the gate also requires — that a shortfall be "investigated and reported"
before anything downstream is claimed — is now satisfied. The discrepancy is explained
in full by two reporting conventions that the published number is likely to use and
that this protocol deliberately does not, with resolution ruled out (2026-09-16) and
nothing left over. On that basis the registered hypotheses may be tested, under the
patient-disjoint protocol and the registered slice-averaged Dice, with every number
reported against this baseline rather than against 84.1. No confirmatory run adopts
the slice-level split or the pooled convention, and 84.1 is not used as a comparison
target anywhere, because it is now known to be measured differently.
