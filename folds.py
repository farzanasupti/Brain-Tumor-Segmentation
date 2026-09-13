"""Stratified group k-fold at patient level, generated once and persisted.

The factorial runs 45 training jobs across 3 backbones and 3 augmentation arms.
Every one of them must see the same folds, or backbone and arm effects get
confounded with split noise and the whole design becomes uninterpretable. So the
assignment is computed once, written to a CSV, and committed. Nothing regenerates
it at training time.

Grouping and stratification
---------------------------
Whole patients go to a single fold -- that is the "group" constraint, and it is
what keeps near-duplicate neighbouring slices out of both train and test.
Within that constraint, tumor type proportions are balanced across folds by
greedy assignment: patients are taken largest-first within each tumor type and
placed in whichever fold currently holds the fewest slices of that type, ties
broken by overall fold size. Patients contribute between a handful and a few
dozen slices each, so balancing on slice count rather than patient count is what
keeps the folds comparable in size.

Train / validation / test
-------------------------
k-fold gives one held-out partition, but training also needs a validation set
for early stopping, and it has to be patient-disjoint from both others. For
fold i out of k:

    test  = fold i
    valid = fold (i+1) mod k
    train = everything else

So across the five folds every patient serves once as test and once as
validation, and no patient is ever in two roles at once.

Stdlib only, so this is testable without a deep learning stack.

Usage:
    python folds.py --manifest data/manifest.csv --out data/folds.csv
    python folds.py --manifest data/manifest.csv --folds data/folds.csv --show 0
"""

import argparse
import csv
import os
import random

import splits as split_lib

DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42


def make_folds(records, n_folds=DEFAULT_N_FOLDS, seed=DEFAULT_SEED):
    """Assign every patient to exactly one fold. Returns {pid: fold_index}."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")

    slices_by_patient = {}
    label_by_patient = {}
    for r in records:
        slices_by_patient.setdefault(r["pid"], []).append(r)
        label_by_patient.setdefault(r["pid"], r["label"])

    for pid, rows in slices_by_patient.items():
        labels = {r["label"] for r in rows}
        if len(labels) > 1:
            raise ValueError(
                f"Patient {pid!r} has more than one tumor type ({sorted(labels)}); "
                f"stratification assumes one type per patient."
            )

    n_patients = len(slices_by_patient)
    if n_patients < n_folds:
        raise ValueError(
            f"{n_patients} patient(s) cannot fill {n_folds} folds."
        )

    rng = random.Random(seed)

    by_label = {}
    for pid, label in label_by_patient.items():
        by_label.setdefault(label, []).append(pid)

    # slices per fold, overall and per tumor type
    fold_total = [0] * n_folds
    fold_label = [{label: 0 for label in by_label} for _ in range(n_folds)]
    assignment = {}

    for label in sorted(by_label):
        patients = sorted(by_label[label])
        rng.shuffle(patients)                      # break size ties at random
        patients.sort(key=lambda p: -len(slices_by_patient[p]))   # largest first

        for pid in patients:
            size = len(slices_by_patient[pid])
            fold = min(
                range(n_folds),
                key=lambda f: (fold_label[f][label], fold_total[f], f),
            )
            assignment[pid] = fold
            fold_label[fold][label] += size
            fold_total[fold] += size

    empty = [f for f in range(n_folds) if fold_total[f] == 0]
    if empty:
        raise ValueError(f"Fold(s) {empty} came out empty.")

    return assignment


def save_folds(path, assignment):
    """Write {pid: fold} to CSV. Commit this file."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pid", "fold"])
        for pid in sorted(assignment):
            writer.writerow([pid, assignment[pid]])


def load_folds(path):
    """Read a fold assignment back. Returns {pid: fold_index}."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"{path!r} is empty.")
    missing = {"pid", "fold"} - set(rows[0])
    if missing:
        raise ValueError(
            f"{path!r} is missing column(s): {', '.join(sorted(missing))}."
        )

    return {r["pid"]: int(r["fold"]) for r in rows}


def n_folds_in(assignment):
    return max(assignment.values()) + 1


def fold_split(records, assignment, fold, n_folds=None, valid_mode="carve",
               valid_frac=0.125, seed=DEFAULT_SEED):
    """Train/valid/test record lists for one fold.

    Test is always fold `fold`. Early stopping also needs a validation set that
    is patient-disjoint from both others, and there are two ways to get one:

      "carve"  (default) hold out whole patients from the remaining folds,
               enough to cover `valid_frac` of their slices. With k=5 and
               valid_frac=0.125 that is roughly 70 / 10 / 20.

      "fold"   spend a whole fold on validation -- test = fold i,
               valid = fold i+1. Simpler, and every patient serves each role
               exactly once across the k folds, but with k=5 it leaves only
               60% of the data for training.

    Prefer "carve" when reproducing published baselines: a model trained on 60%
    of the data will not match a number obtained with 80%, and the reproduction
    check is what makes the rest of the results credible.
    """
    n_folds = n_folds or n_folds_in(assignment)
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}.")
    if valid_mode not in ("carve", "fold"):
        raise ValueError(f"valid_mode must be 'carve' or 'fold', got {valid_mode!r}.")

    unknown = {r["pid"] for r in records} - set(assignment)
    if unknown:
        raise ValueError(
            f"{len(unknown)} patient(s) missing from the fold assignment, "
            f"e.g. {sorted(unknown)[:3]}. Regenerate it from this manifest."
        )

    out = {"train": [], "valid": [], "test": []}

    if valid_mode == "fold":
        valid_fold = (fold + 1) % n_folds
        for r in records:
            f = assignment[r["pid"]]
            role = "test" if f == fold else "valid" if f == valid_fold else "train"
            out[role].append(r)
        return out

    if not 0 < valid_frac < 1:
        raise ValueError(f"valid_frac must be in (0,1), got {valid_frac}.")

    remaining = [r for r in records if assignment[r["pid"]] != fold]
    slices_by_patient = {}
    label_by_patient = {}
    for r in remaining:
        slices_by_patient.setdefault(r["pid"], []).append(r)
        label_by_patient[r["pid"]] = r["label"]

    # Carve per tumor type so validation keeps the same type mix as training.
    # Seeded on the fold as well, so each fold carves a different validation
    # set but any given fold always carves the same one.
    rng = random.Random(f"{seed}:{fold}")
    validation = set()
    by_label = {}
    for pid, label in label_by_patient.items():
        by_label.setdefault(label, []).append(pid)

    for label in sorted(by_label):
        patients = sorted(by_label[label])
        rng.shuffle(patients)
        target = valid_frac * sum(len(slices_by_patient[p]) for p in patients)
        filled = 0
        for pid in patients:
            if filled >= target:
                break
            validation.add(pid)
            filled += len(slices_by_patient[pid])

    if not validation:
        raise ValueError(
            f"valid_frac={valid_frac} carved no validation patients from fold {fold}."
        )

    for r in records:
        if assignment[r["pid"]] == fold:
            out["test"].append(r)
        elif r["pid"] in validation:
            out["valid"].append(r)
        else:
            out["train"].append(r)
    return out


def describe_folds(records, assignment):
    """Per-fold slice, patient and tumor-type counts."""
    n_folds = n_folds_in(assignment)
    rows = [[] for _ in range(n_folds)]
    for r in records:
        rows[assignment[r["pid"]]].append(r)

    labels = sorted({r.get("label_name") or str(r["label"]) for r in records})

    header = f"{'fold':<6}{'slices':>8}{'patients':>10}   " + "  ".join(
        f"{lab:>11}" for lab in labels)
    lines = [header, "-" * len(header)]

    total = len(records)
    for f, rs in enumerate(rows):
        counts = {lab: 0 for lab in labels}
        for r in rs:
            counts[r.get("label_name") or str(r["label"])] += 1
        patients = len({r["pid"] for r in rs})
        cells = "  ".join(f"{counts[lab]:>11}" for lab in labels)
        lines.append(f"{f:<6}{len(rs):>8}{patients:>10}   {cells}")

    sizes = [len(rs) for rs in rows]
    lines.append("")
    lines.append(f"{total} slices, {len(assignment)} patients, {n_folds} folds; "
                 f"largest fold {max(sizes)}, smallest {min(sizes)} "
                 f"({100 * (max(sizes) - min(sizes)) / (total / n_folds):.1f}% spread)")

    missing = [f for f, rs in enumerate(rows)
               if len({r.get("label_name") or str(r["label"]) for r in rs}) < len(labels)]
    if missing:
        lines.append(f"WARNING: fold(s) {missing} are missing a tumor type.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="manifest.csv from prepare_dataset.py")
    parser.add_argument("--out", help="write a new fold assignment here")
    parser.add_argument("--folds", help="read an existing fold assignment instead of generating one")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--show", type=int, default=None,
                        help="also print the train/valid/test split for this fold")
    args = parser.parse_args()

    records = split_lib.load_manifest(args.manifest)

    if args.folds:
        assignment = load_folds(args.folds)
        print(f"Loaded {args.folds}\n")
    else:
        assignment = make_folds(records, args.n_folds, args.seed)
        print(f"Generated {args.n_folds} folds at seed {args.seed}\n")

    print(describe_folds(records, assignment))

    if args.out:
        save_folds(args.out, assignment)
        print(f"\nWrote {args.out} -- commit this file so every run splits identically.")

    if args.show is not None:
        parts = fold_split(records, assignment, args.show)
        print(f"\nFold {args.show} as train/valid/test:\n")
        print(split_lib.describe(parts))


if __name__ == "__main__":
    main()
