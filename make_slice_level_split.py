"""Build the slice-level (patient-leaking) split used as a diagnostic.

Published Dice figures on this dataset are typically obtained by splitting
slices at random, which puts slices from the same patient on both sides of the
split. Our protocol splits at patient level, so the two numbers are not
comparable. This script produces the leaking split so the difference can be
measured rather than argued about.

The trick is to change no training code: each slice becomes its own
pseudo-patient, and folds.make_folds -- already tested, and stratified by tumour
type -- then yields exactly the random slice-level assignment. experiment.py
runs against the two files written here with no flags it does not already have:

    python make_slice_level_split.py
    python experiment.py --manifest data/manifest-slice-level.csv \\
        --folds data/folds-slice-level.csv --folds-only 0 --out runs/diag-leaky

Any result from that run is exploratory: it measures a protocol, not a method.
"""

import argparse
import collections
import csv

import folds as fold_lib
from splits import load_manifest

DEFAULT_MANIFEST = "data/manifest.csv"
DEFAULT_OUT_MANIFEST = "data/manifest-slice-level.csv"
DEFAULT_OUT_FOLDS = "data/folds-slice-level.csv"


def write_pseudo_patient_manifest(manifest_path, out_path):
    """Copy the manifest with pid replaced by slice_id. Returns slice_id -> real pid."""
    with open(manifest_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{manifest_path} holds no slices.")

    fields = list(rows[0])
    true_pid = {row["slice_id"]: row["pid"] for row in rows}
    for row in rows:
        row["pid"] = row["slice_id"]

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return true_pid


def leakage(assignment, true_pid):
    """How many real patients are spread across more than one fold."""
    spans = collections.defaultdict(set)
    for pseudo, fold in assignment.items():
        spans[true_pid[pseudo]].add(fold)
    multi = sum(1 for folds_seen in spans.values() if len(folds_seen) > 1)
    return multi, len(spans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-manifest", default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--out-folds", default=DEFAULT_OUT_FOLDS)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    true_pid = write_pseudo_patient_manifest(args.manifest, args.out_manifest)
    records = load_manifest(args.out_manifest)
    assignment = fold_lib.make_folds(records, n_folds=args.n_folds, seed=args.seed)
    fold_lib.save_folds(args.out_folds, assignment)

    multi, total = leakage(assignment, true_pid)
    print(f"{args.out_manifest}: {len(records)} slices as pseudo-patients")
    print(f"{args.out_folds}: {args.n_folds} folds, seed {args.seed}")
    print(f"real patients spanning more than one fold: {multi}/{total} "
          f"({multi / total * 100:.1f}%) -- this split leaks by construction")


if __name__ == "__main__":
    main()
