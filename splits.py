"""Train/valid/test splitting for the Cheng brain tumor dataset.

Two protocols are provided on purpose:

  patient  -- whole patients are assigned to one split. This is the honest
              protocol: the test set contains patients the model never saw.

  slice    -- slices are shuffled individually, ignoring patient identity. This
              is what the Kaggle PNG mirror forces on you and what most
              published work on this dataset does. With ~13 slices per patient
              at 6 mm thickness, neighbouring slices are near-duplicates, so
              nearly every test slice has a sibling in the training set and the
              reported score is inflated.

Keep both. The gap between them is a result, not an accident.

Stdlib only, so this is testable without a deep learning stack.

Usage:
    python splits.py --manifest data/manifest.csv --protocol patient
    python splits.py --manifest data/manifest.csv --protocol slice
"""

import argparse
import csv
import os
import random

SPLIT_NAMES = ("train", "valid", "test")


def load_manifest(path):
    """Read manifest.csv as written by prepare_dataset.py."""
    with open(path, newline="") as fh:
        records = list(csv.DictReader(fh))

    if not records:
        raise ValueError(f"{path!r} is empty.")

    required = {"slice_id", "pid", "label", "image", "mask"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(
            f"{path!r} is missing column(s): {', '.join(sorted(missing))}. "
            f"Regenerate it with prepare_dataset.py."
        )

    for r in records:
        r["label"] = int(r["label"])
    return records


def resolve_paths(records, root):
    """Turn the manifest's relative image/mask paths into absolute ones."""
    out = []
    for r in records:
        r = dict(r)
        r["image"] = os.path.join(root, r["image"])
        r["mask"] = os.path.join(root, r["mask"])
        out.append(r)
    return out


def patient_disjoint_split(records, valid_frac=0.1, test_frac=0.1, seed=42):
    """Assign whole patients to splits, keeping the tumor type mix comparable.

    Patients are grouped by tumor type and allocated within each type, so all
    three types appear in every split in roughly their dataset proportion.
    Allocation is greedy by slice count: patients have between a handful and a
    few dozen slices each, so splitting on patient *count* would give splits of
    quite different sizes.
    """
    _check_fracs(valid_frac, test_frac)
    rng = random.Random(seed)

    slices_by_patient = {}
    label_by_patient = {}
    for r in records:
        slices_by_patient.setdefault(r["pid"], []).append(r)
        label_by_patient[r["pid"]] = r["label"]

    for pid, rows in slices_by_patient.items():
        labels = {r["label"] for r in rows}
        if len(labels) > 1:
            raise ValueError(
                f"Patient {pid!r} has more than one tumor type ({sorted(labels)}); "
                f"stratification assumes one type per patient."
            )

    patients_by_label = {}
    for pid, label in label_by_patient.items():
        patients_by_label.setdefault(label, []).append(pid)

    assignment = {}
    for label in sorted(patients_by_label):
        patients = sorted(patients_by_label[label])
        rng.shuffle(patients)

        total = sum(len(slices_by_patient[p]) for p in patients)
        targets = {"test": test_frac * total, "valid": valid_frac * total}

        taken = 0
        for split in ("test", "valid"):
            target = targets[split]
            filled = 0
            while taken < len(patients) and filled < target:
                pid = patients[taken]
                assignment[pid] = split
                filled += len(slices_by_patient[pid])
                taken += 1
        for pid in patients[taken:]:
            assignment[pid] = "train"

    splits = {name: [] for name in SPLIT_NAMES}
    for r in records:
        splits[assignment[r["pid"]]].append(r)

    for name, rows in splits.items():
        if not rows:
            raise ValueError(
                f"The {name!r} split came out empty -- too few patients for "
                f"valid_frac={valid_frac}, test_frac={test_frac}."
            )
    return splits


def slice_level_split(records, valid_frac=0.1, test_frac=0.1, seed=42):
    """The leaky protocol: shuffle slices, ignore patient identity.

    Provided so the inflation it causes can be measured directly. Do not report
    numbers from this protocol as generalization performance.
    """
    _check_fracs(valid_frac, test_frac)
    rng = random.Random(seed)

    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    if n_test < 1 or n_valid < 1:
        raise ValueError(
            f"Only {n} slice(s); valid_frac={valid_frac}, test_frac={test_frac} "
            f"leaves an empty split."
        )

    return {
        "test": shuffled[:n_test],
        "valid": shuffled[n_test:n_test + n_valid],
        "train": shuffled[n_test + n_valid:],
    }


def _check_fracs(valid_frac, test_frac):
    if not (0 < valid_frac < 1 and 0 < test_frac < 1):
        raise ValueError("valid_frac and test_frac must each be between 0 and 1.")
    if valid_frac + test_frac >= 1:
        raise ValueError(
            f"valid_frac + test_frac = {valid_frac + test_frac} leaves no training data."
        )


def make_split(records, protocol="patient", valid_frac=0.1, test_frac=0.1, seed=42):
    if protocol == "patient":
        return patient_disjoint_split(records, valid_frac, test_frac, seed)
    if protocol == "slice":
        return slice_level_split(records, valid_frac, test_frac, seed)
    raise ValueError(f"Unknown protocol {protocol!r}; expected 'patient' or 'slice'.")


def shared_patients(splits):
    """Patients appearing in more than one split -- 0 under the patient protocol."""
    seen = {}
    for name in SPLIT_NAMES:
        for r in splits[name]:
            seen.setdefault(r["pid"], set()).add(name)
    return {pid: names for pid, names in seen.items() if len(names) > 1}


def describe(splits):
    """Human-readable summary, including the leakage count."""
    lines = []
    total = sum(len(splits[n]) for n in SPLIT_NAMES)

    header = f"{'split':<7}{'slices':>8}{'%':>7}{'patients':>10}   tumor types"
    lines.append(header)
    lines.append("-" * len(header))

    for name in SPLIT_NAMES:
        rows = splits[name]
        patients = {r["pid"] for r in rows}
        by_type = {}
        for r in rows:
            key = r.get("label_name") or str(r["label"])
            by_type[key] = by_type.get(key, 0) + 1
        types = ", ".join(f"{k} {v}" for k, v in sorted(by_type.items()))
        pct = 100.0 * len(rows) / total if total else 0.0
        lines.append(f"{name:<7}{len(rows):>8}{pct:>6.1f}%{len(patients):>10}   {types}")

    overlap = shared_patients(splits)
    lines.append("")
    if overlap:
        example = list(overlap.items())[:3]
        detail = "; ".join(f"{pid} in {'+'.join(sorted(names))}" for pid, names in example)
        lines.append(f"PATIENT LEAKAGE: {len(overlap)} patient(s) span multiple splits ({detail})")
    else:
        lines.append("No patient leakage: every patient belongs to exactly one split.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="path to manifest.csv")
    parser.add_argument("--protocol", choices=("patient", "slice"), default="patient")
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    splits = make_split(records, args.protocol, args.valid_frac, args.test_frac, args.seed)
    print(f"protocol: {args.protocol}  (seed {args.seed})\n")
    print(describe(splits))


if __name__ == "__main__":
    main()
