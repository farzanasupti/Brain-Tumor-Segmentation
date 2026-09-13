"""Tests for folds.py. Run directly: python test_folds.py

Stdlib only. These folds are load-bearing for all 45 runs of the factorial, so
they are worth testing harder than the code that uses them.
"""

import os
import random
import tempfile

import folds as F
import splits as split_lib

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def synthetic_records(n_patients=(82, 89, 62), n_slices=(708, 1426, 930), seed=7):
    """A manifest shaped like the real dataset: 3064 slices, 233 patients,
    one tumor type per patient, uneven slices per patient."""
    rng = random.Random(seed)
    records = []
    sid = pidn = 1
    for label, npat, total in zip((1, 2, 3), n_patients, n_slices):
        counts = [1] * npat
        for _ in range(total - npat):
            counts[rng.randrange(npat)] += 1
        for c in counts:
            pid = f"P{pidn:04d}"
            pidn += 1
            for _ in range(c):
                records.append({
                    "slice_id": str(sid), "pid": pid, "label": label,
                    "label_name": LABEL_NAMES[label],
                    "image": f"images/{sid}.png", "mask": f"masks/{sid}.png",
                })
                sid += 1
    return records


# ------------------------------------------------------------- assignment

@test
def every_patient_gets_exactly_one_fold():
    records = synthetic_records()
    a = F.make_folds(records)
    pids = {r["pid"] for r in records}
    assert set(a) == pids, "patient set mismatch"
    assert all(isinstance(v, int) for v in a.values())


@test
def folds_are_balanced_in_slice_count():
    records = synthetic_records()
    a = F.make_folds(records, n_folds=5)
    sizes = [0] * 5
    for r in records:
        sizes[a[r["pid"]]] += 1
    spread = (max(sizes) - min(sizes)) / (len(records) / 5)
    assert spread < 0.06, f"fold sizes vary by {spread:.1%}: {sizes}"


@test
def every_fold_carries_every_tumor_type():
    records = synthetic_records()
    a = F.make_folds(records, n_folds=5)
    for f in range(5):
        types = {r["label"] for r in records if a[r["pid"]] == f}
        assert types == {1, 2, 3}, f"fold {f} has only {types}"


@test
def tumor_type_proportions_are_preserved():
    records = synthetic_records()
    a = F.make_folds(records, n_folds=5)
    overall = {}
    for r in records:
        overall[r["label"]] = overall.get(r["label"], 0) + 1

    for f in range(5):
        rows = [r for r in records if a[r["pid"]] == f]
        for label, total in overall.items():
            share = sum(1 for r in rows if r["label"] == label) / len(rows)
            assert abs(share - total / len(records)) < 0.04, \
                f"fold {f} label {label}: {share:.3f} vs {total/len(records):.3f}"


@test
def assignment_is_deterministic_and_seed_sensitive():
    records = synthetic_records()
    assert F.make_folds(records, seed=42) == F.make_folds(records, seed=42)
    assert F.make_folds(records, seed=42) != F.make_folds(records, seed=1)


@test
def other_fold_counts_work():
    records = synthetic_records()
    for k in (2, 3, 5, 10):
        a = F.make_folds(records, n_folds=k)
        assert F.n_folds_in(a) == k, k
        sizes = [0] * k
        for r in records:
            sizes[a[r["pid"]]] += 1
        assert all(s > 0 for s in sizes), (k, sizes)


@test
def degenerate_inputs_are_rejected():
    records = synthetic_records()
    for bad in (lambda: F.make_folds(records, n_folds=1),
                lambda: F.make_folds(records[:3], n_folds=5)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("a degenerate configuration was accepted")

    mixed = [dict(r) for r in records[:60]]
    mixed[0]["pid"] = mixed[1]["pid"]
    mixed[0]["label"] = 3
    try:
        F.make_folds(mixed, n_folds=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a patient with two tumor types was accepted")


# ----------------------------------------------------------------- splits

@test
def fold_split_is_patient_disjoint():
    records = synthetic_records()
    a = F.make_folds(records)
    for mode in ("carve", "fold"):
        for f in range(5):
            parts = F.fold_split(records, a, f, valid_mode=mode)
            assert not split_lib.shared_patients(parts), f"{mode} fold {f} leaked"


@test
def fold_split_uses_every_slice_once():
    records = synthetic_records()
    a = F.make_folds(records)
    for mode in ("carve", "fold"):
        for f in range(5):
            parts = F.fold_split(records, a, f, valid_mode=mode)
            ids = [r["slice_id"] for k in parts for r in parts[k]]
            assert len(ids) == len(set(ids)) == len(records), f"{mode} {f}: {len(ids)}"


@test
def fold_mode_gives_each_patient_each_role_once():
    records = synthetic_records()
    a = F.make_folds(records)
    roles = {pid: {"test": 0, "valid": 0} for pid in a}
    for f in range(5):
        parts = F.fold_split(records, a, f, valid_mode="fold")
        for role in ("test", "valid"):
            for pid in {r["pid"] for r in parts[role]}:
                roles[pid][role] += 1
    assert all(v == {"test": 1, "valid": 1} for v in roles.values()), \
        "a patient did not serve each role exactly once"


@test
def carve_mode_keeps_more_data_for_training():
    """The default must not spend a whole fold on validation."""
    records = synthetic_records()
    a = F.make_folds(records)
    for f in range(5):
        carve = F.fold_split(records, a, f)
        whole = F.fold_split(records, a, f, valid_mode="fold")
        c = len(carve["train"]) / len(records)
        w = len(whole["train"]) / len(records)
        assert 0.66 < c < 0.74, f"fold {f} carve train share {c:.3f}"
        assert 0.58 < w < 0.62, f"fold {f} fold-mode train share {w:.3f}"
        assert c > w + 0.07, f"fold {f}: carve {c:.3f} vs fold {w:.3f}"


@test
def carve_mode_is_deterministic_per_fold():
    records = synthetic_records()
    a = F.make_folds(records)
    for f in range(5):
        x = F.fold_split(records, a, f)
        y = F.fold_split(records, a, f)
        assert {r["slice_id"] for r in x["valid"]} == {r["slice_id"] for r in y["valid"]}
    # different folds must carve different validation patients
    v = [frozenset(r["pid"] for r in F.fold_split(records, a, f)["valid"])
         for f in range(5)]
    assert len(set(v)) == 5, "folds carved identical validation sets"


@test
def carve_mode_preserves_the_tumor_type_mix():
    records = synthetic_records()
    a = F.make_folds(records)
    overall = {}
    for r in records:
        overall[r["label"]] = overall.get(r["label"], 0) + 1
    for f in range(5):
        valid = F.fold_split(records, a, f)["valid"]
        for label, total in overall.items():
            share = sum(1 for r in valid if r["label"] == label) / len(valid)
            assert abs(share - total / len(records)) < 0.06, \
                f"fold {f} label {label}: {share:.3f}"


@test
def bad_validation_settings_are_rejected():
    records = synthetic_records()
    a = F.make_folds(records)
    for bad in (lambda: F.fold_split(records, a, 0, valid_mode="holdout"),
                lambda: F.fold_split(records, a, 0, valid_frac=0.0),
                lambda: F.fold_split(records, a, 0, valid_frac=1.0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("an invalid validation setting was accepted")


@test
def validation_and_test_never_coincide():
    records = synthetic_records()
    a = F.make_folds(records)
    for mode in ("carve", "fold"):
        for f in range(5):
            parts = F.fold_split(records, a, f, valid_mode=mode)
            v = {r["pid"] for r in parts["valid"]}
            t = {r["pid"] for r in parts["test"]}
            assert not (v & t), f"{mode} fold {f} shares valid and test patients"
            assert v and t, f"{mode} fold {f} has an empty valid or test set"


@test
def fold_split_rejects_a_stale_assignment():
    records = synthetic_records()
    a = F.make_folds(records)
    del a[records[0]["pid"]]
    try:
        F.fold_split(records, a, 0)
    except ValueError as exc:
        assert "missing from the fold assignment" in str(exc), exc
        return
    raise AssertionError("a stale assignment was accepted")


@test
def out_of_range_fold_is_rejected():
    records = synthetic_records()
    a = F.make_folds(records, n_folds=5)
    for bad in (-1, 5, 99):
        try:
            F.fold_split(records, a, bad)
        except ValueError:
            continue
        raise AssertionError(f"fold={bad} was accepted")


# ------------------------------------------------------------ persistence

@test
def folds_round_trip_through_csv():
    records = synthetic_records()
    a = F.make_folds(records)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", "folds.csv")
        F.save_folds(path, a)
        assert F.load_folds(path) == a, "round trip changed the assignment"


@test
def a_malformed_fold_file_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.csv")
        with open(path, "w") as fh:
            fh.write("patient,group\nP0001,0\n")
        try:
            F.load_folds(path)
        except ValueError as exc:
            assert "missing column" in str(exc), exc
            return
    raise AssertionError("a malformed fold file was accepted")


@test
def describe_folds_reports_what_it_should():
    records = synthetic_records()
    a = F.make_folds(records)
    text = F.describe_folds(records, a)
    assert "meningioma" in text and "glioma" in text and "pituitary" in text
    assert "233 patients" in text, text.splitlines()[-1]
    assert "WARNING" not in text, "unexpected warning:\n" + text


if __name__ == "__main__":
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"  pass  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    raise SystemExit(1 if failed else 0)
