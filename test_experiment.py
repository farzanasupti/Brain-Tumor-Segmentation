"""Tests for experiment.py. Run directly: python test_experiment.py

The grid runner's job is to be interruptible without losing work, so most of
what matters here is resume behaviour and bookkeeping rather than training.
"""

import csv
import os
import shutil
import tempfile

import cv2
import numpy as np

import backbones
import experiment
import folds as fold_lib
from experiment import (append_result, cell_name, completed_cells,
                        enumerate_cells, preflight, run_grid)

RESULTS = []
TMP = None


def test(fn):
    RESULTS.append(fn)
    return fn


def build_corpus(root, n_patients=12, slices_per_patient=2, size=32):
    """A tiny manifest on disk, with real PNGs and patient-disjoint folds."""
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]

    records, sid = [], 0
    for p in range(n_patients):
        label = (p % 3) + 1
        cy = size * (0.3 + 0.2 * (label - 1))
        for _ in range(slices_per_patient):
            disc = ((yy - cy) ** 2 + (xx - size * 0.5) ** 2) < (size * 0.2) ** 2
            image = np.full((size, size), 40, dtype=np.uint8)
            image[disc] = 200
            ip = os.path.join(root, "images", f"{sid}.png")
            mp = os.path.join(root, "masks", f"{sid}.png")
            cv2.imwrite(ip, image)
            cv2.imwrite(mp, (disc * 255).astype(np.uint8))
            records.append({"slice_id": str(sid), "pid": f"P{p:03d}",
                            "label": label, "label_name": str(label),
                            "image": ip, "mask": mp})
            sid += 1

    assignment = fold_lib.make_folds(records, n_folds=3, seed=1)
    return records, assignment


def tiny_kwargs(**over):
    """Keep cells to seconds: small images, one batch, one epoch."""
    base = dict(image_size=32, batch_size=4, epochs=1, patience=1,
                num_workers=0, max_batches=1, verbose=False,
                # A full-size U-Net writes ~340 MB of checkpoint per cell.
                model_kwargs=dict(widths=(4, 8), bottleneck=16))
    base.update(over)
    return base


# ---------------------------------------------------------------- the grid

@test
def the_grid_is_the_full_product():
    cells = enumerate_cells(["unet"], ["predicted"], ["standard", "region"], [0, 1, 2])
    assert len(cells) == 6, len(cells)
    assert len({c["cell"] for c in cells}) == 6, "cell names collide"


@test
def conditioning_only_multiplies_backbones_that_have_it():
    """Running plain U-Net once per condition would triple its cost for nothing."""
    cells = enumerate_cells(["unet", "cond_unet"], ["predicted", "oracle", "none"],
                            ["standard"], [0])
    plain = [c for c in cells if c["backbone"] == "unet"]
    cond = [c for c in cells if c["backbone"] == "cond_unet"]
    assert len(plain) == 1, plain
    assert len(cond) == 3, cond
    assert plain[0]["condition"] == "n/a"


@test
def cells_carry_their_family_for_the_analysis():
    cells = enumerate_cells(["unet", "cond_unet"], ["predicted"], ["standard"], [0])
    for c in cells:
        assert c["family"] == backbones.family_of(c["backbone"]), c


@test
def cell_names_distinguish_every_axis():
    a = cell_name("cond_unet", "predicted", "region", 0)
    for other in (cell_name("cond_unet", "oracle", "region", 0),
                  cell_name("cond_unet", "predicted", "global", 0),
                  cell_name("cond_unet", "predicted", "region", 1),
                  cell_name("unet", "predicted", "region", 0)):
        assert a != other, (a, other)


# ------------------------------------------------------------- bookkeeping

@test
def completed_cells_reads_back_what_was_written():
    path = os.path.join(TMP, "b1", "results.csv")
    assert completed_cells(path) == set(), "a missing file should read as empty"
    append_result(path, {"cell": "unet__n/a__standard__f0", "test_dice": 0.5})
    append_result(path, {"cell": "unet__n/a__standard__f1", "test_dice": 0.6})
    assert completed_cells(path) == {"unet__n/a__standard__f0",
                                     "unet__n/a__standard__f1"}


@test
def the_results_file_keeps_one_header_and_a_stable_column_order():
    path = os.path.join(TMP, "b2", "results.csv")
    for fold in range(3):
        append_result(path, {"cell": f"c{fold}", "test_dice": 0.1 * fold})
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(experiment.RESULT_FIELDS), rows[0]
    assert len(rows) == 4, f"expected header + 3 rows, got {len(rows)}"


# --------------------------------------------------------------- preflight

@test
def preflight_reports_the_grid_and_the_splits():
    root = os.path.join(TMP, "p1")
    records, assignment = build_corpus(root)
    cells = enumerate_cells(["unet"], ["predicted"], ["standard"], [0, 1, 2])
    text = preflight(cells, records, assignment)
    assert "3 cells" in text, text
    assert "fold 0:" in text and "fold 2:" in text, text
    assert "LEAK" not in text, "patient leakage reported in a patient-disjoint split"


@test
def preflight_names_a_blocked_backbone_instead_of_failing():
    """The point of a pre-flight is to find this before hour sixty, not during."""
    root = os.path.join(TMP, "p2")
    records, assignment = build_corpus(root)
    cells = enumerate_cells(["unet", "swin_umamba"], ["predicted"], ["standard"], [0])
    text = preflight(cells, records, assignment)
    ok, _ = backbones.available("swin_umamba")["swin_umamba"]
    if ok:
        return
    assert "BLOCKED" in text and "swin_umamba" in text, text
    assert "Cannot run" in text, text


# ------------------------------------------------------------------- runs

@test
def a_grid_run_produces_one_row_per_cell():
    root = os.path.join(TMP, "r1")
    records, assignment = build_corpus(root)
    cells = enumerate_cells(["unet"], ["predicted"], ["standard"], [0, 1])
    rows = run_grid(cells, records, assignment, os.path.join(root, "runs"),
                    **tiny_kwargs())
    assert len(rows) == 2, len(rows)
    for row in rows:
        assert 0.0 <= row["test_dice"] <= 1.0, row
        assert row["params"] > 0 and row["seconds"] >= 0


@test
def finished_cells_are_not_rerun():
    root = os.path.join(TMP, "r2")
    records, assignment = build_corpus(root)
    out = os.path.join(root, "runs")
    cells = enumerate_cells(["unet"], ["predicted"], ["standard"], [0, 1])

    first = run_grid(cells, records, assignment, out, **tiny_kwargs())
    again = run_grid(cells, records, assignment, out, **tiny_kwargs())
    assert len(first) == 2 and len(again) == 0, (len(first), len(again))

    with open(os.path.join(out, "results.csv"), newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2, "results.csv gained duplicates"


@test
def limit_stops_after_n_pending_cells():
    """Colab sessions end; --limit lets a run stop cleanly and resume later."""
    root = os.path.join(TMP, "r3")
    records, assignment = build_corpus(root)
    out = os.path.join(root, "runs")
    cells = enumerate_cells(["unet"], ["predicted"], ["standard"], [0, 1, 2])

    assert len(run_grid(cells, records, assignment, out, limit=2, **tiny_kwargs())) == 2
    assert len(run_grid(cells, records, assignment, out, limit=2, **tiny_kwargs())) == 1
    assert len(run_grid(cells, records, assignment, out, limit=2, **tiny_kwargs())) == 0


@test
def a_cell_leaves_the_artefacts_the_analysis_needs():
    root = os.path.join(TMP, "r4")
    records, assignment = build_corpus(root)
    out = os.path.join(root, "runs")
    cells = enumerate_cells(["cond_unet"], ["predicted"], ["region"], [0])
    run_grid(cells, records, assignment, out, **tiny_kwargs())

    cell_dir = os.path.join(out, cells[0]["cell"])
    for name in ("best.pt", "last.pt", "log.csv", "test_slices.csv"):
        assert os.path.exists(os.path.join(cell_dir, name)), f"missing {name}"

    with open(os.path.join(cell_dir, "test_slices.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "no per-slice rows"
    assert {"pid", "type", "dice", "iou", "hd95", "assd",
            "pred_pixels", "true_pixels"} <= set(rows[0]), rows[0]
    assert all(r["pid"] for r in rows), "per-slice rows lost the patient id"


@test
def predicted_masks_are_kept_so_new_metrics_need_no_retraining():
    """Adding a metric after 45 cells have run must not mean running them again."""
    root = os.path.join(TMP, "r7")
    records, assignment = build_corpus(root)
    out = os.path.join(root, "runs")
    cells = enumerate_cells(["unet"], ["predicted"], ["standard"], [0])
    run_grid(cells, records, assignment, out, **tiny_kwargs())

    pred_dir = os.path.join(out, cells[0]["cell"], "predictions")
    assert os.path.isdir(pred_dir), "predictions were not saved"
    saved = [f for f in os.listdir(pred_dir) if f.endswith(".png")]

    with open(os.path.join(out, cells[0]["cell"], "test_slices.csv"), newline="") as fh:
        scored = list(csv.DictReader(fh))
    assert len(saved) == len(scored), f"{len(saved)} masks for {len(scored)} rows"

    import cv2
    import numpy as np
    mask = cv2.imread(os.path.join(pred_dir, saved[0]), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(mask)) <= {0, 255}, "saved prediction is not binary"


@test
def a_conditioned_cell_records_type_accuracy_and_a_plain_one_does_not():
    root = os.path.join(TMP, "r5")
    records, assignment = build_corpus(root)
    out = os.path.join(root, "runs")
    cells = enumerate_cells(["unet", "cond_unet"], ["predicted"], ["standard"], [0])
    rows = {r["backbone"]: r for r in
            run_grid(cells, records, assignment, out, **tiny_kwargs())}
    assert rows["cond_unet"]["test_type_acc"] is not None
    assert rows["unet"]["test_type_acc"] is None


@test
def every_fold_is_tested_on_different_patients():
    """If the runner mixed folds up, three cells would score the same patients."""
    root = os.path.join(TMP, "r6")
    records, assignment = build_corpus(root)
    out = os.path.join(root, "runs")
    cells = enumerate_cells(["unet"], ["predicted"], ["standard"], [0, 1, 2])
    run_grid(cells, records, assignment, out, **tiny_kwargs())

    seen = []
    for cell in cells:
        path = os.path.join(out, cell["cell"], "test_slices.csv")
        with open(path, newline="") as fh:
            seen.append({r["pid"] for r in csv.DictReader(fh)})
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not (seen[i] & seen[j]), f"folds {i} and {j} share test patients"


if __name__ == "__main__":
    TMP = tempfile.mkdtemp(prefix="bts-experiment-tests-")
    failed = 0
    try:
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
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    raise SystemExit(1 if failed else 0)
