"""Tests for prepare_external.py and external_eval.parse_cell.
Run directly: python test_prepare_external.py

Builds tiny synthetic copies of the kaggle_3m and brisc2025 layouts.
"""

import csv
import os
import tempfile

import numpy as np
from PIL import Image

import prepare_external as PE

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def blob(seed, size=64):
    """A random smooth image, distinct per seed."""
    rng = np.random.default_rng(seed)
    a = rng.random((8, 8))
    return (np.kron(a, np.ones((size // 8, size // 8))) * 255).astype(np.uint8)


def square_mask(size=64, on=True):
    m = np.zeros((size, size), np.uint8)
    if on:
        m[20:40, 20:40] = 255
    return m


def make_lgg(root):
    with open(os.path.join(root, "data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Patient", "histological_type"])
        w.writerow(["TCGA_AA_0001", "3"])
        w.writerow(["TCGA_AA_0002", ""])  # no type: dropped
    for case, pid in (("TCGA_AA_0001_19990101", 1), ("TCGA_AA_0002_19990101", 2)):
        os.makedirs(os.path.join(root, case))
        for k, has_tumour in ((1, False), (2, True), (10, True)):
            rgb = np.stack([blob(pid * 100 + k), blob(pid * 100 + k + 50), blob(7)], -1)
            Image.fromarray(rgb).save(os.path.join(root, case, f"{case}_{k}.tif"))
            Image.fromarray(square_mask(on=has_tumour)).save(
                os.path.join(root, case, f"{case}_{k}_mask.tif"))


@test
def lgg_keeps_typed_tumour_slices_with_flair():
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "src"), os.path.join(d, "out")
        os.makedirs(src)
        make_lgg(src)
        rows = PE.prepare_lgg(src, out)
        assert len(rows) == 2, rows  # 2 tumour slices of the typed patient
        assert {r["pid"] for r in rows} == {"TCGA_AA_0001"}
        assert all(r["label"] == 3 and r["label_name"] == "oligodendroglioma" for r in rows)
        # slices sorted numerically (2 before 10), FLAIR = channel 1
        img = np.asarray(Image.open(os.path.join(out, rows[0]["image"])))
        assert img.ndim == 2 and np.array_equal(img, blob(100 + 2 + 50))
        with open(os.path.join(out, "manifest.csv")) as fh:
            assert len(list(csv.DictReader(fh))) == 2


def make_cheng(root):
    os.makedirs(os.path.join(root, "images"))
    with open(os.path.join(root, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slice_id", "pid", "label", "image", "mask"])
        for i in (1, 2):
            Image.fromarray(blob(1000 + i)).save(os.path.join(root, "images", f"{i}.png"))
            w.writerow([i, "p", 1, f"images/{i}.png", f"masks/{i}.png"])


def make_brisc(root):
    for split in ("train", "test"):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(root, "segmentation_task", split, sub))
    files = [("train", "00001", "me", blob(1001)),   # copy of Cheng slice 1
             ("train", "00002", "gl", blob(2002)),   # new
             ("test", "00001", "pi", blob(3003))]    # new
    for split, idx, code, img in files:
        name = f"brisc2025_{split}_{idx}_{code}_ax_t1"
        Image.fromarray(img).convert("RGB").save(
            os.path.join(root, "segmentation_task", split, "images", name + ".jpg"), quality=95)
        Image.fromarray(square_mask()).save(
            os.path.join(root, "segmentation_task", split, "masks", name + ".png"))


@test
def brisc_drops_cheng_copies_and_keeps_types():
    with tempfile.TemporaryDirectory() as d:
        cheng, src, out = (os.path.join(d, x) for x in ("cheng", "src", "out"))
        make_cheng(cheng)
        make_brisc(src)
        rows = PE.prepare_brisc(src, os.path.join(cheng, "manifest.csv"), out, 0.90)
        assert [r["label_name"] for r in rows] == ["glioma", "pituitary"], rows
        assert [r["label"] for r in rows] == [2, 3]
        assert rows[0]["pid"] == "train_00002" and rows[1]["pid"] == "test_00001"
        with open(os.path.join(out, "cheng_match.csv")) as fh:
            audit = list(csv.DictReader(fh))
        assert len(audit) == 3
        assert float(audit[0]["corr"]) > 0.99 and audit[0]["cheng_slice"] == "1"


@test
def thumb_is_unit_norm_and_shift_invariant():
    a = Image.fromarray(blob(5))
    b = Image.fromarray(np.clip(blob(5).astype(int) + 10, 0, 255).astype(np.uint8))
    ta, tb = PE.thumb(a), PE.thumb(b)
    assert abs(np.linalg.norm(ta) - 1) < 1e-6
    assert ta @ tb > 0.99


@test
def parse_cell_round_trips_names():
    try:
        import external_eval as EE
    except ImportError:  # needs torch
        print("  (skipped: torch not installed)")
        return
    c = EE.parse_cell("cond_unet__oracle__standard__f3")
    assert c == {"cell": "cond_unet__oracle__standard__f3", "backbone": "cond_unet",
                 "condition": "oracle", "arm": "standard", "fold": 3}
    assert EE.parse_cell("unet__n-a__region-bounded__f0")["condition"] == "n/a"


if __name__ == "__main__":
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    raise SystemExit(1 if failed else 0)
