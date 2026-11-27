"""Build manifests for the two external datasets, in the format the pipeline reads.

TCGA-LGG (Buda et al., Kaggle "lgg-mri-segmentation", kaggle_3m/)
    110 patients with lower-grade glioma; 3-channel TIFFs (pre-contrast, FLAIR,
    post-contrast) and FLAIR tumour masks, 256x256. data.csv gives
    `histological_type` per patient. Checked against the GDC primary diagnosis
    (2026-11-28): 1 = astrocytoma, 2 = oligoastrocytoma ("Mixed glioma"),
    3 = oligodendroglioma; one patient lacks a code and is dropped. These are the
    type labels for conditioning. Only slices with tumour are kept, as in Cheng,
    and the FLAIR channel is written as a grayscale PNG. Patient IDs are kept, so
    folds.py makes patient-disjoint folds as for Cheng.

BRISC 2025 (Fateh et al., Kaggle "briscdataset/brisc2025", segmentation_task/)
    T1 slices with masks and the type in the file name (gl/me/pi). It has no
    patient IDs and reuses Cheng images, so it is only an external test set.
    Each slice is matched to its nearest Cheng slice (Pearson correlation of
    32x32 z-scored thumbnails); slices at or above --max-corr are excluded.
    The match score is kept in the manifest so the threshold can be varied.

    python prepare_external.py lgg   --src data/ext/lgg/kaggle_3m --out data/lgg
    python prepare_external.py brisc --src data/ext/brisc/brisc2025 \\
        --cheng data/manifest.csv --out data/brisc

Labels follow the pipeline's 1..3 coding. For BRISC they match Cheng
(1 meningioma, 2 glioma, 3 pituitary); for LGG they are the histology codes.
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
from PIL import Image

LGG_TYPES = {1: "astrocytoma", 2: "oligoastrocytoma", 3: "oligodendroglioma"}
BRISC_TYPES = {"me": (1, "meningioma"), "gl": (2, "glioma"), "pi": (3, "pituitary")}
FIELDS = ["slice_id", "pid", "label", "label_name", "image", "mask",
          "height", "width", "tumor_pixels"]
FLAIR = 1  # channel order in kaggle_3m: pre-contrast, FLAIR, post-contrast


def thumb(img):
    """32x32 z-scored, unit-norm thumbnail; dot products are Pearson correlations."""
    a = np.asarray(img.convert("L").resize((32, 32), Image.BOX), dtype=np.float64).ravel()
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-9)


def write_pair(out, sid, image, mask):
    """Save image (uint8 2D) and mask (0/255) PNGs; return manifest paths."""
    img_rel, mask_rel = f"images/{sid}.png", f"masks/{sid}.png"
    Image.fromarray(image).save(os.path.join(out, img_rel))
    Image.fromarray(mask).save(os.path.join(out, mask_rel))
    return img_rel, mask_rel


def lgg_types(src):
    types = {}
    with open(os.path.join(src, "data.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            code = r["histological_type"].strip()
            if code:
                types[r["Patient"]] = int(float(code))
    return types


def slice_number(name):
    m = re.search(r"_(\d+)(?:_mask)?\.tif$", name)
    return int(m.group(1)) if m else None


def prepare_lgg(src, out):
    types = lgg_types(src)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    rows, dropped_pat, empty = [], 0, 0
    for case in sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))):
        patient = "_".join(case.split("_")[:3])  # TCGA_CS_4941_19960909 -> TCGA_CS_4941
        if patient not in types:
            dropped_pat += 1
            continue
        files = [f for f in os.listdir(os.path.join(src, case))
                 if f.endswith(".tif") and not f.endswith("_mask.tif")]
        for f in sorted(files, key=slice_number):
            mask = np.asarray(Image.open(os.path.join(src, case, f[:-4] + "_mask.tif")).convert("L"))
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)
            n = int((mask > 0).sum())
            if n == 0:
                empty += 1
                continue
            image = np.asarray(Image.open(os.path.join(src, case, f)).convert("RGB"))[:, :, FLAIR]
            sid = len(rows) + 1
            img_rel, mask_rel = write_pair(out, sid, np.ascontiguousarray(image), mask)
            label = types[patient]
            rows.append({"slice_id": sid, "pid": patient, "label": label,
                         "label_name": LGG_TYPES[label], "image": img_rel, "mask": mask_rel,
                         "height": mask.shape[0], "width": mask.shape[1], "tumor_pixels": n})
    write_manifest(out, rows)
    print(f"LGG: {len(rows)} tumour slices from {len({r['pid'] for r in rows})} patients; "
          f"{empty} empty slices skipped; {dropped_pat} case(s) without a type dropped")
    return rows


def cheng_thumbs(cheng_manifest):
    root = os.path.dirname(os.path.abspath(cheng_manifest))
    with open(cheng_manifest, newline="") as fh:
        recs = list(csv.DictReader(fh))
    mat = np.stack([thumb(Image.open(os.path.join(root, r["image"]))) for r in recs])
    return [r["slice_id"] for r in recs], mat


def prepare_brisc(src, cheng_manifest, out, max_corr):
    ids, cheng = cheng_thumbs(cheng_manifest)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    rows, excluded, audit = [], 0, []
    for split in ("train", "test"):
        idir = os.path.join(src, "segmentation_task", split, "images")
        mdir = os.path.join(src, "segmentation_task", split, "masks")
        for f in sorted(os.listdir(idir)):
            parts = f.split("_")  # brisc2025_train_00001_gl_ax_t1.jpg
            code, plane = parts[3], parts[4]
            img = Image.open(os.path.join(idir, f))
            scores = cheng @ thumb(img)
            j = int(scores.argmax())
            corr = float(scores[j])
            audit.append({"file": f, "split": split, "type": code, "plane": plane,
                          "cheng_slice": ids[j], "corr": f"{corr:.4f}"})
            if corr >= max_corr:
                excluded += 1
                continue
            mask = np.asarray(Image.open(os.path.join(mdir, f[:-4] + ".png")).convert("L"))
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)
            n = int((mask > 0).sum())
            if n == 0:
                continue
            label, name = BRISC_TYPES[code]
            sid = len(rows) + 1
            img_rel, mask_rel = write_pair(out, sid, np.asarray(img.convert("L")), mask)
            rows.append({"slice_id": sid, "pid": f"{split}_{parts[2]}", "label": label,
                         "label_name": name, "image": img_rel, "mask": mask_rel,
                         "height": mask.shape[0], "width": mask.shape[1], "tumor_pixels": n,
                         "plane": plane, "cheng_corr": f"{corr:.4f}"})
    write_manifest(out, rows, extra=["plane", "cheng_corr"])
    with open(os.path.join(out, "cheng_match.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    print(f"BRISC: {len(rows)} slices kept; {excluded} of {len(audit)} excluded "
          f"(nearest-Cheng correlation >= {max_corr})")
    return rows


def write_manifest(out, rows, extra=()):
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + list(extra))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="dataset", required=True)
    a = sub.add_parser("lgg")
    a.add_argument("--src", required=True, help="the kaggle_3m directory")
    a.add_argument("--out", required=True)
    b = sub.add_parser("brisc")
    b.add_argument("--src", required=True, help="the brisc2025 directory")
    b.add_argument("--cheng", required=True, help="Cheng manifest.csv (with its PNGs)")
    b.add_argument("--out", required=True)
    b.add_argument("--max-corr", type=float, default=0.90,
                   help="exclude slices this close to a Cheng slice (default 0.90)")
    args = p.parse_args(argv)
    if args.dataset == "lgg":
        prepare_lgg(args.src, args.out)
    else:
        prepare_brisc(args.src, args.cheng, args.out, args.max_corr)


if __name__ == "__main__":
    sys.exit(main())
