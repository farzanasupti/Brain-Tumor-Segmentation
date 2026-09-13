"""Convert the original Cheng et al. brain tumor dataset into image/mask PNGs
that keep the patient ID and tumor type.

The Kaggle PNG mirror of this dataset discards both, which makes patient-disjoint
evaluation impossible: 3064 slices come from only 233 patients, so a random
slice-level split puts near-duplicate slices of the same tumor in both train and
test. This script recovers that metadata from the original .mat files.

Source: https://doi.org/10.6084/m9.figshare.1512427.v5  (CC BY 4.0)
Download the four .zip parts; point --src at the directory holding them (they
are read directly, no need to unzip) or at a directory of extracted .mat files.

Each .mat holds a `cjdata` struct with:
    label        1 = meningioma, 2 = glioma, 3 = pituitary
    PID          patient ID (character codes)
    image        512x512 uint16 T1-CE slice
    tumorMask    512x512 binary tumor mask
    tumorBorder  tumor outline coordinates (unused; tumorMask is authoritative)

The files are MATLAB v7.3, which is HDF5 underneath -- scipy.io.loadmat cannot
read them, so h5py is required. h5py also returns MATLAB arrays transposed,
hence the .T below.

Usage:
    python prepare_dataset.py --src /path/to/figshare --out data/
    python prepare_dataset.py --src /path/to/figshare --out data/ --manifest-only
"""

import argparse
import csv
import io
import os
import zipfile

import cv2
import h5py
import numpy as np

LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}

#: Shipped alongside the slices but not slices: the authors' own 5-fold
#: cross-validation indices. Skipped quietly rather than reported as damage.
NON_SLICE_MAT = {"cvind"}


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_cjdata(fileobj_or_path):
    """Pull image, mask, patient ID and tumor type out of one .mat file."""
    with h5py.File(fileobj_or_path, "r") as f:
        cjdata = f["cjdata"]

        label = int(np.array(cjdata["label"]).flatten()[0])

        # PID is stored as an array of character codes, not a string.
        pid = "".join(chr(int(c)) for c in np.array(cjdata["PID"]).flatten()).strip()

        image = np.array(cjdata["image"]).T
        mask = np.array(cjdata["tumorMask"]).T

    return image, mask, pid, label


def to_uint8(image):
    """Scale a 16-bit slice to 8-bit for PNG storage.

    Per-slice min-max, because window/level varies between scans and a plain
    cast to uint8 would clip most of the dynamic range. This is a preprocessing
    decision: record it when reporting results.
    """
    image = image.astype(np.float64)
    lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.round((image - lo) / (hi - lo) * 255.0).astype(np.uint8)


def iter_mat_files(src):
    """Yield (slice_id, opener) for every .mat under src, zipped or loose.

    slice_id is the .mat basename ("1" for 1.mat), which is what the Kaggle
    mirror used to name its PNGs -- so the manifest this script writes can be
    joined onto an existing Kaggle-derived dataset by filename.
    """
    zips = sorted(
        os.path.join(src, n) for n in os.listdir(src) if n.lower().endswith(".zip")
    )
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".mat"):
                    continue
                slice_id = os.path.splitext(os.path.basename(name))[0]
                if slice_id.lower() in NON_SLICE_MAT:
                    continue
                # h5py needs a seekable file object, so read the member fully.
                payload = zf.read(name)
                yield slice_id, io.BytesIO(payload)

    loose = sorted(n for n in os.listdir(src) if n.lower().endswith(".mat"))
    for name in loose:
        slice_id = os.path.splitext(name)[0]
        if slice_id.lower() in NON_SLICE_MAT:
            continue
        yield slice_id, os.path.join(src, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True,
                        help="directory holding the figshare .zip parts or extracted .mat files")
    parser.add_argument("--out", required=True,
                        help="output directory; images/ and masks/ are created inside it")
    parser.add_argument("--manifest-only", action="store_true",
                        help="write manifest.csv but no PNGs (to attach metadata to an existing copy)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N slices (smoke test)")
    args = parser.parse_args()

    create_dir(args.out)
    if not args.manifest_only:
        create_dir(os.path.join(args.out, "images"))
        create_dir(os.path.join(args.out, "masks"))

    rows = []
    skipped = []

    for n, (slice_id, opener) in enumerate(iter_mat_files(args.src)):
        if args.limit is not None and n >= args.limit:
            break

        try:
            image, mask, pid, label = read_cjdata(opener)
        except Exception as exc:                      # keep going; report at the end
            skipped.append((slice_id, repr(exc)))
            continue

        if label not in LABEL_NAMES:
            skipped.append((slice_id, f"unexpected label {label!r}"))
            continue

        image8 = to_uint8(image)
        mask8 = (np.asarray(mask) > 0).astype(np.uint8) * 255

        if mask8.shape != image8.shape:
            skipped.append((slice_id, f"shape mismatch {image8.shape} vs {mask8.shape}"))
            continue

        image_rel = os.path.join("images", f"{slice_id}.png")
        mask_rel = os.path.join("masks", f"{slice_id}.png")

        if not args.manifest_only:
            cv2.imwrite(os.path.join(args.out, image_rel), image8)
            cv2.imwrite(os.path.join(args.out, mask_rel), mask8)

        rows.append({
            "slice_id": slice_id,
            "pid": pid,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "image": image_rel,
            "mask": mask_rel,
            "height": image8.shape[0],
            "width": image8.shape[1],
            "tumor_pixels": int((mask8 > 0).sum()),
        })

        if len(rows) % 250 == 0:
            print(f"  {len(rows)} slices...")

    if not rows:
        raise SystemExit(
            f"No readable .mat files found under {args.src!r}. Expected the four "
            f"figshare .zip parts, or a directory of extracted .mat files."
        )

    rows.sort(key=lambda r: (r["pid"], r["slice_id"]))

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    patients = {r["pid"] for r in rows}
    by_type = {}
    for r in rows:
        by_type[r["label_name"]] = by_type.get(r["label_name"], 0) + 1
    empty = sum(1 for r in rows if r["tumor_pixels"] == 0)

    print(f"\nWrote {manifest_path}")
    print(f"  slices  : {len(rows)}")
    print(f"  patients: {len(patients)}")
    print(f"  types   : " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"  slices/patient: min={min(sum(1 for r in rows if r['pid'] == p) for p in patients)} "
          f"max={max(sum(1 for r in rows if r['pid'] == p) for p in patients)} "
          f"mean={len(rows) / len(patients):.1f}")
    if empty:
        print(f"  warning: {empty} slice(s) have an empty tumor mask")
    if skipped:
        print(f"  skipped : {len(skipped)}")
        for slice_id, why in skipped[:10]:
            print(f"    {slice_id}: {why}")

    expected_slices, expected_patients = 3064, 233
    if len(rows) != expected_slices or len(patients) != expected_patients:
        print(f"\n  note: expected {expected_slices} slices from {expected_patients} "
              f"patients; got {len(rows)} from {len(patients)}. Check that all four "
              f"figshare parts are present.")


if __name__ == "__main__":
    main()
