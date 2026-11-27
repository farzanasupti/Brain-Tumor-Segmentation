"""Score trained cells on an external test set, without retraining.

Every cell directory under the given run directories that holds a best.pt is
rebuilt from its name (backbone__condition__arm__fN), loaded, and evaluated on
every slice of --manifest with the test-time pipeline (no augmentation). Oracle
cells receive the external set's true type, as at test time on Cheng.

    python external_eval.py --manifest data/brisc/manifest.csv \\
        --runs runs/main runs/region-bounded --out runs/brisc-external

Writes <out>/<cell>/test_slices.csv (same columns as a cell's own
test_slices.csv) and <out>/results.csv with one row per cell. Cells already in
results.csv are skipped.
"""

import argparse
import csv
import os
import sys

import torch

import backbones
import segmetrics
import splits as split_lib
from dataloading import make_loader
from engine import evaluate, load_checkpoint, pick_device
from experiment import completed_cells, write_per_slice

FIELDS = ["cell", "backbone", "condition", "arm", "fold", "source_run", "n",
          "test_dice", "test_iou", "test_hd95", "test_type_acc"]


def append_result(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def parse_cell(name):
    backbone, condition, arm, fold = name.split("__")
    return {"cell": name, "backbone": backbone,
            "condition": "n/a" if condition == "n-a" else condition,
            "arm": arm, "fold": int(fold.lstrip("f"))}


def find_cells(run_dirs):
    cells = []
    for run in run_dirs:
        for name in sorted(os.listdir(run)):
            if name.count("__") == 3 and os.path.exists(os.path.join(run, name, "best.pt")):
                cells.append((run, parse_cell(name)))
    return cells


def score_cell(run, cell, records, out, image_size, batch_size, device):
    kwargs = {"img_size": image_size}
    if cell["condition"] != "n/a":
        kwargs["condition"] = cell["condition"]
    model = backbones.build(cell["backbone"], in_channels=3, out_channels=1, **kwargs)
    load_checkpoint(os.path.join(run, cell["cell"], "best.pt"), model, device=device)
    model.to(device)
    loader = make_loader(records, arm=cell["arm"], training=False, batch_size=batch_size,
                         image_size=image_size)
    metrics, rows = evaluate(model, loader, None, device, per_item=True, boundary=True)
    cell_out = os.path.join(out, cell["cell"])
    os.makedirs(cell_out, exist_ok=True)
    write_per_slice(os.path.join(cell_out, "test_slices.csv"), rows)
    hd = segmetrics.summarise([r["hd95"] for r in rows], "hd95")
    return {**cell, "source_run": run, "n": metrics["n"],
            "test_dice": round(metrics["dice"], 6), "test_iou": round(metrics["iou"], 6),
            "test_hd95": round(hd["hd95"], 4),
            "test_type_acc": (round(metrics["type_acc"], 6)
                              if metrics["type_acc"] is not None else "")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    root = os.path.dirname(os.path.abspath(args.manifest))
    records = split_lib.resolve_paths(split_lib.load_manifest(args.manifest), root)
    os.makedirs(args.out, exist_ok=True)
    results = os.path.join(args.out, "results.csv")
    done = completed_cells(results)
    device = pick_device(args.device)
    for run, cell in find_cells(args.runs):
        if cell["cell"] in done:
            continue
        with torch.no_grad():
            row = score_cell(run, cell, records, args.out, args.image_size,
                             args.batch_size, device)
        append_result(results, row)
        print(f"{cell['cell']}: dice {row['test_dice']:.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
