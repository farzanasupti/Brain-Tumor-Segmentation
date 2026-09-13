"""Grid runner over (backbone, condition, augmentation arm, fold).

Trains one cell at a time, appends a row per finished cell to results.csv, and
skips cells already in that file. A Colab disconnect at hour sixty costs the
cell in flight and nothing else -- rerun the same command and it picks up.

    python experiment.py --manifest data/manifest.csv --folds data/folds.csv \
        --backbones unet cond_unet --arms standard region --dry-run

    python experiment.py --manifest data/manifest.csv --folds data/folds.csv \
        --backbones unet cond_unet --arms standard region --limit 4

Per-cell artefacts land under runs/<run-name>/<cell>/: best.pt, last.pt, the
per-epoch log, per-slice test scores carrying pid and tumor type, and the
predicted masks themselves -- so any metric thought of after the fact can be
computed without retraining a single cell.

--dry-run is the pre-flight check: it enumerates the grid, reports which
backbones this environment can actually build, and estimates nothing it has not
measured.
"""

import argparse
import csv
import itertools
import os
import time

import backbones
import folds as fold_lib
import segmetrics
import splits as split_lib
from dataloading import make_loader
from engine import evaluate, fit, load_checkpoint, pick_device

RESULT_FIELDS = (
    "cell", "backbone", "family", "condition", "arm", "fold", "seed",
    "params", "best_valid_dice", "test_dice", "test_iou", "test_hd95",
    "test_hd95_undefined", "test_assd", "test_type_acc",
    "epochs_run", "stopped_early", "seconds", "finished_at",
)

CONDITIONABLE = ("cond_unet",)


def cell_name(backbone, condition, arm, fold):
    """Stable identifier, used as both the resume key and the directory name."""
    return f"{backbone}__{condition}__{arm}__f{fold}"


def enumerate_cells(backbone_names, conditions, arms, fold_ids):
    """Every cell in the grid. Conditioning only varies for backbones that have it."""
    cells = []
    for backbone in backbone_names:
        applicable = conditions if backbone in CONDITIONABLE else ("n/a",)
        for condition, arm, fold in itertools.product(applicable, arms, fold_ids):
            cells.append({
                "backbone": backbone,
                "family": backbones.family_of(backbone),
                "condition": condition,
                "arm": arm,
                "fold": fold,
                "cell": cell_name(backbone, condition, arm, fold),
            })
    return cells


def completed_cells(results_path):
    if not results_path or not os.path.exists(results_path):
        return set()
    with open(results_path, newline="") as fh:
        return {row["cell"] for row in csv.DictReader(fh) if row.get("cell")}


def append_result(results_path, row):
    directory = os.path.dirname(os.path.abspath(results_path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    new = not os.path.exists(results_path)
    with open(results_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULT_FIELDS})


def write_per_slice(path, rows):
    if not rows:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_cell(cell, records, assignment, out_dir, *, image_size=256, batch_size=16,
             epochs=150, lr=1e-4, patience=20, type_weight=0.2, seed=42,
             num_workers=2, valid_mode="carve", device=None, max_batches=None,
             model_kwargs=None, save_predictions=True, micro_batch=None,
             verbose=True):
    """Train and test one cell. Returns the results row.

    `model_kwargs` is forwarded to the backbone builder, for the protocol's
    reduced-budget variants (narrower widths, smaller feature_size) and for
    tests that cannot afford a 31M-parameter checkpoint per cell.
    """
    started = time.time()
    device = device or pick_device()
    cell_dir = os.path.join(out_dir, cell["cell"])
    os.makedirs(cell_dir, exist_ok=True)

    parts = fold_lib.fold_split(records, assignment, cell["fold"],
                                valid_mode=valid_mode, seed=seed)

    kwargs = {"img_size": image_size, **(model_kwargs or {})}
    if cell["condition"] != "n/a":
        kwargs["condition"] = cell["condition"]
    model = backbones.build(cell["backbone"], in_channels=3, out_channels=1, **kwargs)

    # A model that only fits a micro-batch of 4 still trains at the protocol's
    # effective batch via accumulation. Keep micro_batch equal across compared
    # cells: BatchNorm normalises over the micro-batch, not the effective batch.
    micro = micro_batch or batch_size
    if batch_size % micro:
        raise ValueError(
            f"batch_size {batch_size} is not a multiple of micro_batch {micro}.")
    accumulate = batch_size // micro

    common = dict(image_size=image_size, seed=seed, num_workers=num_workers)
    train_loader = make_loader(parts["train"], arm=cell["arm"], training=True,
                               batch_size=micro, **common)
    valid_loader = make_loader(parts["valid"], arm=cell["arm"], training=False,
                               batch_size=micro, **common)
    test_loader = make_loader(parts["test"], arm=cell["arm"], training=False,
                              batch_size=micro, **common)

    best_path = os.path.join(cell_dir, "best.pt")
    outcome = fit(
        model, train_loader, valid_loader,
        epochs=epochs, lr=lr, patience=patience, type_weight=type_weight,
        device=device, checkpoint_path=best_path,
        last_path=os.path.join(cell_dir, "last.pt"),
        log_path=os.path.join(cell_dir, "log.csv"),
        max_batches=max_batches, accumulate=accumulate, verbose=verbose,
    )

    # Test the *best* checkpoint, not whatever the last epoch left behind.
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, device=device)
    metrics, per_slice = evaluate(
        model, test_loader, None, device, max_batches=max_batches,
        per_item=True, boundary=True,
        save_predictions_to=(os.path.join(cell_dir, "predictions")
                             if save_predictions else None))
    write_per_slice(os.path.join(cell_dir, "test_slices.csv"), per_slice)

    boundary = segmetrics.summarise([r.get("hd95", float("nan")) for r in per_slice],
                                    "hd95")
    surface = segmetrics.summarise([r.get("assd", float("nan")) for r in per_slice],
                                   "assd")

    return {
        **cell,
        "seed": seed,
        "params": backbones.count_parameters(model),
        "best_valid_dice": round(outcome["best_dice"], 6),
        "test_dice": round(metrics["dice"], 6),
        "test_iou": round(metrics["iou"], 6),
        "test_hd95": round(boundary["hd95"], 4),
        "test_hd95_undefined": boundary["hd95_undefined"],
        "test_assd": round(surface["assd"], 4),
        "test_type_acc": (round(metrics["type_acc"], 6)
                          if metrics["type_acc"] is not None else None),
        "epochs_run": outcome["epochs_run"],
        "stopped_early": outcome["stopped_early"],
        "seconds": round(time.time() - started, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_grid(cells, records, assignment, out_dir, *, results_path=None,
             limit=None, verbose=True, **cell_kwargs):
    """Run every cell not already recorded. Returns the rows produced this call."""
    results_path = results_path or os.path.join(out_dir, "results.csv")
    done = completed_cells(results_path)

    pending = [c for c in cells if c["cell"] not in done]
    if verbose:
        print(f"{len(cells)} cells in the grid, {len(done)} already done, "
              f"{len(pending)} pending.")
    if limit is not None:
        pending = pending[:limit]

    produced = []
    for index, cell in enumerate(pending, 1):
        if verbose:
            print(f"\n[{index}/{len(pending)}] {cell['cell']}")
        row = run_cell(cell, records, assignment, out_dir, verbose=verbose,
                       **cell_kwargs)
        append_result(results_path, row)
        produced.append(row)
        if verbose:
            print(f"  test Dice {row['test_dice']:.4f}  "
                  f"IoU {row['test_iou']:.4f}  HD95 {row['test_hd95']:.2f}px  "
                  f"({row['seconds']:.0f}s)")
    return produced


def preflight(cells, records, assignment):
    """What --dry-run prints: the grid, and whether it can actually be run."""
    lines = []
    status = backbones.available()

    wanted = sorted({c["backbone"] for c in cells})
    blocked = [b for b in wanted if not status[b][0]]

    lines.append(f"{len(cells)} cells: "
                 f"{len(wanted)} backbone(s) x "
                 f"{len({c['arm'] for c in cells})} arm(s) x "
                 f"{len({c['fold'] for c in cells})} fold(s)"
                 + (f" x {len({c['condition'] for c in cells}) - 0} condition(s)"
                    if any(c["condition"] != "n/a" for c in cells) else ""))
    lines.append("")

    for backbone in wanted:
        ok, detail = status[backbone]
        mark = "ready" if ok else "BLOCKED"
        lines.append(f"  {backbone:<14}{backbones.family_of(backbone):<13}"
                     f"{mark:<9}{detail}")

    lines.append("")
    sizes = []
    for fold in sorted({c["fold"] for c in cells}):
        parts = fold_lib.fold_split(records, assignment, fold)
        leak = split_lib.shared_patients(parts)
        sizes.append(f"  fold {fold}: train {len(parts['train'])}, "
                     f"valid {len(parts['valid'])}, test {len(parts['test'])}"
                     + (f"  LEAK: {len(leak)} patients" if leak else ""))
    lines.extend(sizes)

    if blocked:
        lines.append("")
        lines.append(f"Cannot run: {', '.join(blocked)} unavailable in this "
                     f"environment. Cells using them would fail at build time.")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--folds", required=True, help="folds.csv from folds.py")
    parser.add_argument("--out", default="runs/main")
    parser.add_argument("--backbones", nargs="+", default=["unet"])
    parser.add_argument("--conditions", nargs="+", default=["predicted"],
                        help="only applies to conditionable backbones")
    parser.add_argument("--arms", nargs="+", default=["standard"])
    parser.add_argument("--folds-only", nargs="+", type=int, default=None,
                        help="restrict to these fold indices")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="effective batch size, reached by accumulation")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="what actually fits in VRAM; must divide --batch-size. "
                             "Hold it equal across cells you intend to compare.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--type-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None,
                        help="run at most this many pending cells, then stop")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    records = split_lib.load_manifest(args.manifest)
    records = split_lib.resolve_paths(
        records, os.path.dirname(os.path.abspath(args.manifest)))
    assignment = fold_lib.load_folds(args.folds)

    fold_ids = (args.folds_only if args.folds_only is not None
                else sorted(set(assignment.values())))
    cells = enumerate_cells(args.backbones, args.conditions, args.arms, fold_ids)

    if args.dry_run:
        print(preflight(cells, records, assignment))
        return

    run_grid(
        cells, records, assignment, args.out,
        image_size=args.image_size, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        type_weight=args.type_weight, seed=args.seed,
        num_workers=args.num_workers, limit=args.limit,
        micro_batch=args.micro_batch,
    )


if __name__ == "__main__":
    main()
