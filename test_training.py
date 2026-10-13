"""Tests for dataloading.py, losses.py and engine.py.

Run directly: python test_training.py   (requires torch, opencv, numpy)

Synthetic slices are written to a temporary directory so the whole data path --
imread, resize, augment, tensor conversion -- is exercised for real rather than
mocked.
"""

import os
import shutil
import tempfile

import cv2
import numpy as np
import torch

import backbones
from dataloading import SliceDataset, make_loader
from engine import (evaluate, fit, load_checkpoint, pick_device, save_checkpoint,
                    train_epoch)
from losses import DiceBCELoss, JointLoss, binary_scores, dice_loss, type_accuracy

RESULTS = []
TMP = None


def test(fn):
    RESULTS.append(fn)
    return fn


def make_dataset_on_disk(root, n=12, size=32):
    """Slices with a bright disc whose position depends on the tumor type, so a
    type head has something real to learn and segmentation is solvable fast."""
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    records = []
    yy, xx = np.mgrid[0:size, 0:size]

    for i in range(n):
        label = (i % 3) + 1                      # 1, 2, 3
        cy = size * (0.3 + 0.2 * (label - 1))
        disc = ((yy - cy) ** 2 + (xx - size * 0.5) ** 2) < (size * 0.2) ** 2

        image = np.full((size, size), 40, dtype=np.uint8)
        image[disc] = 200
        mask = (disc * 255).astype(np.uint8)

        image_path = os.path.join(root, "images", f"{i}.png")
        mask_path = os.path.join(root, "masks", f"{i}.png")
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)

        records.append({"slice_id": str(i), "pid": f"P{i // 3}", "label": label,
                        "label_name": ["meningioma", "glioma", "pituitary"][label - 1],
                        "image": image_path, "mask": mask_path})
    return records


def tiny_model(name="unet", **kwargs):
    """A U-Net small enough to train in seconds on CPU."""
    return backbones.build(name, in_channels=3, out_channels=1,
                           widths=(4, 8), bottleneck=16, **kwargs)


# -------------------------------------------------------------------- data

@test
def dataset_yields_the_expected_tensors():
    records = make_dataset_on_disk(os.path.join(TMP, "d1"))
    ds = SliceDataset(records, arm="standard", training=True, image_size=32)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32), item["image"].shape
    assert item["mask"].shape == (1, 32, 32), item["mask"].shape
    assert item["image"].dtype == torch.float32 and item["mask"].dtype == torch.float32
    assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}


@test
def dataset_converts_labels_to_zero_based_class_indices():
    """The dataset labels types 1/2/3; losses and one-hot need 0/1/2. Doing it
    anywhere but here risks doing it twice or not at all."""
    records = make_dataset_on_disk(os.path.join(TMP, "d2"), n=3)
    ds = SliceDataset(records, training=False, image_size=32)
    got = sorted(int(ds[i]["type"]) for i in range(3))
    assert got == [0, 1, 2], got


@test
def an_out_of_range_tumor_type_is_rejected():
    records = make_dataset_on_disk(os.path.join(TMP, "d3"), n=2)
    records[0]["label"] = 0
    ds = SliceDataset(records, training=False, image_size=32)
    try:
        ds[0]
    except ValueError as exc:
        assert "1..3" in str(exc), exc
        return
    raise AssertionError("tumor type 0 was accepted")


@test
def evaluation_mode_does_not_augment():
    records = make_dataset_on_disk(os.path.join(TMP, "d4"), n=2)
    ds = SliceDataset(records, arm="region", training=False, image_size=32)
    a, b = ds[0]["image"], ds[0]["image"]
    assert torch.equal(a, b), "eval-mode reads were not identical"


@test
def augmentation_is_reproducible_within_an_epoch_and_varies_across_them():
    records = make_dataset_on_disk(os.path.join(TMP, "d5"), n=4)
    ds = SliceDataset(records, arm="region", training=True, image_size=32, seed=7)

    ds.set_epoch(0)
    first, again = ds[1]["image"], ds[1]["image"]
    assert torch.equal(first, again), "same epoch gave different augmentations"

    ds.set_epoch(1)
    assert not torch.equal(ds[1]["image"], first), "epoch change did not vary it"


@test
def arms_see_identical_geometry_on_the_same_item():
    """Paired arms are what the within-fold contrasts in the analysis assume."""
    records = make_dataset_on_disk(os.path.join(TMP, "d6"), n=4)
    made = [SliceDataset(records, arm=arm, training=True, image_size=32, seed=3)
            for arm in ("standard", "global", "region")]
    for ds in made:
        ds.set_epoch(2)
    masks = [ds[2]["mask"] for ds in made]
    assert torch.equal(masks[0], masks[1]) and torch.equal(masks[0], masks[2]), \
        "arms diverged geometrically on the same item"


@test
def loader_batches_and_carries_metadata():
    records = make_dataset_on_disk(os.path.join(TMP, "d7"), n=8)
    loader = make_loader(records, batch_size=4, image_size=32, num_workers=0)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 32, 32), batch["image"].shape
    assert batch["type"].shape == (4,)
    assert len(batch["pid"]) == 4 and isinstance(batch["pid"][0], str)


@test
def single_channel_input_is_supported():
    records = make_dataset_on_disk(os.path.join(TMP, "d8"), n=2)
    ds = SliceDataset(records, training=False, image_size=32, in_channels=1)
    assert ds[0]["image"].shape == (1, 32, 32), ds[0]["image"].shape


# ------------------------------------------------------------------ losses

@test
def dice_loss_rewards_a_correct_mask():
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    good = torch.where(target > 0, 10.0, -10.0)
    bad = -good
    assert dice_loss(good, target) < 0.05, dice_loss(good, target)
    assert dice_loss(bad, target) > 0.9, dice_loss(bad, target)


@test
def dice_loss_averages_per_item_not_over_the_flattened_batch():
    """One large tumor must not outvote a missed small one.

    Compared against the batch-flattened Dice this is meant to replace, rather
    than against a hand-picked threshold: with one item perfect and one missed
    entirely, per-item scores ~0.25 while flattening scores ~0.008, because the
    large tumor's correct pixels swamp the small tumor's absent ones.
    """
    target = torch.zeros(2, 1, 8, 8)
    target[0, :, :, :] = 1.0          # large
    target[1, :, 0, 0] = 1.0          # tiny
    logits = torch.where(target > 0, 10.0, -10.0)
    logits[1] = -10.0                 # miss the tiny one entirely

    per_item = float(dice_loss(logits, target))
    probs, flat_target = torch.sigmoid(logits).flatten(), target.flatten()
    flattened = 1.0 - float((2 * (probs * flat_target).sum() + 1)
                            / (probs.sum() + flat_target.sum() + 1))

    assert per_item > 10 * flattened, (
        f"per-item {per_item:.4f} is not meaningfully harsher than "
        f"flattened {flattened:.4f}")
    assert abs(per_item - 0.25) < 0.02, f"expected ~0.25, got {per_item:.4f}"


@test
def dicebce_weights_are_honoured():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    logits = torch.randn(1, 1, 8, 8)
    only_bce = DiceBCELoss(dice_weight=0.0)(logits, target)
    only_dice = DiceBCELoss(dice_weight=1.0)(logits, target)
    assert not torch.allclose(only_bce, only_dice)
    try:
        DiceBCELoss(dice_weight=1.5)
    except ValueError:
        return
    raise AssertionError("dice_weight=1.5 was accepted")


@test
def joint_loss_tolerates_a_backbone_with_no_type_head():
    target = torch.zeros(2, 1, 8, 8)
    logits = torch.randn(2, 1, 8, 8)
    criterion = JointLoss(type_weight=0.2)
    total, parts = criterion(logits, target, None, None)
    assert float(parts["type"]) == 0.0
    assert torch.isfinite(total)


@test
def joint_loss_adds_the_type_term_when_there_is_one():
    target = torch.zeros(2, 1, 8, 8)
    seg = torch.randn(2, 1, 8, 8)
    types = torch.tensor([0, 2])
    confident = torch.tensor([[9.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    wrong = torch.tensor([[0.0, 0.0, 9.0], [9.0, 0.0, 0.0]])
    criterion = JointLoss(type_weight=1.0)
    good, _ = criterion(seg, target, confident, types)
    bad, _ = criterion(seg, target, wrong, types)
    assert bad > good, (float(bad), float(good))


@test
def both_empty_counts_as_agreement():
    """Slices with no tumor exist; scoring a correct empty prediction 0 would
    punish the model for being right."""
    logits = torch.full((1, 1, 8, 8), -10.0)
    target = torch.zeros(1, 1, 8, 8)
    dice, iou = binary_scores(logits, target)
    assert float(dice[0]) == 1.0 and float(iou[0]) == 1.0, (dice, iou)


@test
def binary_scores_match_a_hand_computed_case():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :] = 1.0                       # 8 positives
    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, :1, :] = 10.0                      # predict 4, all correct
    dice, iou = binary_scores(logits, target)
    assert abs(float(dice[0]) - 2 * 4 / (4 + 8)) < 1e-5, float(dice[0])
    assert abs(float(iou[0]) - 4 / 8) < 1e-5, float(iou[0])


@test
def type_accuracy_handles_absence():
    assert type_accuracy(None, None) is None
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    assert type_accuracy(logits, torch.tensor([0, 1])) == 1.0
    assert type_accuracy(logits, torch.tensor([1, 1])) == 0.5


# ------------------------------------------------------------------ engine

@test
def one_epoch_of_training_runs_and_reports():
    records = make_dataset_on_disk(os.path.join(TMP, "e1"), n=8)
    loader = make_loader(records, batch_size=4, image_size=32, num_workers=0)
    model = tiny_model()
    criterion = JointLoss(type_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = train_epoch(model, loader, criterion, optimizer, pick_device())
    assert set(stats) == {"loss", "seg", "type"}
    assert all(np.isfinite(v) for v in stats.values()), stats


@test
def training_actually_reduces_the_loss():
    records = make_dataset_on_disk(os.path.join(TMP, "e2"), n=12)
    loader = make_loader(records, arm="standard", batch_size=4, image_size=32,
                         num_workers=0)
    torch.manual_seed(0)
    model = tiny_model()
    criterion = JointLoss(type_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = train_epoch(model, loader, criterion, optimizer, pick_device())["loss"]
    for _ in range(6):
        last = train_epoch(model, loader, criterion, optimizer, pick_device())["loss"]
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


@test
def accumulation_takes_one_step_per_group():
    """A model that only fits a micro-batch of 4 must still train at the
    protocol's effective batch of 16."""
    records = make_dataset_on_disk(os.path.join(TMP, "e10"), n=8)
    loader = make_loader(records, batch_size=2, image_size=32, num_workers=0)

    class CountingAdam(torch.optim.AdamW):
        steps = 0
        def step(self, *a, **k):
            CountingAdam.steps += 1
            return super().step(*a, **k)

    for accumulate, expected in ((1, 4), (2, 2), (4, 1)):
        CountingAdam.steps = 0
        model = tiny_model()
        train_epoch(model, loader, JointLoss(), CountingAdam(model.parameters(), lr=1e-3),
                    pick_device(), accumulate=accumulate)
        assert CountingAdam.steps == expected, \
            f"accumulate={accumulate}: {CountingAdam.steps} steps, expected {expected}"


@test
def a_trailing_partial_group_is_not_dropped():
    """5 micro-batches at accumulate=2 is 2 full groups plus a remainder; losing
    the remainder would silently discard part of every epoch."""
    records = make_dataset_on_disk(os.path.join(TMP, "e11"), n=10)
    loader = make_loader(records, batch_size=2, image_size=32, num_workers=0)

    class CountingAdam(torch.optim.AdamW):
        steps = 0
        def step(self, *a, **k):
            CountingAdam.steps += 1
            return super().step(*a, **k)

    CountingAdam.steps = 0
    model = tiny_model()
    train_epoch(model, loader, JointLoss(), CountingAdam(model.parameters(), lr=1e-3),
                pick_device(), accumulate=2)
    assert CountingAdam.steps == 3, f"expected 2 full groups + 1 remainder, got {CountingAdam.steps}"


@test
def accumulation_approximates_a_larger_batch():
    """Not exactly equal -- BatchNorm normalises over the micro-batch -- but the
    gradient direction must agree closely or the approximation is worthless."""
    records = make_dataset_on_disk(os.path.join(TMP, "e12"), n=8)
    big = make_loader(records, batch_size=8, image_size=32, num_workers=0, shuffle=False)
    small = make_loader(records, batch_size=2, image_size=32, num_workers=0, shuffle=False)

    def grads(loader, accumulate):
        torch.manual_seed(0)
        model = tiny_model()
        captured = {}

        class Capturing(torch.optim.SGD):
            def step(self, *a, **k):
                # grads are live here; train_epoch clears them straight after
                captured["g"] = torch.cat([p.grad.flatten().clone()
                                           for p in model.parameters()
                                           if p.grad is not None])
                return super().step(*a, **k)

        opt = Capturing(model.parameters(), lr=0.0)          # capture, do not move
        train_epoch(model, loader, JointLoss(0.0), opt, pick_device(), accumulate=accumulate)
        return captured["g"]

    a, b = grads(big, 1), grads(small, 4)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert cos > 0.9, f"accumulated gradient diverged from the full batch (cos={cos:.3f})"


@test
def an_invalid_accumulation_is_rejected():
    records = make_dataset_on_disk(os.path.join(TMP, "e13"), n=4)
    loader = make_loader(records, batch_size=2, image_size=32, num_workers=0)
    model = tiny_model()
    try:
        train_epoch(model, loader, JointLoss(), torch.optim.AdamW(model.parameters()),
                    pick_device(), accumulate=0)
    except ValueError as exc:
        assert "accumulate" in str(exc), exc
        return
    raise AssertionError("accumulate=0 was accepted")


@test
def evaluate_reports_metrics_and_optional_per_item_rows():
    records = make_dataset_on_disk(os.path.join(TMP, "e3"), n=6)
    loader = make_loader(records, training=False, batch_size=3, image_size=32,
                         num_workers=0)
    model = tiny_model("cond_unet")
    metrics, rows = evaluate(model, loader, JointLoss(), pick_device(), per_item=True)

    assert metrics["n"] == 6 and 0.0 <= metrics["dice"] <= 1.0
    assert metrics["type_acc"] is not None, "conditioned model reported no type accuracy"
    assert len(rows) == 6
    assert {"slice_id", "pid", "type", "predicted_type", "dice", "iou"} <= set(rows[0])


@test
def a_plain_backbone_reports_no_type_accuracy():
    records = make_dataset_on_disk(os.path.join(TMP, "e4"), n=4)
    loader = make_loader(records, training=False, batch_size=2, image_size=32,
                         num_workers=0)
    metrics = evaluate(tiny_model("unet"), loader, None, pick_device())
    assert metrics["type_acc"] is None


@test
def best_checkpoint_is_weights_only_and_last_keeps_the_optimizer():
    """best.pt is for evaluation and last.pt for resuming; storing optimizer
    state in both doubled every cell's disk footprint."""
    root = os.path.join(TMP, "e14")
    records = make_dataset_on_disk(root, n=8)
    train = make_loader(records, batch_size=4, image_size=32, num_workers=0)
    valid = make_loader(records, training=False, batch_size=4, image_size=32,
                        num_workers=0)
    best, last = os.path.join(root, "best.pt"), os.path.join(root, "last.pt")
    fit(tiny_model(), train, valid, epochs=2, patience=10,
        checkpoint_path=best, last_path=last, verbose=False)

    b = torch.load(best, map_location="cpu", weights_only=False)
    l = torch.load(last, map_location="cpu", weights_only=False)
    assert "optimizer" not in b and "model" in b, sorted(b)
    assert "optimizer" in l, "last.pt lost the state needed to resume"
    assert os.path.getsize(best) < os.path.getsize(last)
    load_checkpoint(best, tiny_model())          # still loads for evaluation


@test
def checkpoints_round_trip():
    records = make_dataset_on_disk(os.path.join(TMP, "e5"), n=4)
    path = os.path.join(TMP, "e5", "ckpt.pt")
    torch.manual_seed(0)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(path, model, optimizer, epoch=3, best_dice=0.42, history=[{"a": 1}])

    torch.manual_seed(1)
    restored = tiny_model()
    state = load_checkpoint(path, restored)
    assert state["epoch"] == 3 and abs(state["best_dice"] - 0.42) < 1e-9
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b), "weights differ after a round trip"


@test
def fit_trains_logs_and_checkpoints():
    root = os.path.join(TMP, "e6")
    records = make_dataset_on_disk(root, n=8)
    train = make_loader(records, batch_size=4, image_size=32, num_workers=0)
    valid = make_loader(records, training=False, batch_size=4, image_size=32,
                        num_workers=0)
    log = os.path.join(root, "log.csv")
    best = os.path.join(root, "best.pt")

    result = fit(tiny_model(), train, valid, epochs=3, lr=3e-3, patience=10,
                 checkpoint_path=best, log_path=log, verbose=False)

    assert len(result["history"]) == 3, result["history"]
    assert os.path.exists(best) and os.path.exists(log)
    with open(log) as fh:
        assert len(fh.readlines()) == 4, "log should hold a header plus three rows"
    assert result["best_dice"] >= 0.0


@test
def fit_resumes_instead_of_restarting():
    root = os.path.join(TMP, "e7")
    records = make_dataset_on_disk(root, n=8)
    train = make_loader(records, batch_size=4, image_size=32, num_workers=0)
    valid = make_loader(records, training=False, batch_size=4, image_size=32,
                        num_workers=0)
    last = os.path.join(root, "last.pt")

    torch.manual_seed(0)
    first = fit(tiny_model(), train, valid, epochs=2, patience=10,
                last_path=last, verbose=False)
    assert len(first["history"]) == 2

    torch.manual_seed(0)
    again = fit(tiny_model(), train, valid, epochs=2, patience=10,
                last_path=last, verbose=False)
    assert again["epochs_run"] == 0, "a finished cell was retrained"

    torch.manual_seed(0)
    more = fit(tiny_model(), train, valid, epochs=4, patience=10,
               last_path=last, verbose=False)
    assert len(more["history"]) == 4, len(more["history"])
    assert more["epochs_run"] == 2, more["epochs_run"]


@test
def early_stopping_fires():
    root = os.path.join(TMP, "e8")
    records = make_dataset_on_disk(root, n=6)
    train = make_loader(records, batch_size=3, image_size=32, num_workers=0)
    valid = make_loader(records, training=False, batch_size=3, image_size=32,
                        num_workers=0)
    result = fit(tiny_model(), train, valid, epochs=25, lr=1e-6, patience=1,
                 verbose=False)
    assert result["stopped_early"], "patience=1 at lr=1e-6 should have stopped"
    assert len(result["history"]) < 25


@test
def an_empty_loader_is_reported_not_silently_averaged():
    records = make_dataset_on_disk(os.path.join(TMP, "e9"), n=4)
    loader = make_loader(records, training=False, batch_size=2, image_size=32,
                         num_workers=0)
    try:
        evaluate(tiny_model(), loader, None, pick_device(), max_batches=0)
    except RuntimeError as exc:
        assert "no batches" in str(exc), exc
        return
    raise AssertionError("an empty pass returned metrics")


if __name__ == "__main__":
    TMP = tempfile.mkdtemp(prefix="bts-training-tests-")
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
