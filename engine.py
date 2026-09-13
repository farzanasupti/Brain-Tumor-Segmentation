"""Training and evaluation loop for one cell of the experiment grid.

One cell is a (backbone, augmentation arm, fold) triple. fit() trains it, early
stops on validation Dice, keeps the best checkpoint, appends a CSV row per
epoch, and can resume an interrupted run -- a Colab disconnect at hour sixty
must not cost the study a cell.

Every backbone is driven through backbones.split_output(), so a model that also
predicts tumor type trains through the same code path as one that does not.
"""

import csv
import os
import time

import torch

import segmetrics
from backbones import split_output
from losses import JointLoss, binary_scores, type_accuracy

EPOCH_FIELDS = ("epoch", "train_loss", "train_seg", "train_type", "valid_loss",
                "valid_dice", "valid_iou", "valid_type_acc", "lr", "seconds")


def pick_device(prefer="auto"):
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(batch, device):
    return (batch["image"].to(device, non_blocking=True),
            batch["mask"].to(device, non_blocking=True),
            batch["type"].to(device, non_blocking=True))


def train_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    totals = {"loss": 0.0, "seg": 0.0, "type": 0.0}
    seen = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, masks, types = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        seg_logits, type_logits = split_output(model(images))
        loss, parts = criterion(seg_logits, masks, type_logits, types)
        loss.backward()
        optimizer.step()

        n = images.size(0)
        totals["loss"] += loss.item() * n
        totals["seg"] += parts["segmentation"].item() * n
        totals["type"] += parts["type"].item() * n
        seen += n

    if seen == 0:
        raise RuntimeError("train_epoch saw no batches; the loader is empty.")
    return {k: v / seen for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, criterion=None, device=None, max_batches=None,
             per_item=False, boundary=False, save_predictions_to=None):
    """Validation / test pass. Returns mean Dice, IoU, loss and type accuracy.

    With per_item=True also returns one row per slice, carrying pid and type so
    results can be aggregated per patient and per tumor type later.

    `boundary=True` adds HD95 and ASSD per slice. They cost a pair of distance
    transforms each, so they are off during per-epoch validation and on at test
    time. `save_predictions_to` writes each predicted mask as a PNG, which makes
    any metric thought of later computable without retraining a single cell.
    """
    device = device or pick_device()
    if save_predictions_to:
        os.makedirs(save_predictions_to, exist_ok=True)
    model.eval()

    dice_sum = iou_sum = loss_sum = 0.0
    type_correct = type_seen = 0
    seen = 0
    rows = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, masks, types = _to_device(batch, device)

        seg_logits, type_logits = split_output(model(images))
        dice, iou = binary_scores(seg_logits, masks)

        n = images.size(0)
        dice_sum += dice.sum().item()
        iou_sum += iou.sum().item()
        seen += n

        if criterion is not None:
            loss, _ = criterion(seg_logits, masks, type_logits, types)
            loss_sum += loss.item() * n

        if type_logits is not None:
            type_correct += (type_logits.argmax(1) == types).sum().item()
            type_seen += n

        if per_item or save_predictions_to:
            pred_masks = (torch.sigmoid(seg_logits) > 0.5).float().cpu().numpy()
            true_masks = (masks > 0.5).float().cpu().numpy()
            predicted = (type_logits.argmax(1).tolist() if type_logits is not None
                         else [None] * n)

            for j in range(n):
                if save_predictions_to:
                    import cv2
                    cv2.imwrite(
                        os.path.join(save_predictions_to,
                                     f"{batch['slice_id'][j]}.png"),
                        (pred_masks[j].squeeze() * 255).astype("uint8"))
                if not per_item:
                    continue

                row = {
                    "slice_id": batch["slice_id"][j],
                    "pid": batch["pid"][j],
                    "type": int(types[j].item()),
                    "predicted_type": predicted[j],
                    "dice": float(dice[j].item()),
                    "iou": float(iou[j].item()),
                    "pred_pixels": int(pred_masks[j].sum()),
                    "true_pixels": int(true_masks[j].sum()),
                }
                if boundary:
                    row["hd95"] = segmetrics.hd95(pred_masks[j], true_masks[j])
                    row["assd"] = segmetrics.assd(pred_masks[j], true_masks[j])
                rows.append(row)

    if seen == 0:
        raise RuntimeError("evaluate saw no batches; the loader is empty.")

    metrics = {
        "dice": dice_sum / seen,
        "iou": iou_sum / seen,
        "loss": loss_sum / seen if criterion is not None else float("nan"),
        "type_acc": type_correct / type_seen if type_seen else None,
        "n": seen,
    }
    return (metrics, rows) if per_item else metrics


def _append_row(path, row):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EPOCH_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in EPOCH_FIELDS})


def save_checkpoint(path, model, optimizer, epoch, best_dice, history):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    tmp = f"{path}.tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "history": history,
    }, tmp)
    os.replace(tmp, path)      # atomic: a crash mid-write cannot corrupt the file


def load_checkpoint(path, model, optimizer=None, device=None):
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    return state


def fit(model, train_loader, valid_loader, *, epochs=150, lr=1e-4,
        weight_decay=1e-2, type_weight=0.2, dice_weight=0.5, patience=20,
        device=None, checkpoint_path=None, last_path=None, log_path=None,
        resume=True, max_batches=None, verbose=True):
    """Train one cell. Returns the best validation metrics and the history."""
    device = device or pick_device()
    model.to(device)

    criterion = JointLoss(type_weight=type_weight, dice_weight=dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch, best_dice, history = 0, -1.0, []

    if resume and last_path and os.path.exists(last_path):
        state = load_checkpoint(last_path, model, optimizer, device)
        start_epoch = state.get("epoch", 0) + 1
        best_dice = state.get("best_dice", -1.0)
        history = state.get("history", [])
        if verbose:
            print(f"Resumed from {last_path} at epoch {start_epoch} "
                  f"(best Dice {best_dice:.4f})")

    # Constructed after the resume, so the cosine schedule is restored by
    # last_epoch rather than by replaying step() calls before any optimizer step.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, last_epoch=start_epoch - 1)

    if start_epoch >= epochs:
        if verbose:
            print(f"Already trained {start_epoch}/{epochs} epochs; nothing to do.")
        return {"best_dice": best_dice, "epochs_run": 0, "history": history,
                "stopped_early": False}

    since_improved = 0
    stopped_early = False

    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        started = time.time()
        train_stats = train_epoch(model, train_loader, criterion, optimizer,
                                  device, max_batches)
        valid = evaluate(model, valid_loader, criterion, device, max_batches)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "train_seg": round(train_stats["seg"], 6),
            "train_type": round(train_stats["type"], 6),
            "valid_loss": round(valid["loss"], 6),
            "valid_dice": round(valid["dice"], 6),
            "valid_iou": round(valid["iou"], 6),
            "valid_type_acc": (round(valid["type_acc"], 6)
                               if valid["type_acc"] is not None else None),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        _append_row(log_path, row)

        improved = valid["dice"] > best_dice
        if improved:
            best_dice = valid["dice"]
            since_improved = 0
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, optimizer, epoch,
                                best_dice, history)
        else:
            since_improved += 1

        if last_path:
            save_checkpoint(last_path, model, optimizer, epoch, best_dice, history)

        if verbose:
            mark = " *" if improved else ""
            acc = ("" if valid["type_acc"] is None
                   else f" type {valid['type_acc']:.3f}")
            print(f"epoch {epoch:3d}  train {train_stats['loss']:.4f}  "
                  f"val {valid['loss']:.4f}  dice {valid['dice']:.4f}"
                  f"{acc}  {row['seconds']}s{mark}")

        if since_improved >= patience:
            stopped_early = True
            if verbose:
                print(f"Early stop: no improvement in {patience} epochs.")
            break

    return {
        "best_dice": best_dice,
        "epochs_run": len(history) - len([h for h in history[:start_epoch]]),
        "history": history,
        "stopped_early": stopped_early,
    }
