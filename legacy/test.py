import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
from metrics import dice_loss, dice_coef
from train import get_split, create_dir, run_paths, DATASET_PATH, MANIFEST_PATH, H, W
import splits as split_lib


def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((H, 10, 3)) * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


def metadata_lookup(manifest_path):
    """image path -> (patient id, tumor type), so scores can be broken down by type."""
    if not manifest_path:
        return {}
    root = os.path.dirname(os.path.abspath(manifest_path))
    records = split_lib.resolve_paths(split_lib.load_manifest(manifest_path), root)
    return {r["image"]: (r["pid"], r.get("label_name", str(r["label"]))) for r in records}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net on the test split.")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--protocol", choices=("patient", "slice"), default="patient",
                        help="must match the protocol the model was trained with")
    parser.add_argument("--model", default=None, help="override the checkpoint path")
    parser.add_argument("--split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    """ Seeding """
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    """ Directory for storing files """
    create_dir(os.path.join("results", args.protocol))
    create_dir("files")

    default_model, _ = run_paths(args.protocol)
    model_path = args.model or default_model

    """ Load the model """
    if not os.path.exists(model_path):
        raise SystemExit(
            f"No trained model at {model_path!r}. Run:\n"
            f"    python train.py --protocol {args.protocol}"
            + (f" --manifest {args.manifest}" if args.manifest else "")
        )
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(model_path)

    """ Dataset -- same protocol and seed as training, so this is the held-out split """
    (train_x, _), (valid_x, _), (test_x, test_y) = get_split(
        args.manifest, args.protocol, args.dataset, args.split, args.seed)
    print(f"Test : {len(test_x)} - {len(test_y)}")

    meta = metadata_lookup(args.manifest)

    """ Prediction and Evaluation """
    SCORE = []
    for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_y)):
        """ Extracting the name """
        name = os.path.basename(x_path)
        pid, label_name = meta.get(x_path, ("", ""))

        """ Reading the image """
        image = cv2.imread(x_path, cv2.IMREAD_COLOR) ## [H, w, 3]
        image = cv2.resize(image, (W, H))            ## [H, w, 3]
        x = image/255.0                              ## [H, w, 3]
        x = np.expand_dims(x, axis=0)                ## [1, H, w, 3]

        """ Reading the mask """
        mask = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        """ Prediction """
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_results(image, mask, y_pred, os.path.join("results", args.protocol, name))

        """ Flatten the array """
        mask = mask/255.0
        mask = (mask > 0.5).astype(np.int32).flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        accuracy_value = accuracy_score(mask, y_pred)
        # dice_coef flattens with a Keras layer, which needs a batch dimension.
        dice_score = float(dice_coef(mask.reshape(1, -1), y_pred.reshape(1, -1)))
        SCORE.append([name, pid, label_name, f1_value, jac_value, recall_value,
                      precision_value, accuracy_value, dice_score])

    df = pd.DataFrame(SCORE, columns=["Image", "PID", "Type", "F1", "Jaccard",
                                      "Recall", "Precision", "Accuracy", "Dice Score"])

    metric_cols = ["F1", "Jaccard", "Recall", "Precision", "Accuracy", "Dice Score"]
    print(f"\n=== protocol: {args.protocol} | {len(df)} test slices ===")
    for col in metric_cols:
        print(f"{col}: {df[col].mean():0.5f}")

    if df["Type"].any():
        print("\nBy tumor type:")
        print(df.groupby("Type")[metric_cols].mean().round(5).to_string())

    if df["PID"].any():
        # Per-patient means first, then average -- one patient with many slices
        # should not dominate the headline number.
        per_patient = df.groupby("PID")[metric_cols].mean()
        print(f"\nPatient-averaged over {len(per_patient)} patients:")
        for col in metric_cols:
            print(f"{col}: {per_patient[col].mean():0.5f}")

    score_path = os.path.join("files", f"score_{args.protocol}.csv")
    df.to_csv(score_path, index=False)
    print(f"\nPer-slice scores: {score_path}")
    print(f"Prediction images: results/{args.protocol}/")
