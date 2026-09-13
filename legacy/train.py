import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from unet import build_unet
from metrics import dice_loss, dice_coef
import splits as split_lib

""" Global parameters """
H = 256
W = 256

""" Dataset location, used only when no manifest is available. Override with the
    BTS_DATASET environment variable. The directory must contain an "images" and
    a "masks" subdirectory. """
DATASET_PATH = os.environ.get(
    "BTS_DATASET",
    "/content/drive/MyDrive/Brain Tumor Segmentation Dataset",
)

""" Manifest produced by prepare_dataset.py. Preferred: it carries patient IDs,
    which the raw PNG directories do not. """
MANIFEST_PATH = os.environ.get("BTS_MANIFEST", "")


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_paths(protocol):
    """Checkpoint and log paths, kept separate per protocol so a patient-disjoint
    run does not overwrite a slice-level one (you need both to compare)."""
    return (
        os.path.join("files", f"model_{protocol}.keras"),
        os.path.join("files", f"log_{protocol}.csv"),
    )


def load_split(manifest_path, protocol="patient", split=0.1, seed=42):
    """Split via the manifest, so patients can be kept disjoint."""
    records = split_lib.load_manifest(manifest_path)
    records = split_lib.resolve_paths(records, os.path.dirname(os.path.abspath(manifest_path)))
    parts = split_lib.make_split(records, protocol, valid_frac=split, test_frac=split, seed=seed)

    print(f"Protocol: {protocol} (seed {seed})\n")
    print(split_lib.describe(parts))
    print()

    def xy(name):
        return [r["image"] for r in parts[name]], [r["mask"] for r in parts[name]]

    return xy("train"), xy("valid"), xy("test")


def load_dataset(path, split=0.1):
    """Fallback for a bare images/ + masks/ directory with no manifest.

    Pairs by filename: the two directories do not necessarily hold the same
    files (duplicate downloads leave images with no mask and vice versa), and
    zipping two sorted lists would silently misalign every pair after the first
    gap. Unmatched files are dropped.

    This cannot produce a patient-disjoint split -- the PNGs carry no patient
    ID. Use prepare_dataset.py and a manifest for anything you intend to report.
    """
    images = {os.path.basename(p): p for p in glob(os.path.join(path, "images", "*.png"))}
    masks = {os.path.basename(p): p for p in glob(os.path.join(path, "masks", "*.png"))}

    paired_names = sorted(set(images) & set(masks))

    if not paired_names:
        raise ValueError(
            f"No image/mask pairs found under {path!r}. Expected an 'images' and a "
            f"'masks' subdirectory of .png files with matching names "
            f"(found {len(images)} images, {len(masks)} masks). "
            f"Set BTS_DATASET to the correct location."
        )

    unpaired = len(images) + len(masks) - 2 * len(paired_names)
    if unpaired:
        print(f"Skipping {unpaired} file(s) with no counterpart "
              f"({len(images) - len(paired_names)} images, {len(masks) - len(paired_names)} masks).")

    dataset = [(images[name], masks[name]) for name in paired_names]
    dataset = shuffle(dataset, random_state=42)

    split_size = int(len(dataset) * split)
    if split_size < 1:
        raise ValueError(
            f"Only {len(dataset)} pair(s) available; split={split} leaves no "
            f"validation or test data."
        )

    train, temp = train_test_split(dataset, test_size=split_size * 2, random_state=42)
    valid, test = train_test_split(temp, test_size=0.5, random_state=42)

    train_x, train_y = zip(*train)
    valid_x, valid_y = zip(*valid)
    test_x, test_y = zip(*test)

    return (list(train_x), list(train_y)), (list(valid_x), list(valid_y)), (list(test_x), list(test_y))


def get_split(manifest_path=None, protocol="patient", dataset_path=None, split=0.1, seed=42):
    """Load a split from the manifest when there is one, else from the directory."""
    manifest_path = manifest_path if manifest_path is not None else MANIFEST_PATH
    dataset_path = dataset_path if dataset_path is not None else DATASET_PATH

    if manifest_path:
        print(f"Manifest: {manifest_path}")
        return load_split(manifest_path, protocol, split, seed)

    print(f"Dataset: {dataset_path}")
    print("WARNING: no manifest, so the split is slice-level and patients appear\n"
          "         in both train and test. Reported scores will be inflated.\n"
          "         Build a manifest with prepare_dataset.py, then pass --manifest.\n")
    return load_dataset(dataset_path, split)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None:
        raise RuntimeError(f"Could not read image {path}")
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (h, w)
    if x is None:
        raise RuntimeError(f"Could not read mask {path}")
    x = cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST)  ## (h, w)
    x = x / 255.0               ## (h, w)
    x = x.astype(np.float32)    ## (h, w)
    x = np.expand_dims(x, axis=-1)## (h, w, 1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net baseline.")
    parser.add_argument("--manifest", default=MANIFEST_PATH,
                        help="manifest.csv from prepare_dataset.py (enables patient-disjoint splits)")
    parser.add_argument("--dataset", default=DATASET_PATH,
                        help="fallback images/+masks/ directory, used when no manifest is given")
    parser.add_argument("--protocol", choices=("patient", "slice"), default="patient",
                        help="patient: honest, patient-disjoint. slice: the leaky protocol, for comparison")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--split", type=float, default=0.1, help="valid and test fraction each")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    """ Seeding """
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    """ Directory for storing files """
    create_dir("files")

    model_path, csv_path = run_paths(args.protocol)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = get_split(
        args.manifest, args.protocol, args.dataset, args.split, args.seed)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=args.batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=args.batch_size)

    """ Model """
    model = build_unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(args.lr), metrics=[dice_coef])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    print(f"\nBest checkpoint: {model_path}")
    print(f"Training log   : {csv_path}")
