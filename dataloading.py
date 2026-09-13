"""Dataset and loaders for the PyTorch training loop.

Reads the image/mask pairs named by a manifest record, applies one augmentation
arm, and yields tensors plus the tumor-type label.

Reproducibility and arm pairing
-------------------------------
Augmentation is seeded per item from (base_seed, epoch, index) rather than from
a single generator advanced by iteration order. Three consequences, all wanted:

  * DataLoader workers cannot desynchronise the stream, because no stream is
    shared between them;
  * re-running an epoch reproduces it exactly;
  * two augmentation arms with the same base seed see *identical* geometric
    transforms on every item, because Augmenter draws geometry and intensity
    from separate spawned streams. The arms are therefore paired item by item,
    which is what the analysis plan's within-fold contrasts assume.

Call set_epoch() between epochs or every epoch will augment identically.

The tumor type is converted from the dataset's 1/2/3 labels to the 0/1/2 class
indices the loss and the conditioning path expect. That conversion happens here,
once, so it cannot be done twice or forgotten downstream.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from augment import ARMS, Augmenter
from conditioning import NUM_TUMOR_TYPES

DEFAULT_IMAGE_SIZE = 256


class SliceDataset(Dataset):
    """One MRI slice per item: image, binary mask, tumor type."""

    def __init__(self, records, arm="standard", training=True,
                 image_size=DEFAULT_IMAGE_SIZE, seed=42, in_channels=3):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}.")
        if in_channels not in (1, 3):
            raise ValueError(f"in_channels must be 1 or 3, got {in_channels}.")
        if not records:
            raise ValueError("SliceDataset was given no records.")

        self.records = list(records)
        self.arm = arm
        self.training = training
        self.image_size = image_size
        self.seed = seed
        self.in_channels = in_channels
        self.epoch = 0

    def set_epoch(self, epoch):
        """Vary augmentation between epochs while keeping each one reproducible."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def _read(self, record):
        flag = cv2.IMREAD_COLOR if self.in_channels == 3 else cv2.IMREAD_GRAYSCALE
        image = cv2.imread(record["image"], flag)
        if image is None:
            raise RuntimeError(f"Could not read image {record['image']}")
        mask = cv2.imread(record["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask {record['mask']}")

        size = (self.image_size, self.image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        if image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)[..., None]
        return image, mask

    def __getitem__(self, index):
        record = self.records[index]
        image, mask = self._read(record)

        if self.training:
            # Entropy from the item, not from a shared generator: worker-safe,
            # epoch-reproducible, and identical across arms.
            augmenter = Augmenter(self.arm, seed=[self.seed, self.epoch, index])
            image, mask = augmenter(image, mask, training=True)

        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1)))

        label = int(record["label"])
        if not 1 <= label <= NUM_TUMOR_TYPES:
            raise ValueError(
                f"Slice {record.get('slice_id')} has tumor type {label}; the "
                f"Cheng dataset labels them 1..{NUM_TUMOR_TYPES}."
            )

        return {
            "image": image.float(),
            "mask": mask.float(),
            "type": torch.tensor(label - 1, dtype=torch.long),   # 1..3 -> 0..2
            "slice_id": str(record.get("slice_id", index)),
            "pid": str(record.get("pid", "")),
        }


def make_loader(records, arm="standard", training=True, batch_size=16,
                image_size=DEFAULT_IMAGE_SIZE, seed=42, num_workers=2,
                in_channels=3, shuffle=None):
    """A DataLoader over one split. Shuffles when training unless told otherwise."""
    dataset = SliceDataset(records, arm=arm, training=training,
                           image_size=image_size, seed=seed,
                           in_channels=in_channels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=bool(num_workers),
    )
    return loader
