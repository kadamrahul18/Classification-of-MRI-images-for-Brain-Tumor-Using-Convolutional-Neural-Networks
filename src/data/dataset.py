import math
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class SliceDataset:
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        class_names: Iterable[str],
        image_size: int = 256,
        augmentation=None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_ids = sorted(self.images_dir.glob("*.png"))
        self.mask_ids = sorted(self.masks_dir.glob("*.png"))
        if len(self.image_ids) != len(self.mask_ids):
            raise ValueError("Number of images and masks does not match")
        self.num_classes = len(list(class_names))
        self.image_size = (image_size, image_size)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_path = self.image_ids[idx]
        mask_path = self.mask_ids[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Failed to read image or mask for index {idx}")

        mask = np.where(mask == 4, 3, mask)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=-1)
        mask = to_categorical(mask, num_classes=self.num_classes).astype("float32")
        return image, mask


class DataLoader(Sequence):
    def __init__(
        self,
        dataset: SliceDataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = [self.dataset[i] for i in batch_indexes]
        images, masks = zip(*batch)
        return np.stack(images, axis=0), np.stack(masks, axis=0)

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
        else:
            self.indexes = np.arange(len(self.dataset))


def build_dataloader(
    images_dir: Path,
    masks_dir: Path,
    class_names: Iterable[str],
    batch_size: int,
    image_size: int,
    augmentation=None,
    shuffle: bool = False,
) -> DataLoader:
    dataset = SliceDataset(images_dir, masks_dir, class_names, image_size, augmentation)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
