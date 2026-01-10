import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch


def list_msd_task01_cases(dataset_root: Path) -> List[Tuple[Path, Path]]:
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Expected imagesTr/ and labelsTr/ in MSD Task01 root")
    images = sorted(images_dir.glob("*.nii.gz"))
    labels = {p.stem.replace(".nii", ""): p for p in labels_dir.glob("*.nii.gz")}
    pairs = []
    for image_path in images:
        key = image_path.stem.replace(".nii", "")
        if key not in labels:
            raise FileNotFoundError(f"Missing label for {image_path.name}")
        pairs.append((image_path, labels[key]))
    return pairs


def split_cases(
    cases: Sequence[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    rng = random.Random(seed)
    indices = list(range(len(cases)))
    rng.shuffle(indices)
    shuffled = [cases[i] for i in indices]
    train_split = int(train_ratio * len(shuffled))
    val_split = int((train_ratio + val_ratio) * len(shuffled))
    return {
        "train": shuffled[:train_split],
        "val": shuffled[train_split:val_split],
        "test": shuffled[val_split:],
    }


def _case_id(path: Path) -> str:
    return path.stem.replace(".nii", "")


def load_case_ids(list_path: Path) -> List[str]:
    with list_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line and not line.startswith("#")]


def build_splits(
    cases: Sequence[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    list_files: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, List[Tuple[Path, Path]]]:
    if list_files and any(list_files.values()):
        required = ["train", "val", "test"]
        if not all(list_files.get(key) for key in required):
            raise ValueError("list_files must include train, val, and test lists when provided")
        case_map = {_case_id(img): (img, lbl) for img, lbl in cases}
        splits = {}
        for key in required:
            ids = load_case_ids(Path(list_files[key]))
            missing = [case_id for case_id in ids if case_id not in case_map]
            if missing:
                raise FileNotFoundError(f"Missing cases for split '{key}': {missing[:3]}")
            splits[key] = [case_map[case_id] for case_id in ids]
        return splits
    return split_cases(cases, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


def _to_channel_first(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume, got shape {volume.shape}")
    if volume.shape[-1] == 4:
        return np.transpose(volume, (3, 2, 0, 1))
    if volume.shape[0] == 4:
        return np.transpose(volume, (0, 3, 1, 2))
    raise ValueError(f"Unexpected channel dimension for volume with shape {volume.shape}")


def normalize_modalities(
    volume: np.ndarray,
    percentiles: Tuple[float, float],
) -> np.ndarray:
    normalized = np.zeros_like(volume, dtype=np.float32)
    for idx in range(volume.shape[0]):
        channel = volume[idx]
        mask = channel != 0
        if mask.any():
            lo, hi = np.percentile(channel[mask], percentiles)
            channel = np.clip(channel, lo, hi)
            mean = channel[mask].mean()
            std = channel[mask].std()
        else:
            mean = channel.mean()
            std = channel.std()
        std = std if std > 0 else 1.0
        normalized[idx] = (channel - mean) / std
    return normalized


def one_hot_encode(label: np.ndarray, num_classes: int) -> np.ndarray:
    label = label.astype(np.int64)
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label volume, got shape {label.shape}")
    one_hot = np.eye(num_classes, dtype=np.float32)[label]
    return np.transpose(one_hot, (3, 2, 0, 1))


def _pad_to_shape(volume: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    pad_width = []
    for dim, target in zip(volume.shape, target_shape):
        total = max(target - dim, 0)
        pad_before = total // 2
        pad_after = total - pad_before
        pad_width.append((pad_before, pad_after))
    return np.pad(volume, pad_width, mode="constant")


def crop_or_pad(volume: np.ndarray, center: Sequence[int], roi_size: Sequence[int]) -> np.ndarray:
    volume = _pad_to_shape(volume, roi_size)
    slices = []
    for dim, c, size in zip(volume.shape, center, roi_size):
        start = int(c - size // 2)
        start = max(start, 0)
        end = start + size
        if end > dim:
            end = dim
            start = end - size
        slices.append(slice(start, end))
    return volume[tuple(slices)]


def sample_center(label: np.ndarray, roi_size: Sequence[int], pos_ratio: float, rng: np.random.Generator) -> Tuple[int, int, int]:
    foreground = np.argwhere(label > 0)
    if foreground.size > 0 and rng.random() < pos_ratio:
        center = foreground[rng.integers(0, len(foreground))]
    else:
        center = np.array([rng.integers(0, dim) for dim in label.shape])
    center = np.maximum(center, np.array(roi_size) // 2)
    center = np.minimum(center, np.array(label.shape) - np.array(roi_size) // 2 - 1)
    return tuple(center.tolist())


class MSDTask01Dataset3D(torch.utils.data.Dataset):
    def __init__(
        self,
        cases: Sequence[Tuple[Path, Path]],
        roi_size: Optional[Sequence[int]],
        label_mode: str,
        num_classes: int,
        pos_ratio: float,
        percentiles: Tuple[float, float],
        mode: str = "train",
        seed: int = 42,
    ):
        self.cases = list(cases)
        self.roi_size = tuple(roi_size) if roi_size is not None else None
        self.label_mode = label_mode
        self.num_classes = num_classes
        self.pos_ratio = pos_ratio
        self.percentiles = percentiles
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        image_path, label_path = self.cases[idx]
        image = nib.load(str(image_path)).get_fdata().astype(np.float32)
        label = nib.load(str(label_path)).get_fdata().astype(np.int64)

        image = _to_channel_first(image)
        image = normalize_modalities(image, self.percentiles)
        label = label.astype(np.int64)

        if self.label_mode == "binary":
            label = (label > 0).astype(np.int64)
            num_classes = 2
        else:
            num_classes = self.num_classes

        if self.roi_size is not None:
            center = sample_center(label, self.roi_size, self.pos_ratio, self.rng)
            image = np.stack([crop_or_pad(ch, center, self.roi_size) for ch in image], axis=0)
            label = crop_or_pad(label, center, self.roi_size)

        label_one_hot = one_hot_encode(label, num_classes)
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label_one_hot.astype(np.float32))
        return image_tensor, label_tensor
