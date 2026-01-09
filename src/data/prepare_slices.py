import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import nibabel as nib
import numpy as np


CHANNEL_TO_INDEX = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype="uint8")
    scaled = (volume - vmin) / (vmax - vmin)
    return (scaled * 255.0).astype("uint8")


def select_slice_indices(image_array: np.ndarray, slices_per_volume: int) -> List[int]:
    """Select slice indices with highest voxel sums (proxy for information content)."""
    sums = [np.sum(image_array[:, :, i]) for i in range(image_array.shape[2])]
    top_indices = np.argsort(sums)[::-1][:slices_per_volume]
    return top_indices.tolist()


def select_mask_biased_indices(
    mask_array: np.ndarray,
    slices_per_volume: int,
    rng: np.random.Generator,
) -> List[int]:
    """Prefer slices with non-empty masks, then backfill with highest totals."""
    sums = np.array([np.sum(mask_array[:, :, i]) for i in range(mask_array.shape[2])])
    non_empty = np.where(sums > 0)[0]
    if len(non_empty) >= slices_per_volume:
        chosen = rng.choice(non_empty, size=slices_per_volume, replace=False)
        return sorted(chosen.tolist())
    remaining = np.setdiff1d(np.arange(mask_array.shape[2]), non_empty)
    ranked = remaining[np.argsort(sums[remaining])[::-1]]
    filled = np.concatenate([non_empty, ranked])[:slices_per_volume]
    return filled.tolist()


def save_slices(
    array: np.ndarray,
    slice_indices: List[int],
    output_dir: Path,
    prefix: str,
    counter_offset: int,
):
    for idx, slice_idx in enumerate(slice_indices):
        data = array[:, :, slice_idx]
        filename = f"{prefix}_{counter_offset + idx:05d}.png"
        output_path = output_dir / filename
        imageio.imwrite(output_path, data)


def build_file_lists_brats(dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    modalities = []
    for pattern in ["*t1.nii.gz", "*t1ce.nii.gz", "*t2.nii.gz", "*flair.nii.gz"]:
        modalities.extend(sorted(dataset_root.rglob(pattern)))
    segmentations = sorted(dataset_root.rglob("*seg.nii.gz"))
    # replicate segmentations to align with 4 modalities per case
    segmentations = segmentations * 4
    return modalities, segmentations


def build_file_lists_msd(dataset_root: Path) -> List[Tuple[Path, Path]]:
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Expected imagesTr/ and labelsTr/ in MSD Task01 root")
    images = sorted(images_dir.glob("*.nii.gz"))
    labels = {p.name.replace(".nii.gz", ""): p for p in labels_dir.glob("*.nii.gz")}
    pairs = []
    for image_path in images:
        key = image_path.name.replace(".nii.gz", "")
        if key not in labels:
            raise FileNotFoundError(f"Missing label for {image_path.name}")
        pairs.append((image_path, labels[key]))
    return pairs


def load_msd_channel(image_path: Path, channel: str) -> np.ndarray:
    image = nib.load(str(image_path)).get_fdata()
    if image.ndim != 4:
        raise ValueError(f"Expected 4D image volume for MSD, got shape {image.shape}")
    channel_idx = CHANNEL_TO_INDEX[channel]
    # MSD Task01 is commonly stored as (H, W, D, C) with order [t1, t1ce, t2, flair].
    if image.shape[-1] == 4:
        return image[..., channel_idx]
    if image.shape[0] == 4:
        return image[channel_idx, ...]
    raise ValueError(f"Unexpected MSD channel dimension for {image_path}")


def convert_label_mode(mask_array: np.ndarray, label_mode: str) -> np.ndarray:
    if label_mode == "binary":
        mask_array = (mask_array > 0).astype("uint8")
    else:
        mask_array = mask_array.astype("uint8")
    return mask_array


def build_output_paths(output_root: Path) -> Dict[str, Path]:
    output_paths = {
        "train_images": output_root / "train_frames" / "train",
        "train_masks": output_root / "train_masks" / "train",
        "val_images": output_root / "val_frames" / "val",
        "val_masks": output_root / "val_masks" / "val",
        "test_images": output_root / "test_frames" / "test",
        "test_masks": output_root / "test_masks" / "test",
    }
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_paths


def split_items(items: List, train_ratio: float, val_ratio: float) -> Dict[str, List]:
    total = len(items)
    train_split = int(train_ratio * total)
    val_split = int((train_ratio + val_ratio) * total)
    return {
        "train": items[:train_split],
        "val": items[train_split:val_split],
        "test": items[val_split:],
    }


def prepare_brats(
    dataset_root: Path,
    output_root: Path,
    slices_per_volume: int,
    label_mode: str,
    train_ratio: float,
    val_ratio: float,
):
    output_paths = build_output_paths(output_root)
    brains, segs = build_file_lists_brats(dataset_root)
    if len(brains) != len(segs):
        raise ValueError("Number of modality volumes does not match segmentation volumes")

    splits = split_items(list(zip(brains, segs)), train_ratio, val_ratio)
    slice_cache: Dict[Path, List[int]] = {}

    for split_name, pairs in splits.items():
        for idx, (brain_path, seg_path) in enumerate(pairs):
            if seg_path not in slice_cache:
                image_array = nib.load(str(seg_path)).get_fdata()
                slice_cache[seg_path] = select_slice_indices(image_array, slices_per_volume)

            slice_indices = slice_cache[seg_path]
            counter_offset = idx * slices_per_volume

            img_dir_key = f"{split_name}_images"
            mask_dir_key = f"{split_name}_masks"

            brain_array = nib.load(str(brain_path)).get_fdata()
            mask_array = nib.load(str(seg_path)).get_fdata()
            mask_array = convert_label_mode(mask_array, label_mode)

            save_slices(brain_array, slice_indices, output_paths[img_dir_key], f"{split_name}_frame", counter_offset)
            save_slices(mask_array, slice_indices, output_paths[mask_dir_key], f"{split_name}_mask", counter_offset)


def prepare_msd_task01(
    dataset_root: Path,
    output_root: Path,
    slices_per_volume: int,
    channel: str,
    label_mode: str,
    train_ratio: float,
    val_ratio: float,
):
    output_paths = build_output_paths(output_root)
    pairs = build_file_lists_msd(dataset_root)
    splits = split_items(pairs, train_ratio, val_ratio)
    rng = np.random.default_rng(42)

    for split_name, split_pairs in splits.items():
        for idx, (image_path, label_path) in enumerate(split_pairs):
            image_array = load_msd_channel(image_path, channel)
            label_array = nib.load(str(label_path)).get_fdata()
            label_array = convert_label_mode(label_array, label_mode)

            slice_indices = select_mask_biased_indices(label_array, slices_per_volume, rng)
            counter_offset = idx * slices_per_volume

            img_dir_key = f"{split_name}_images"
            mask_dir_key = f"{split_name}_masks"

            image_uint8 = normalize_to_uint8(image_array)
            save_slices(image_uint8, slice_indices, output_paths[img_dir_key], f"{split_name}_frame", counter_offset)
            save_slices(label_array, slice_indices, output_paths[mask_dir_key], f"{split_name}_mask", counter_offset)


def prepare_dataset(
    dataset_root: Path,
    output_root: Path,
    slices_per_volume: int = 20,
    dataset_format: str = "brats",
    channel: str = "flair",
    label_mode: str = "binary",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
):
    if dataset_format == "brats":
        prepare_brats(dataset_root, output_root, slices_per_volume, label_mode, train_ratio, val_ratio)
    elif dataset_format == "msd_task01":
        prepare_msd_task01(dataset_root, output_root, slices_per_volume, channel, label_mode, train_ratio, val_ratio)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    print(f"Finished preparing dataset at {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NIfTI volumes to PNG slices")
    parser.add_argument(
        "--dataset-format",
        choices=["brats", "msd_task01"],
        default="brats",
        help="Dataset format to process",
    )
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root")
    parser.add_argument("--output-root", default="./Dataset", help="Where to store PNG slices")
    parser.add_argument("--slices-per-volume", type=int, default=20, help="Number of slices to export per volume")
    parser.add_argument(
        "--channel",
        choices=sorted(CHANNEL_TO_INDEX.keys()),
        default="flair",
        help="MRI channel to extract for msd_task01",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="Export masks as binary or multiclass labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    prepare_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        slices_per_volume=args.slices_per_volume,
        dataset_format=args.dataset_format,
        channel=args.channel,
        label_mode=args.label_mode,
    )


if __name__ == "__main__":
    main()
