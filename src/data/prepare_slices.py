import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import nibabel as nib
import numpy as np


def select_slice_indices(image_array: np.ndarray, slices_per_volume: int) -> List[int]:
    """Select slice indices with highest voxel sums (proxy for information content)."""
    sums = [np.sum(image_array[:, :, i]) for i in range(image_array.shape[2])]
    top_indices = np.argsort(sums)[::-1][:slices_per_volume]
    return top_indices.tolist()


def save_slices(
    nii_path: Path,
    slice_indices: List[int],
    output_dir: Path,
    prefix: str,
    counter_offset: int,
    cast_uint8: bool = False,
):
    image_array = nib.load(str(nii_path)).get_fdata()
    for idx, slice_idx in enumerate(slice_indices):
        data = image_array[:, :, slice_idx]
        filename = f"{prefix}_{counter_offset + idx:05d}.png"
        output_path = output_dir / filename
        if cast_uint8:
            imageio.imwrite(output_path, data.astype("uint8"))
        else:
            imageio.imwrite(output_path, data)


def build_file_lists(dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    modalities = []
    for pattern in ["*t1.nii.gz", "*t1ce.nii.gz", "*t2.nii.gz", "*flair.nii.gz"]:
        modalities.extend(sorted(dataset_root.rglob(pattern)))
    segmentations = sorted(dataset_root.rglob("*seg.nii.gz"))
    # replicate segmentations to align with 4 modalities per case
    segmentations = segmentations * 4
    return modalities, segmentations


def prepare_dataset(
    dataset_root: Path,
    output_root: Path,
    slices_per_volume: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
):
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

    brains, segs = build_file_lists(dataset_root)
    if len(brains) != len(segs):
        raise ValueError("Number of modality volumes does not match segmentation volumes")

    total = len(brains)
    train_split = int(train_ratio * total)
    val_split = int((train_ratio + val_ratio) * total)

    splits = {
        "train": (brains[:train_split], segs[:train_split]),
        "val": (brains[train_split:val_split], segs[train_split:val_split]),
        "test": (brains[val_split:], segs[val_split:]),
    }

    slice_cache: Dict[Path, List[int]] = {}

    for split_name, (brain_paths, seg_paths) in splits.items():
        for idx, (brain_path, seg_path) in enumerate(zip(brain_paths, seg_paths)):
            if seg_path not in slice_cache:
                image_array = nib.load(str(seg_path)).get_fdata()
                slice_cache[seg_path] = select_slice_indices(image_array, slices_per_volume)

            slice_indices = slice_cache[seg_path]
            counter_offset = idx * slices_per_volume

            img_dir_key = f"{split_name}_images"
            mask_dir_key = f"{split_name}_masks"

            save_slices(brain_path, slice_indices, output_paths[img_dir_key], f"{split_name}_frame", counter_offset)
            save_slices(seg_path, slice_indices, output_paths[mask_dir_key], f"{split_name}_mask", counter_offset, cast_uint8=True)

    print(f"Finished preparing dataset at {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BraTS NIfTI volumes to PNG slices")
    parser.add_argument("--dataset-root", required=True, help="Path to raw BraTS dataset root")
    parser.add_argument("--output-root", default="./Dataset", help="Where to store PNG slices")
    parser.add_argument("--slices-per-volume", type=int, default=20, help="Number of slices to export per volume")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    prepare_dataset(dataset_root=dataset_root, output_root=output_root, slices_per_volume=args.slices_per_volume)


if __name__ == "__main__":
    main()
