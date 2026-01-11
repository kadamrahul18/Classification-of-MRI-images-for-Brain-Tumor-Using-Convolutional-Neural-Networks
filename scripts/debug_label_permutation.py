import argparse
from itertools import permutations
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from src.data.msd_task01_3d import _to_channel_first, list_msd_task01_cases, normalize_modalities


def parse_args():
    parser = argparse.ArgumentParser(description="Find label axis permutation that best aligns with images")
    parser.add_argument(
        "--dataset-root",
        default="./data/raw/msd_task01/Task01_BrainTumour",
        help="MSD Task01 root with imagesTr/labelsTr",
    )
    parser.add_argument("--case-index", type=int, default=0, help="Case index from imagesTr")
    parser.add_argument("--channel", type=int, default=0, help="Which modality to visualize")
    parser.add_argument("--out-dir", default="outputs/debug_alignment_perm", help="Output folder")
    return parser.parse_args()


def _normalize_slice(slice_array: np.ndarray) -> np.ndarray:
    vmin = float(slice_array.min())
    vmax = float(slice_array.max())
    if vmax <= vmin:
        return np.zeros_like(slice_array, dtype=np.uint8)
    scaled = (slice_array - vmin) / (vmax - vmin)
    return (scaled * 255.0).astype(np.uint8)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = list_msd_task01_cases(Path(args.dataset_root))
    image_path, label_path = cases[args.case_index]
    image = nib.load(str(image_path)).get_fdata().astype(np.float32)
    label = nib.load(str(label_path)).get_fdata().astype(np.int64)

    image = _to_channel_first(image)
    image = normalize_modalities(image, (0.5, 99.5))
    brain = image[args.channel]
    mask = brain != 0
    if mask.any():
        thresh = np.percentile(brain[mask], 50)
        brain_mask = brain > thresh
    else:
        brain_mask = brain > 0

    perms = list(permutations((0, 1, 2)))
    scores = []
    for perm in perms:
        lbl_perm = np.transpose(label, perm)
        if lbl_perm.shape != brain.shape:
            scores.append((perm, -1))
            continue
        overlap = (lbl_perm > 0) & brain_mask
        scores.append((perm, int(overlap.sum())))

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    print("perm overlap scores:", scores_sorted)
    best_perm, best_score = scores_sorted[0]
    if best_score <= 0:
        print("No positive overlap found.")
        return

    lbl_best = np.transpose(label, best_perm)
    areas = lbl_best.sum(axis=(1, 2))
    z = int(np.argmax(areas))
    img_slice = brain[z]
    gt_slice = lbl_best[z]

    img_slice = _normalize_slice(img_slice)
    gt_slice = (gt_slice > 0).astype(np.uint8) * 255

    overlay = np.stack([img_slice] * 3, axis=-1).astype(np.uint8)
    overlay[gt_slice > 0, 1] = 255

    Image.fromarray(img_slice).save(out_dir / "input.png")
    Image.fromarray(gt_slice).save(out_dir / "gt.png")
    Image.fromarray(overlay).save(out_dir / "overlay.png")
    print(f"best_perm={best_perm} wrote {out_dir}")


if __name__ == "__main__":
    main()
