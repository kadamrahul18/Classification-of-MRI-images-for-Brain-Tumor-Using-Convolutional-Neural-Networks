import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.msd_task01_3d import MSDTask01Dataset3D, list_msd_task01_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Debug ROI alignment on a single patch")
    parser.add_argument(
        "--dataset-root",
        default="./data/raw/msd_task01/Task01_BrainTumour",
        help="MSD Task01 root with imagesTr/labelsTr",
    )
    parser.add_argument("--roi-size", default="96,96,96", help="ROI size as D,H,W")
    parser.add_argument("--case-index", type=int, default=0, help="Case index from imagesTr")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pos-ratio", type=float, default=1.0, help="Probability of tumor-centered sampling")
    parser.add_argument("--max-pos-attempts", type=int, default=10, help="Max resamples for tumor patches")
    parser.add_argument("--out-dir", default="outputs/debug_alignment", help="Output folder")
    parser.add_argument("--channel", type=int, default=0, help="Which modality to visualize")
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
    roi_size = tuple(int(x) for x in args.roi_size.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = list_msd_task01_cases(Path(args.dataset_root))
    case = cases[args.case_index]
    ds = MSDTask01Dataset3D(
        [case],
        roi_size=roi_size,
        label_mode="binary",
        num_classes=2,
        pos_ratio=args.pos_ratio,
        percentiles=(0.5, 99.5),
        mode="train",
        seed=args.seed,
        max_pos_attempts=args.max_pos_attempts,
    )

    img, lbl = ds[0]
    print("image shape", tuple(img.shape), "label shape", tuple(lbl.shape))
    print("label unique", np.unique(lbl.numpy()))
    print("label sum", float(lbl.sum().item()))

    gt = lbl[1].numpy()
    areas = gt.sum(axis=(1, 2))
    z = int(np.argmax(areas))
    img_slice = img[args.channel, z].numpy()
    gt_slice = gt[z]

    img_slice = _normalize_slice(img_slice)
    gt_slice = (gt_slice > 0).astype(np.uint8) * 255

    overlay = np.stack([img_slice] * 3, axis=-1).astype(np.uint8)
    overlay[gt_slice > 0, 1] = 255

    Image.fromarray(img_slice).save(out_dir / "input.png")
    Image.fromarray(gt_slice).save(out_dir / "gt.png")
    Image.fromarray(overlay).save(out_dir / "overlay.png")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
