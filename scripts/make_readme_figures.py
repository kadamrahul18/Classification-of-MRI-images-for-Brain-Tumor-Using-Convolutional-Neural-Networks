import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Export baseline figures for README")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id>")
    parser.add_argument("--max-cases", type=int, default=3, help="Max cases for grid")
    parser.add_argument("--out-dir", default="docs/assets", help="Output directory for figures")
    return parser.parse_args()


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _overlay(mask: Image.Image, base: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    base_arr = np.array(base).astype(np.uint8)
    mask_arr = np.array(mask.convert("L"))
    overlay = base_arr.copy()
    overlay[mask_arr > 0] = color
    return Image.fromarray(overlay)


def _select_latest_vis_dir(run_dir: Path) -> Path:
    vis_root = run_dir / "vis"
    if not vis_root.exists():
        raise FileNotFoundError(f"No vis folder found in {run_dir}")
    epochs = sorted(vis_root.glob("epoch_*"))
    if not epochs:
        raise FileNotFoundError(f"No epoch folders found in {vis_root}")
    return epochs[-1]


def _collect_cases(vis_dir: Path, max_cases: int) -> List[int]:
    cases = []
    for path in vis_dir.glob("case_*_input.png"):
        name = path.stem
        case_id = int(name.split("_")[1])
        cases.append(case_id)
    return sorted(cases)[:max_cases]


def _make_grid(run_dir: Path, out_dir: Path, max_cases: int):
    vis_dir = _select_latest_vis_dir(run_dir)
    case_ids = _collect_cases(vis_dir, max_cases)
    if not case_ids:
        raise FileNotFoundError(f"No case images found in {vis_dir}")

    rows = []
    for case_id in case_ids:
        input_img = _load_image(vis_dir / f"case_{case_id}_input.png")
        gt_mask = Image.open(vis_dir / f"case_{case_id}_gt.png")
        pred_mask = Image.open(vis_dir / f"case_{case_id}_pred.png")
        gt_overlay = _overlay(gt_mask, input_img, (0, 255, 0))
        pred_overlay = _overlay(pred_mask, input_img, (255, 0, 0))
        row = Image.new("RGB", (input_img.width * 3, input_img.height))
        row.paste(input_img, (0, 0))
        row.paste(gt_overlay, (input_img.width, 0))
        row.paste(pred_overlay, (input_img.width * 2, 0))
        rows.append(row)

    grid = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)))
    for idx, row in enumerate(rows):
        grid.paste(row, (0, idx * row.height))

    grid.save(out_dir / "baseline_examples.png")


def _make_curves(run_dir: Path, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in {run_dir}")

    epochs = []
    train_loss = []
    val_loss = []
    val_dice = []
    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row.get("train_loss", 0)))
            val_loss.append(float(row.get("val_loss", 0)))
            val_dice.append(float(row.get("val_dice_tumor", row.get("val_foreground_dice", 0))))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_dice, color="green", label="val_dice_tumor")
    ax2.set_ylabel("Dice")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "baseline_curves.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _make_grid(run_dir, out_dir, args.max_cases)
    _make_curves(run_dir, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
