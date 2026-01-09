import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from src.data.dataset import build_dataloader
from src.models.unet import build_unet
from src.utils.config import apply_overrides, load_config, resolve_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on val/test splits")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--data-root", dest="data_root", help="Override dataset root directory")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--weights", help="Path to model weights .h5 file")
    parser.add_argument(
        "--dataset-format",
        choices=["brats", "msd_task01"],
        default="brats",
        help="Dataset format used to generate slices",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="Label mode used to generate masks",
    )
    return parser.parse_args()


def build_model(cfg) -> tf.keras.Model:
    image_size = cfg["data"]["image_size"]
    input_channels = cfg["model"]["input_channels"]
    class_names = cfg["data"]["class_names"]
    model = build_unet(
        input_size=(image_size, image_size, input_channels),
        num_classes=len(class_names),
        base_filters=cfg["model"].get("base_filters", 32),
        learning_rate=cfg["training"]["learning_rate"],
    )
    return model


def resolve_weights_path(cfg, weights_override: str = None) -> Path:
    if weights_override:
        return Path(weights_override).expanduser().resolve()
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_name = cfg["training"]["checkpoint_filename"]
    return (checkpoint_dir / checkpoint_name).expanduser().resolve()


def accumulate_dice(
    model: tf.keras.Model,
    loader,
    num_classes: int,
) -> Tuple[List[float], float]:
    intersections = np.zeros(num_classes, dtype=np.float64)
    totals = np.zeros(num_classes, dtype=np.float64)
    for images, masks in loader:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=-1)
        true_classes = np.argmax(masks, axis=-1)
        for class_idx in range(num_classes):
            pred_mask = pred_classes == class_idx
            true_mask = true_classes == class_idx
            intersections[class_idx] += np.logical_and(pred_mask, true_mask).sum()
            totals[class_idx] += pred_mask.sum() + true_mask.sum()

    dice_per_class = []
    for class_idx in range(num_classes):
        if totals[class_idx] == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersections[class_idx]) / totals[class_idx]
        dice_per_class.append(float(dice))
    mean_dice = float(np.mean(dice_per_class)) if dice_per_class else 0.0
    return dice_per_class, mean_dice


def evaluate_split(model: tf.keras.Model, cfg, split_name: str) -> Dict[str, float]:
    class_names = cfg["data"]["class_names"]
    images_dir = cfg["data"][f"{split_name}_images"]
    masks_dir = cfg["data"][f"{split_name}_masks"]
    batch_size = cfg["training"]["batch_size"]
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Missing {split_name} data at {images_dir} or {masks_dir}")
    loader = build_dataloader(
        images_dir,
        masks_dir,
        class_names,
        batch_size=batch_size,
        image_size=cfg["data"]["image_size"],
        augmentation=None,
        shuffle=False,
    )
    if len(loader) == 0:
        raise ValueError(f"No samples found for {split_name} split")
    dice_per_class, mean_dice = accumulate_dice(model, loader, len(class_names))
    return {
        "dice_per_class": dict(zip(class_names, dice_per_class)),
        "mean_dice": mean_dice,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    cfg = resolve_paths(cfg)

    weights_path = resolve_weights_path(cfg, args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    model = build_model(cfg)
    model.load_weights(weights_path)

    metrics = {
        "dataset_format": args.dataset_format,
        "label_mode": args.label_mode,
        "val": evaluate_split(model, cfg, "val"),
        "test": evaluate_split(model, cfg, "test"),
    }

    output_dir = Path("outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
