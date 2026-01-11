import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from src.data.msd_task01_3d import MSDTask01Dataset3D, build_splits, list_msd_task01_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net on MSD Task01")
    parser.add_argument("--config", default="configs/config_3d.yaml", help="Path to YAML config file")
    parser.add_argument("--weights", required=True, help="Path to model weights .pt")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict, num_classes: int) -> UNet:
    model_cfg = cfg["model"]
    return UNet(
        spatial_dims=3,
        in_channels=model_cfg["in_channels"],
        out_channels=num_classes,
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        num_res_units=model_cfg["num_res_units"],
        norm=model_cfg["norm"],
    )


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _ensure_5d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(0)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")


def _compute_dice(pred: torch.Tensor, target: torch.Tensor, include_background: bool) -> torch.Tensor:
    pred = _ensure_5d(pred, "pred")
    target = _ensure_5d(target, "target")
    if target.shape[1] != pred.shape[1]:
        target = torch.argmax(target, dim=1, keepdim=True)
        target = torch.nn.functional.one_hot(
            target.long().squeeze(1), num_classes=pred.shape[1]
        ).permute(0, 4, 1, 2, 3).float()
    target = target.to(pred.device)
    pred = pred.float()
    target = target.float()
    if not include_background and pred.shape[1] > 1:
        pred = pred[:, 1:]
        target = target[:, 1:]
    intersection = (pred * target).sum(dim=(2, 3, 4))
    denom = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = torch.where(denom > 0, (2.0 * intersection) / denom, torch.ones_like(denom))
    return dice.mean(dim=0)


def evaluate_split(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    overlap: float,
    sw_batch_size: int,
    num_classes: int,
) -> Tuple[List[float], int]:
    model.eval()

    dice_scores = []
    with torch.no_grad():
        for image, label in dataset:
            image = image.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            logits = sliding_window_inference(
                image, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap
            )
            pred_labels = torch.argmax(logits, dim=1, keepdim=True)
            pred = torch.nn.functional.one_hot(
                pred_labels.squeeze(1), num_classes=num_classes
            ).permute(0, 4, 1, 2, 3).float()
            dice = _compute_dice(pred, label, include_background=True)
            dice_scores.append(dice.cpu().numpy())

    if not dice_scores:
        return [0.0] * num_classes, 0
    dice_scores = np.stack(dice_scores, axis=0)
    return dice_scores.mean(axis=0).tolist(), len(dice_scores)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    inference_cfg = cfg["inference"]

    cases = list_msd_task01_cases(Path(data_cfg["root"]))
    splits = build_splits(
        cases,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg.get("seed", 42),
        list_files=data_cfg.get("list_files"),
    )

    num_classes = len(data_cfg.get("class_names", ["background", "tumor"]))
    if data_cfg["label_mode"] == "binary":
        num_classes = 2

    percentiles = tuple(data_cfg.get("percentiles", [0.5, 99.5]))
    val_dataset = MSDTask01Dataset3D(
        splits["val"],
        roi_size=None,
        label_mode=data_cfg["label_mode"],
        num_classes=num_classes,
        pos_ratio=0.0,
        percentiles=percentiles,
        mode="val",
        seed=data_cfg.get("seed", 42),
    )
    test_dataset = MSDTask01Dataset3D(
        splits["test"],
        roi_size=None,
        label_mode=data_cfg["label_mode"],
        num_classes=num_classes,
        pos_ratio=0.0,
        percentiles=percentiles,
        mode="test",
        seed=data_cfg.get("seed", 42),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    roi_size = tuple(inference_cfg["roi_size"])
    overlap = inference_cfg["overlap"]
    sw_batch_size = inference_cfg["sw_batch_size"]

    val_dice, val_count = evaluate_split(
        model, val_dataset, device, roi_size, overlap, sw_batch_size, num_classes
    )
    test_dice, test_count = evaluate_split(
        model, test_dataset, device, roi_size, overlap, sw_batch_size, num_classes
    )

    class_names = data_cfg.get("class_names", [f"class_{i}" for i in range(num_classes)])
    if len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    metrics = {
        "dataset_format": "msd_task01",
        "label_mode": data_cfg["label_mode"],
        "val": {
            "dice_per_class": dict(zip(class_names, [float(x) for x in val_dice])),
            "mean_dice": float(np.mean(val_dice)) if val_dice else 0.0,
            "number_of_volumes": val_count,
        },
        "test": {
            "dice_per_class": dict(zip(class_names, [float(x) for x in test_dice])),
            "mean_dice": float(np.mean(test_dice)) if test_dice else 0.0,
            "number_of_volumes": test_count,
        },
        "inference": {
            "roi_size": list(roi_size),
            "overlap": overlap,
            "sw_batch_size": sw_batch_size,
        },
    }

    git_hash = get_git_hash()
    if git_hash:
        metrics["git_commit"] = git_hash

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics_3d.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
