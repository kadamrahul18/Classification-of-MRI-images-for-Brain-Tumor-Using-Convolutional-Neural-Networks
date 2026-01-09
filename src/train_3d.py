import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete

from src.data.msd_task01_3d import MSDTask01Dataset3D, build_splits, list_msd_task01_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net on MSD Task01")
    parser.add_argument("--config", default="configs/config_3d.yaml", help="Path to YAML config file")
    parser.add_argument("--max-epochs", type=int, help="Override max epochs")
    parser.add_argument("--limit-train-batches", type=int, help="Limit train batches per epoch")
    parser.add_argument("--limit-val-batches", type=int, help="Limit val batches per epoch")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    return parser.parse_args()


def set_seeds(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(cfg: Dict, args) -> Dict:
    cfg = json.loads(json.dumps(cfg))
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.limit_train_batches is not None:
        cfg["training"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        cfg["training"]["limit_val_batches"] = args.limit_val_batches
    if args.deterministic:
        cfg["training"]["deterministic"] = True
    return cfg


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


def build_dataloaders(cfg: Dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    data_cfg = cfg["data"]
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
    roi_size = data_cfg.get("roi_size")
    percentiles = tuple(data_cfg.get("percentiles", [0.5, 99.5]))

    train_dataset = MSDTask01Dataset3D(
        splits["train"],
        roi_size=roi_size,
        label_mode=data_cfg["label_mode"],
        num_classes=num_classes,
        pos_ratio=data_cfg.get("pos_ratio", 0.5),
        percentiles=percentiles,
        mode="train",
        seed=data_cfg.get("seed", 42),
    )
    val_dataset = MSDTask01Dataset3D(
        splits["val"],
        roi_size=roi_size,
        label_mode=data_cfg["label_mode"],
        num_classes=num_classes,
        pos_ratio=0.0,
        percentiles=percentiles,
        mode="val",
        seed=data_cfg.get("seed", 42),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    return train_loader, val_loader


def compute_dice(pred: torch.Tensor, target: torch.Tensor, include_background: bool) -> torch.Tensor:
    dice_metric = DiceMetric(include_background=include_background, reduction="none")
    dice_metric(y_pred=pred, y=target)
    return dice_metric.aggregate()


def main():
    args = parse_args()
    cfg = resolve_config(load_config(args.config), args)
    run_dir = Path(cfg["training"]["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "train_config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    set_seeds(cfg["training"].get("seed", 42), cfg["training"].get("deterministic", False))

    data_cfg = cfg["data"]
    num_classes = len(data_cfg.get("class_names", ["background", "tumor"]))
    if data_cfg["label_mode"] == "binary":
        num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes).to(device)

    loss_fn = DiceCELoss(
        to_onehot_y=False,
        softmax=True,
        include_background=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    train_loader, val_loader = build_dataloaders(cfg)

    best_dice = -1.0
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_mean_dice"])

        for epoch in range(cfg["training"]["max_epochs"]):
            model.train()
            train_loss = 0.0
            batch_count = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                if cfg["training"].get("limit_train_batches") and batch_idx >= cfg["training"]["limit_train_batches"]:
                    break
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                batch_count += 1

            train_loss /= max(1, batch_count)

            model.eval()
            dice_scores: List[float] = []
            with torch.no_grad():
                for val_idx, (images, labels) in enumerate(val_loader):
                    if cfg["training"].get("limit_val_batches") and val_idx >= cfg["training"]["limit_val_batches"]:
                        break
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    preds = post_pred(torch.softmax(logits, dim=1))
                    dice = compute_dice(preds, labels, include_background=True)
                    dice_scores.append(dice.mean().item())

            val_mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_mean_dice:.6f}"])
            f.flush()

            if val_mean_dice > best_dice:
                best_dice = val_mean_dice
                torch.save(model.state_dict(), run_dir / "best.pt")

    print(f"Best val mean Dice: {best_dice:.4f}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
