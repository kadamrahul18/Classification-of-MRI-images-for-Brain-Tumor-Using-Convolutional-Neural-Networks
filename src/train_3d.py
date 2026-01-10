import argparse
import csv
import json
import random
import logging
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
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from src.data.msd_task01_3d import MSDTask01Dataset3D, build_splits, list_msd_task01_cases


def setup_logging(log_path: Path):
    logger = logging.getLogger("train_3d")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger


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


def _normalize_slice(slice_array: np.ndarray) -> np.ndarray:
    vmin = float(slice_array.min())
    vmax = float(slice_array.max())
    if vmax <= vmin:
        return np.zeros_like(slice_array, dtype=np.uint8)
    scaled = (slice_array - vmin) / (vmax - vmin)
    return (scaled * 255.0).astype(np.uint8)


def _select_slice_index(label_one_hot: np.ndarray) -> int:
    if label_one_hot.shape[0] > 1:
        foreground = label_one_hot[1:].sum(axis=0)
    else:
        foreground = label_one_hot[0]
    areas = foreground.sum(axis=(1, 2))
    if areas.max() == 0:
        return label_one_hot.shape[1] // 2
    return int(np.argmax(areas))


def _make_overlay(input_slice: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    base = np.stack([input_slice] * 3, axis=-1).astype(np.float32)
    overlay = base.copy()
    overlay[gt_mask > 0, 1] = 255
    overlay[pred_mask > 0, 0] = 255
    return overlay.astype(np.uint8)


def _save_vis_images(vis_dir: Path, case_idx: int, input_slice, gt_slice, pred_slice, overlay):
    Image.fromarray(input_slice).save(vis_dir / f"case_{case_idx}_input.png")
    Image.fromarray(gt_slice).save(vis_dir / f"case_{case_idx}_gt.png")
    Image.fromarray(pred_slice).save(vis_dir / f"case_{case_idx}_pred.png")
    Image.fromarray(overlay).save(vis_dir / f"case_{case_idx}_overlay.png")


def _log_visuals(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    run_dir: Path,
    epoch: int,
    num_classes: int,
    max_cases: int,
):
    vis_dir = run_dir / "vis" / f"epoch_{epoch:02d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    overlays = []
    model.eval()
    case_idx = 0
    with torch.no_grad():
        for images, labels in val_loader:
            for b in range(images.shape[0]):
                if case_idx >= max_cases:
                    break
                image = images[b : b + 1].to(device)
                label = labels[b].cpu().numpy()
                logits = model(image)
                pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()[0]

                slice_idx = _select_slice_index(label)
                input_slice = images[b, 0, slice_idx].cpu().numpy()
                input_slice = _normalize_slice(input_slice)

                gt_slice = np.argmax(label[:, slice_idx], axis=0).astype(np.uint8)
                pred_slice = pred[slice_idx].astype(np.uint8)

                if num_classes > 1:
                    scale = 255 // max(1, num_classes - 1)
                else:
                    scale = 255
                pred_viz = (pred_slice * scale).astype(np.uint8)
                gt_viz = (gt_slice * scale).astype(np.uint8)

                overlay = _make_overlay(input_slice, gt_slice, pred_slice)
                _save_vis_images(vis_dir, case_idx, input_slice, gt_viz, pred_viz, overlay)
                overlays.append(overlay)
                case_idx += 1
            if case_idx >= max_cases:
                break

    if overlays:
        grid = np.concatenate(overlays, axis=1)
        writer.add_image("vis/overlay", grid, epoch, dataformats="HWC")


def main():
    args = parse_args()
    cfg = resolve_config(load_config(args.config), args)
    run_dir = Path(cfg["training"]["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir / "train.log")

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
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    train_loader, val_loader = build_dataloaders(cfg)
    log_interval = cfg["training"].get("log_interval", 10)
    vis_interval = cfg["training"].get("vis_interval", 5)
    max_vis_cases = cfg["training"].get("max_vis_cases", 3)
    logger.info("Train batches per epoch: %s | Val batches: %s", len(train_loader), len(val_loader))
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU name: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA capability: %s", torch.cuda.get_device_capability(0))

    best_dice = -1.0
    metrics_path = run_dir / "metrics.csv"
    tb_writer = SummaryWriter(log_dir=run_dir)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "val_mean_dice"])

        for epoch in range(cfg["training"]["max_epochs"]):
            logger.info("Epoch %s/%s", epoch + 1, cfg["training"]["max_epochs"])
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model.train()
            train_loss = 0.0
            batch_count = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                if cfg["training"].get("limit_train_batches") and batch_idx >= cfg["training"]["limit_train_batches"]:
                    break
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                batch_count += 1
                if batch_idx % log_interval == 0:
                    logger.info(
                        "  train step %s/%s loss=%.4f",
                        batch_idx + 1,
                        len(train_loader),
                        loss.item(),
                    )

            train_loss /= max(1, batch_count)

            model.eval()
            dice_scores: List[float] = []
            dice_per_class: List[np.ndarray] = []
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_idx, (images, labels) in enumerate(val_loader):
                    if cfg["training"].get("limit_val_batches") and val_idx >= cfg["training"]["limit_val_batches"]:
                        break
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    val_loss += loss_fn(logits, labels).item()
                    val_batches += 1
                    preds = post_pred(torch.softmax(logits, dim=1))
                    dice = compute_dice(preds, labels, include_background=True)
                    dice_scores.append(dice.mean().item())
                    dice_per_class.append(dice.mean(dim=0).cpu().numpy())
                    if val_idx % log_interval == 0:
                        logger.info(
                            "  val step %s/%s mean_dice=%.4f",
                            val_idx + 1,
                            len(val_loader),
                            dice.mean().item(),
                        )

            val_mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
            val_loss = val_loss / max(1, val_batches)
            csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_mean_dice:.6f}"])
            f.flush()

            tb_writer.add_scalar("loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("loss/val", val_loss, epoch + 1)
            tb_writer.add_scalar("dice_mean/val", val_mean_dice, epoch + 1)
            if dice_per_class and num_classes > 2:
                per_class = np.mean(np.stack(dice_per_class, axis=0), axis=0)
                for idx, score in enumerate(per_class):
                    tb_writer.add_scalar(f"dice/val_class_{idx}", float(score), epoch + 1)
            lr = optimizer.param_groups[0]["lr"]
            tb_writer.add_scalar("lr", lr, epoch + 1)
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
                tb_writer.add_scalar("gpu_mem_max_mb", gpu_mem, epoch + 1)

            if val_mean_dice > best_dice:
                best_dice = val_mean_dice
                torch.save(model.state_dict(), run_dir / "best.pt")

            if (epoch + 1) % vis_interval == 0:
                _log_visuals(
                    model,
                    val_loader,
                    device,
                    tb_writer,
                    run_dir,
                    epoch + 1,
                    num_classes,
                    max_vis_cases=max_vis_cases,
                )

    logger.info("Best val mean Dice: %.4f", best_dice)
    logger.info("Run directory: %s", run_dir)
    tb_writer.close()


if __name__ == "__main__":
    main()
