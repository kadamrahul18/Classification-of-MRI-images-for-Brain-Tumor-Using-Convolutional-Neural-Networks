import argparse
import csv
import json
import random
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss, DiceLoss
from monai.networks.nets import UNet
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
        max_pos_attempts=data_cfg.get("max_pos_attempts", 10),
        min_pos_voxel_frac=data_cfg.get("min_pos_voxel_frac", 0.0),
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

    num_workers = cfg["training"]["num_workers"]
    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def _ensure_5d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(0)
    if tensor.ndim == 5:
        return tensor
    if tensor.ndim == 6 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")


def compute_dice(pred: torch.Tensor, target: torch.Tensor, include_background: bool) -> torch.Tensor:
    pred = _ensure_5d(pred, "pred")
    target = _ensure_5d(target, "target")
    if pred.shape[0] != target.shape[0]:
        raise ValueError(f"Batch mismatch: pred {pred.shape} vs target {target.shape}")
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
    return dice


def _validate_labels(labels: torch.Tensor, num_classes: int, is_binary: bool) -> None:
    if labels.ndim != 5:
        raise ValueError(f"Expected labels shape (B,C,D,H,W), got {tuple(labels.shape)}")
    if is_binary:
        if labels.shape[1] not in (1, 2):
            raise ValueError(
                f"Binary labels must have C=1 or C=2, got C={labels.shape[1]}"
            )
    elif labels.shape[1] != num_classes:
        raise ValueError(
            f"Label channel mismatch: expected C={num_classes}, got C={labels.shape[1]}"
        )
    if labels.min().item() < -1e-3 or labels.max().item() > 1 + 1e-3:
        raise ValueError("Labels must be in [0,1] for one-hot encoding")
    if not is_binary or labels.shape[1] == 2:
        channel_sum = labels.sum(dim=1)
        if not torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-3):
            raise ValueError("Labels must be one-hot encoded (sum across channels = 1).")


def _pred_to_onehot(
    logits: torch.Tensor,
    num_classes: int,
    is_binary: bool,
    threshold: float = 0.5,
) -> torch.Tensor:
    if is_binary:
        logit_tumor = logits[:, 1:2] if logits.shape[1] >= 2 else logits
        probs = torch.sigmoid(logit_tumor)
        pred = (probs > threshold).float()
        if pred.shape[1] == 1 and num_classes == 2:
            pred = torch.cat([1.0 - pred, pred], dim=1)
        return pred
    pred_labels = torch.argmax(logits, dim=1, keepdim=True)
    return torch.nn.functional.one_hot(
        pred_labels.squeeze(1), num_classes=num_classes
    ).permute(0, 4, 1, 2, 3).float()


def _foreground_mask(labels: torch.Tensor) -> torch.Tensor:
    if labels.shape[1] == 1:
        return labels[:, 0] > 0.5
    return labels[:, 1:].sum(dim=1) > 0


def _log_sanity_stats(
    logger: logging.Logger,
    preds: torch.Tensor,
    labels: torch.Tensor,
    prefix: str,
) -> None:
    foreground_gt = _foreground_mask(labels)
    foreground_pred = _foreground_mask(preds)
    tumor_voxel_frac = foreground_gt.float().mean().item()
    pred_tumor_frac = foreground_pred.float().mean().item()
    tp = (foreground_pred & foreground_gt).sum().item()
    fp = (foreground_pred & ~foreground_gt).sum().item()
    fn = (~foreground_pred & foreground_gt).sum().item()
    logger.info(
        "%s tumor_voxel_frac=%.6f pred_tumor_voxel_frac=%.6f tp=%s fp=%s fn=%s",
        prefix,
        tumor_voxel_frac,
        pred_tumor_frac,
        int(tp),
        int(fp),
        int(fn),
    )


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

    (run_dir / "git_commit.txt").write_text(
        subprocess.getoutput("git rev-parse --short HEAD").strip() or "unknown",
        encoding="utf-8",
    )
    env_lines = [sys.version.split()[0]]
    for pkg in ("torch", "monai"):
        try:
            mod = __import__(pkg)
            env_lines.append(f"{pkg}=={getattr(mod, '__version__', 'unknown')}")
        except Exception:
            env_lines.append(f"{pkg} not installed")
    (run_dir / "env.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    gpu_info = subprocess.getoutput("nvidia-smi -L") if torch.cuda.is_available() else "cpu"
    (run_dir / "gpu.txt").write_text(gpu_info + "\n", encoding="utf-8")

    with (run_dir / "train_config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    set_seeds(cfg["training"].get("seed", 42), cfg["training"].get("deterministic", False))

    data_cfg = cfg["data"]
    num_classes = len(data_cfg.get("class_names", ["background", "tumor"]))
    if data_cfg["label_mode"] == "binary":
        num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes).to(device)

    is_binary = data_cfg["label_mode"] == "binary"
    if is_binary:
        pos_weight_value = float(cfg["training"].get("pos_weight", 5.0))
        pos_weight = torch.tensor([pos_weight_value], device=device)
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        dice_loss = DiceLoss(sigmoid=True, include_background=False)

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            if labels.shape[1] == 2:
                target = labels[:, 1:2]
            elif labels.shape[1] == 1:
                target = labels
            else:
                raise ValueError(
                    f"Expected binary labels with C=1 or C=2, got shape {tuple(labels.shape)}"
                )
            logit_tumor = logits[:, 1:2] if logits.shape[1] >= 2 else logits
            return dice_loss(logit_tumor, target) + bce_loss(logit_tumor, target)

    else:
        loss_fn = DiceCELoss(
            to_onehot_y=False,
            softmax=True,
            include_background=False,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    train_loader, val_loader = build_dataloaders(cfg)
    log_interval = cfg["training"].get("log_interval", 0)
    vis_interval = cfg["training"].get("vis_interval", 5)
    max_vis_cases = cfg["training"].get("max_vis_cases", 3)
    ignore_empty_foreground = cfg["training"].get("ignore_empty_foreground", True)
    log_sanity_steps = int(cfg["training"].get("log_sanity_steps", 50))
    logger.info("Train batches per epoch: %s | Val batches: %s", len(train_loader), len(val_loader))
    logger.info("Using device: %s", device)
    logger.info("ROI size: %s", data_cfg.get("roi_size"))
    logger.info("num_workers: %s", cfg["training"]["num_workers"])
    logger.info(
        "limit_train_batches: %s | limit_val_batches: %s",
        cfg["training"].get("limit_train_batches"),
        cfg["training"].get("limit_val_batches"),
    )
    logger.info(
        "ignore_empty_foreground: %s | pred_rule: %s",
        ignore_empty_foreground,
        f"sigmoid>{cfg['training'].get('prediction_threshold', 0.5)}" if is_binary else "argmax over softmax logits",
    )
    if torch.cuda.is_available():
        logger.info("GPU name: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA capability: %s", torch.cuda.get_device_capability(0))
    logger.info("AMP enabled: %s", torch.cuda.is_available())
    logger.info("prediction_threshold: %s", cfg["training"].get("prediction_threshold", 0.5))
    logger.info(
        "pred_rule: %s",
        f"sigmoid>{cfg['training'].get('prediction_threshold', 0.5)}" if is_binary else "argmax over softmax logits",
    )
    if "OMP_NUM_THREADS" not in os.environ or "MKL_NUM_THREADS" not in os.environ:
        logger.warning(
            "For best throughput set: OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1"
        )

    best_fg_dice = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    metrics_path = run_dir / "metrics.csv"
    tb_writer = SummaryWriter(log_dir=run_dir)
    metrics_json_path = run_dir / "metrics_per_epoch.json"
    metrics_history = []
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_mean_dice",
                "val_foreground_dice",
                "val_dice_background",
                "val_dice_tumor",
                "lr",
            ]
        )

        global_step = 0
        for epoch in range(cfg["training"]["max_epochs"]):
            logger.info("Epoch %s/%s", epoch + 1, cfg["training"]["max_epochs"])
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model.train()
            train_loss = 0.0
            batch_count = 0
            step_time = 0.0
            step_window = 0
            pos_batches = 0
            total_batches = 0
            pos_tumor_fracs: List[float] = []
            logged_device = False
            for batch_idx, (images, labels) in enumerate(train_loader):
                if cfg["training"].get("limit_train_batches") and batch_idx >= cfg["training"]["limit_train_batches"]:
                    break
                start = time.perf_counter()
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if not logged_device:
                    logger.info("Model device: %s | Batch device: %s", next(model.parameters()).device, images.device)
                    logged_device = True
                _validate_labels(labels, num_classes, is_binary)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                batch_count += 1
                total_batches += 1
                preds = _pred_to_onehot(logits, num_classes, is_binary, cfg["training"].get("prediction_threshold", 0.5))
                if _foreground_mask(labels).any():
                    pos_batches += 1
                    pos_tumor_fracs.append(_foreground_mask(labels).float().mean().item())
                step_time += time.perf_counter() - start
                step_window += 1
                if global_step < log_sanity_steps or batch_idx == 0:
                    _log_sanity_stats(logger, preds, labels, "train")
                if log_interval and batch_idx % log_interval == 0:
                    avg_step = step_time / max(1, step_window)
                    logger.info(
                        "  train step %s/%s loss=%.4f avg_step=%.3fs",
                        batch_idx + 1,
                        len(train_loader),
                        loss.item(),
                        avg_step,
                    )
                    step_time = 0.0
                    step_window = 0
                global_step += 1

            train_loss /= max(1, batch_count)
            observed_pos_ratio = pos_batches / max(1, total_batches)
            logger.info("Observed train pos_ratio (patches w/ tumor): %.3f", observed_pos_ratio)
            tb_writer.add_scalar("data/pos_ratio", observed_pos_ratio, epoch + 1)
            if pos_tumor_fracs:
                median_pos_frac = float(np.median(pos_tumor_fracs))
                logger.info("Median tumor_voxel_frac (pos patches): %.6f", median_pos_frac)
                tb_writer.add_scalar("data/median_tumor_frac_pos", median_pos_frac, epoch + 1)

            model.eval()
            dice_scores: List[float] = []
            dice_per_class: List[np.ndarray] = []
            dice_sum = np.zeros(num_classes, dtype=np.float64)
            dice_count = np.zeros(num_classes, dtype=np.float64)
            val_loss = 0.0
            val_batches = 0
            debug_shapes = cfg["training"].get("debug_shapes", False)
            with torch.no_grad():
                for val_idx, (images, labels) in enumerate(val_loader):
                    if cfg["training"].get("limit_val_batches") and val_idx >= cfg["training"]["limit_val_batches"]:
                        break
                    images = images.to(device)
                    labels = labels.to(device)
                    _validate_labels(labels, num_classes, is_binary)
                    logits = model(images)
                    val_loss += loss_fn(logits, labels).item()
                    val_batches += 1
                    preds = _pred_to_onehot(logits, num_classes, is_binary, cfg["training"].get("prediction_threshold", 0.5))
                    if debug_shapes and val_idx == 0:
                        logger.info(
                            "val shapes | images=%s labels=%s logits=%s preds=%s",
                            tuple(images.shape),
                            tuple(labels.shape),
                            tuple(logits.shape),
                            tuple(preds.shape),
                        )
                    dice = compute_dice(preds, labels, include_background=True)
                    dice_scores.append(dice.mean().item())
                    dice_per_class.append(dice.mean(dim=0).cpu().numpy())
                    target_sum = labels.sum(dim=(2, 3, 4))
                    for cls_idx in range(num_classes):
                        if cls_idx == 0:
                            valid = np.ones(target_sum.shape[0], dtype=bool)
                        else:
                            valid = target_sum[:, cls_idx].cpu().numpy() > 0
                            if not ignore_empty_foreground:
                                valid = np.ones_like(valid, dtype=bool)
                        dice_vals = dice[:, cls_idx].detach().cpu().numpy()
                        if valid.any():
                            dice_sum[cls_idx] += dice_vals[valid].sum()
                            dice_count[cls_idx] += valid.sum()
                    if log_interval and val_idx % log_interval == 0:
                        logger.info(
                            "  val step %s/%s mean_dice=%.4f",
                            val_idx + 1,
                            len(val_loader),
                            dice.mean().item(),
                        )

            val_mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
            val_loss = val_loss / max(1, val_batches)
            lr = optimizer.param_groups[0]["lr"]
            val_dice_per_class = np.where(
                dice_count > 0, dice_sum / np.maximum(dice_count, 1.0), 0.0
            )
            val_dice_background = float(val_dice_per_class[0]) if num_classes > 0 else 0.0
            val_dice_tumor = float(val_dice_per_class[1]) if num_classes > 1 else 0.0
            if num_classes > 1:
                val_foreground_dice = float(np.mean(val_dice_per_class[1:]))
            else:
                val_foreground_dice = float(val_dice_per_class[0]) if num_classes > 0 else 0.0
            csv_writer.writerow(
                [
                    epoch + 1,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_mean_dice:.6f}",
                    f"{val_foreground_dice:.6f}",
                    f"{val_dice_background:.6f}",
                    f"{val_dice_tumor:.6f}",
                    f"{lr:.8f}",
                ]
            )
            f.flush()

            tb_writer.add_scalar("loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("loss/val", val_loss, epoch + 1)
            tb_writer.add_scalar("dice_mean/val", val_mean_dice, epoch + 1)
            tb_writer.add_scalar("dice_foreground/val", val_foreground_dice, epoch + 1)
            if dice_per_class:
                for idx in range(num_classes):
                    tb_writer.add_scalar(
                        f"dice/val_class_{idx}",
                        float(val_dice_per_class[idx]) if idx < len(val_dice_per_class) else 0.0,
                        epoch + 1,
                    )
            if num_classes > 1:
                tb_writer.add_scalar("dice/val_background", val_dice_background, epoch + 1)
                tb_writer.add_scalar("dice/val_tumor", val_dice_tumor, epoch + 1)
            tb_writer.add_scalar("lr", lr, epoch + 1)
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
                tb_writer.add_scalar("gpu_mem_max_mb", gpu_mem, epoch + 1)

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                gpu_mem = 0.0
            logger.info(
                "Epoch %s summary | train_loss=%.4f val_loss=%.4f val_mean_dice=%.4f val_foreground_dice=%.4f val_dice_background=%.4f val_dice_tumor=%.4f lr=%.6f gpu_mem_max_mb=%.1f",
                epoch + 1,
                train_loss,
                val_loss,
                val_mean_dice,
                val_foreground_dice,
                val_dice_background,
                val_dice_tumor,
                lr,
                gpu_mem,
            )

            metrics_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_mean_dice": float(val_mean_dice),
                    "val_foreground_dice": float(val_foreground_dice),
                    "val_dice_per_class": val_dice_per_class.tolist(),
                    "val_dice_background": float(val_dice_background),
                    "val_dice_tumor": float(val_dice_tumor),
                    "lr": float(lr),
                    "ignore_empty_foreground": bool(ignore_empty_foreground),
                    "include_background": True,
                    "pred_rule": "argmax over softmax logits",
                    "class_names": data_cfg.get("class_names", []),
                }
            )
            with metrics_json_path.open("w", encoding="utf-8") as json_f:
                json.dump(metrics_history, json_f, indent=2)

            improved = False
            if val_foreground_dice > best_fg_dice:
                improved = True
            elif abs(val_foreground_dice - best_fg_dice) < 1e-6 and val_loss < best_val_loss:
                improved = True
            if improved:
                best_fg_dice = val_foreground_dice
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), run_dir / "best.pt")
                logger.info(
                    "New best model | val_tumor_dice=%.4f val_foreground_dice=%.4f epoch=%s",
                    val_dice_tumor,
                    val_foreground_dice,
                    best_epoch,
                )

            if (epoch + 1) % vis_interval == 0:
                _log_visuals(
                    model,
                    val_loader,
                    device,
                    tb_writer,
                    run_dir,
                    epoch + 1,
                    num_classes,
                    max_vis_cases,
                )

    logger.info("Best val tumor/foreground Dice: %.4f at epoch %s", best_fg_dice, best_epoch)
    logger.info("Run directory: %s", run_dir)
    tb_writer.close()


if __name__ == "__main__":
    main()
