import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a 3D run into baseline metrics artifacts")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id>")
    parser.add_argument(
        "--output-json",
        default="outputs/baseline_metrics.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_metrics_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _best_epoch_from_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {}
    best = None
    for row in history:
        try:
            score = float(row.get("val_foreground_dice", 0.0))
            loss = float(row.get("val_loss", 1e9))
        except (TypeError, ValueError):
            continue
        if best is None:
            best = row
            continue
        best_score = float(best.get("val_foreground_dice", 0.0))
        best_loss = float(best.get("val_loss", 1e9))
        if score > best_score or (abs(score - best_score) < 1e-8 and loss < best_loss):
            best = row
    return best or {}


def _extract_sanity_tp_fp_fn(train_log: Path) -> Dict[str, Any]:
    if not train_log.exists():
        return {}
    pattern = re.compile(r"tp=(\d+) fp=(\d+) fn=(\d+)")
    last = None
    with train_log.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                last = match
    if not last:
        return {}
    return {
        "tp": int(last.group(1)),
        "fp": int(last.group(2)),
        "fn": int(last.group(3)),
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg_path = run_dir / "train_config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing train_config_resolved.yaml in {run_dir}")
    cfg = _load_yaml(cfg_path)

    metrics_json = _load_metrics_json(run_dir / "metrics_per_epoch.json")
    metrics_csv = _load_metrics_csv(run_dir / "metrics.csv")

    if not metrics_csv and metrics_json:
        csv_path = run_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
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
            for row in metrics_json:
                writer.writerow(
                    [
                        row.get("epoch"),
                        row.get("train_loss"),
                        row.get("val_loss"),
                        row.get("val_mean_dice"),
                        row.get("val_foreground_dice"),
                        row.get("val_dice_background"),
                        row.get("val_dice_tumor"),
                        row.get("lr"),
                    ]
                )
        metrics_csv = _load_metrics_csv(csv_path)

    best_row = _best_epoch_from_history(metrics_json or metrics_csv)
    best_epoch = int(best_row.get("epoch", 0)) if best_row else 0

    dice_per_class = best_row.get("val_dice_per_class") if best_row else None
    if isinstance(dice_per_class, list):
        dice_per_class = [float(x) for x in dice_per_class]

    train_log = run_dir / "train.log"
    sanity = _extract_sanity_tp_fp_fn(train_log)

    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    summary = {
        "dataset_format": "msd_task01",
        "label_mode": data_cfg.get("label_mode", "unknown"),
        "roi_size": data_cfg.get("roi_size"),
        "epochs": training_cfg.get("max_epochs"),
        "best_epoch": best_epoch,
        "dice_per_class": dice_per_class,
        "dice_background": float(best_row.get("val_dice_background", 0.0)) if best_row else 0.0,
        "dice_tumor": float(best_row.get("val_dice_tumor", 0.0)) if best_row else 0.0,
        "foreground_dice": float(best_row.get("val_foreground_dice", 0.0)) if best_row else 0.0,
        "ignore_empty_foreground": training_cfg.get("ignore_empty_foreground", True),
        "prediction_threshold": training_cfg.get("prediction_threshold", 0.5),
    }
    if sanity:
        summary["sanity_tp_fp_fn"] = sanity

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    outputs_dir = output_path.parent
    metrics_csv_out = outputs_dir / "metrics.csv"
    if not metrics_csv_out.exists() and metrics_csv:
        with metrics_csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_csv[0].keys())
            writer.writeheader()
            writer.writerows(metrics_csv)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
