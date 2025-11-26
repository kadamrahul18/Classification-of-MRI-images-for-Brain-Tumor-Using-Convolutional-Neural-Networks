import argparse
import copy
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if getattr(args, "data_root", None):
        cfg["data"]["root"] = args.data_root
    if getattr(args, "epochs", None):
        cfg["training"]["epochs"] = args.epochs
    if getattr(args, "batch_size", None):
        cfg["training"]["batch_size"] = args.batch_size
    if getattr(args, "learning_rate", None):
        cfg["training"]["learning_rate"] = args.learning_rate
    return cfg

def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    data_root = Path(cfg["data"]["root"]).expanduser().resolve()
    cfg["data"]["root"] = data_root
    for key in [
        "train_images",
        "train_masks",
        "val_images",
        "val_masks",
        "test_images",
        "test_masks",
    ]:
        cfg["data"][key] = (data_root / cfg["data"][key]).resolve()
    training = cfg.get("training", {})
    for path_key in ["log_dir", "checkpoint_dir"]:
        if path_key in training:
            training[path_key] = Path(training[path_key]).expanduser().resolve()
            training[path_key].mkdir(parents=True, exist_ok=True)
    return cfg
