import argparse
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.data.augmentations import get_training_augmentation
from src.data.dataset import build_dataloader
from src.models.unet import build_unet
from src.utils.config import apply_overrides, load_config, resolve_paths


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_callbacks(cfg):
    log_dir = cfg["training"]["log_dir"]
    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    checkpoint_path = Path(checkpoint_dir) / cfg["training"]["checkpoint_filename"]

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"], verbose=1, restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True
    )
    return [tensorboard_callback, early_stopping, model_checkpoint]


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for brain tumor segmentation")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--data-root", dest="data_root", help="Override dataset root directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", help="Override learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    cfg = resolve_paths(cfg)

    set_random_seeds(cfg["training"].get("seed", 42))

    image_size = cfg["data"]["image_size"]
    input_channels = cfg["model"]["input_channels"]
    class_names = cfg["data"]["class_names"]

    train_aug = get_training_augmentation() if cfg.get("augmentation", {}).get("enable", False) else None

    train_loader = build_dataloader(
        cfg["data"]["train_images"],
        cfg["data"]["train_masks"],
        class_names,
        batch_size=cfg["training"]["batch_size"],
        image_size=image_size,
        augmentation=train_aug,
        shuffle=True,
    )
    val_loader = build_dataloader(
        cfg["data"]["val_images"],
        cfg["data"]["val_masks"],
        class_names,
        batch_size=cfg["training"]["batch_size"],
        image_size=image_size,
        augmentation=None,
        shuffle=False,
    )

    model = build_unet(
        input_size=(image_size, image_size, input_channels),
        num_classes=len(class_names),
        base_filters=cfg["model"].get("base_filters", 32),
        learning_rate=cfg["training"]["learning_rate"],
    )

    steps_per_epoch = cfg["training"].get("steps_per_epoch") or len(train_loader)
    validation_steps = cfg["training"].get("validation_steps") or len(val_loader)

    callbacks = create_callbacks(cfg)

    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=cfg["training"]["epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        use_multiprocessing=cfg["training"].get("use_multiprocessing", False),
        workers=cfg["training"].get("workers", 1),
    )


if __name__ == "__main__":
    main()
