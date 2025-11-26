# Repository Guidelines

## Project Structure & Module Organization
- `configs/config.yaml`: centralizes paths, hyperparameters, and augmentation toggles.
- `src/data/augmentations.py`: albumentations pipeline for spatial transforms.
- `src/data/dataset.py`: PNG slice loader + Keras `Sequence` dataloader with one-hot masks (grayscale inputs, shape `(H,W,1)`).
- `src/data/prepare_slices.py`: converts BraTS NIfTI volumes to PNG slices and splits into train/val/test.
- `src/data/bias_correction.py`: N4 bias-field correction for BraTS volumes.
- `src/models/unet.py`: parameterized U-Net builder with configurable base filters and input shape.
- `src/training/train.py`: training entry that wires config, dataloaders, callbacks, and model. Root `train.py` is a thin wrapper.

## Build, Test, and Development Commands
- `python -m src.data.bias_correction --input-dir /raw/brats --output-dir /preprocessed` — optional N4 bias correction.
- `python -m src.data.prepare_slices --dataset-root /raw/brats --output-root ./Dataset --slices-per-volume 20` — export PNG slices and create splits.
- `python train.py --config configs/config.yaml [--data-root ./Dataset --epochs 30 --batch-size 8]` — train U-Net with overrides.
- Install deps: `pip install -r requirements.txt`.

## Coding Style & Naming Conventions
- PEP8 with 4-space indentation; snake_case for functions, variables, and filenames.
- Keep images as single-channel float32 in [0,1]; masks are one-hot with 4 channels (background, non-enhancing, edema, enhancing).
- Preserve filename patterns (`*_frame_#####.png`, `*_mask_#####.png`) and consistent numbering so loaders align frames and masks.

## Testing Guidelines
- No automated tests yet; sanity-check preprocessing on a small subset and visually inspect paired frames/masks.
- Before long runs, run a 1–2 epoch smoke test with small batch and confirm TensorBoard logs and checkpoint writes.
- If adding tests, consider unit tests for `SliceDataset` shape/dtype expectations and smoke tests that instantiate `build_unet()` and train on a tiny batch.

## Commit & Pull Request Guidelines
- Prefer short, imperative commit messages (e.g., `Refine augmentation pipeline`) that describe the change scope.
- PRs should summarize dataset location, preprocessing choices (bias correction on/off), and key training hyperparameters; include sample logs or plots when available.
- Link related issues and call out changes to data paths, augmentation behavior, or model architecture for targeted review.
