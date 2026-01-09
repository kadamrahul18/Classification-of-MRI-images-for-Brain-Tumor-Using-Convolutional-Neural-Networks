# Repository Guidelines

## Project Structure & Module Organization
- `src/` houses the core package.
  - `src/data/` preprocessing and dataset utilities (bias correction, slice prep, augmentations).
  - `src/models/` model definitions (U-Net).
  - `src/training/` training entrypoints and loops.
  - `src/service/` FastAPI inference service.
  - `src/utils/` config helpers.
- `configs/config.yaml` is the primary runtime configuration.
- `train.py` is the top-level training entrypoint.
- `requirements.txt` and `Dockerfile` define dependencies and container runtime.
- `Dataset/`, `weights/`, `outputs/`, and `logs/` are ignored and used for data, checkpoints, and artifacts.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`.
- Prepare slices: `python -m src.data.prepare_slices --dataset-root /path/to/brats2019 --output-root ./Dataset`.
- Optional bias correction: `python -m src.data.bias_correction --input-dir /path/to/brats2019 --output-dir /path/to/brats2019_preprocessed`.
- Train: `python train.py --config configs/config.yaml --data-root ./Dataset`.
- Run API (local): `uvicorn src.service.api:app --host 0.0.0.0 --port 8080`.
- Build container: `docker build -t brain-seg:latest .`.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8-style naming (modules/functions `snake_case`, classes `CamelCase`).
- Keep filenames descriptive and colocate utilities with their domains (e.g., `src/data/*`).
- No formatter or linter is configured; keep changes minimal and readable.

## Testing Guidelines
- No automated test suite is present.
- If adding tests, document the framework and add a clear command in this file and `README.md`.

## Commit & Pull Request Guidelines
- Git history does not show a strict convention; use concise, descriptive commit messages (e.g., “add bias correction CLI flags”).
- PRs should include: a summary, linked issue (if any), and example commands or screenshots for user-facing changes (e.g., API responses).

## Configuration & Runtime Notes
- All runtime parameters live in `configs/config.yaml`; prefer config changes over hard-coded values.
- Model outputs and logs default to `./outputs/` (see `training` config). Keep large artifacts out of git.
