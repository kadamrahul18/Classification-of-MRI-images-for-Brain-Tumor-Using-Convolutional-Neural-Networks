# 3D Brain Tumor Segmentation (MSD Task01) — Reproducible Baseline

An end-to-end medical imaging pipeline for 3D brain tumor segmentation with strong correctness checks, reproducible configs, and GPU-ready training/evaluation.

## What This Repo Demonstrates
- **Data pipeline rigor**: NIfTI loading, modality normalization, ROI sampling safeguards, and label alignment checks.
- **Training correctness**: per-class Dice, foreground Dice (ignoring empty tumor), tumor-based checkpointing.
- **Reproducibility**: resolved configs, environment capture, and deterministic-ish runs.
- **Engineering hygiene**: scripts, structured outputs, and clear metrics artifacts.

## Quickstart (Baseline v1.0.0)

### 1) Create environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-3d.txt
```

### 2) Download dataset
```bash
python scripts/download_msd_task01.py
```

### 3) Run baseline (single GPU)
```bash
bash scripts/run_baseline_3d.sh
```

The script:
- Trains with `configs/config_3d_baseline.yaml`
- Evaluates the best checkpoint
- Copies artifacts into `outputs/baseline_v1/<timestamp>/`

Generate summary + figures:
```bash
python scripts/summarize_run.py --run-dir outputs/runs/<run_id>
python scripts/make_readme_figures.py --run-dir outputs/runs/<run_id>
```

## Results — Baseline v1.0.0

**Note:** Fill the table after running the baseline script (it writes `outputs/baseline_metrics.json`).  
The baseline config uses **limited epochs/batches** to keep runtime under ~2 hours on a V100.

| GPU | ROI | Epochs | Best Epoch | Tumor Dice | Foreground Dice | Dataset Split |
| --- | --- | ------ | ---------- | ---------- | ---------------- | ------------- |
| TBD | 96³ | 20 | TBD | TBD | TBD | 70/20/10 |

Figures below are placeholders until you run `scripts/make_readme_figures.py`.

![Baseline Examples](docs/assets/baseline_examples.png)
![Baseline Curves](docs/assets/baseline_curves.png)

## Monitoring
```bash
tensorboard --logdir outputs/runs
```
Logged scalars include loss, per-class Dice, foreground Dice, LR, and GPU memory. Visual overlays are saved every `vis_interval` epochs.

## 2D Baseline (Optional)
The original 2D slice pipeline remains available for comparison.
```bash
python -m src.data.prepare_slices --dataset-format msd_task01 --dataset-root ./data/raw/msd_task01/Task01_BrainTumour --output-root ./Dataset
python train.py --config configs/config.yaml --epochs 20
```

## Known Limitations / Next Improvements
- Longer training schedules and LR scheduling.
- Augmentations (3D flips/rotations/intensity jitter).
- Multiclass tumor labels (enhancing vs non-enhancing).
- Larger ROI for more context (if GPU memory allows).

## Serving (Local)
The FastAPI service supports **2D PNG inference** for quick demos:
```bash
uvicorn src.service.api:app --host 0.0.0.0 --port 8080
curl -F "file=@example.png" http://localhost:8080/predict -o mask.png
```

## Metric Conventions (3D)
- **Per-class Dice** is always reported (background + tumor).
- **Foreground Dice** ignores empty‑tumor patches/volumes.
- **Best checkpoint** is selected by tumor/foreground Dice (not background mean).
