#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: python executable not found." >&2
    exit 1
  fi
fi
CONFIG_PATH="${ROOT_DIR}/configs/config_3d_baseline.yaml"
DATA_ROOT="${ROOT_DIR}/data/raw/msd_task01/Task01_BrainTumour"
RUNS_DIR="${ROOT_DIR}/outputs/runs"
BASELINE_DIR="${ROOT_DIR}/outputs/baseline_v1/$(date +%Y%m%d_%H%M%S)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required." >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)"
if [[ -z "${GPU_NAME}" ]]; then
  echo "ERROR: Unable to detect GPU name via nvidia-smi." >&2
  exit 1
fi
echo "GPU detected: ${GPU_NAME}"

if [[ ! -d "${DATA_ROOT}/imagesTr" || ! -d "${DATA_ROOT}/labelsTr" ]]; then
  echo "Dataset not found at ${DATA_ROOT}. Downloading MSD Task01..."
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_msd_task01.py"
fi

echo "Running training with ${CONFIG_PATH}"
"${PYTHON_BIN}" -m src.train_3d --config "${CONFIG_PATH}"

LATEST_RUN="$(ls -td "${RUNS_DIR}"/* | head -n1)"
if [[ -z "${LATEST_RUN}" || ! -d "${LATEST_RUN}" ]]; then
  echo "ERROR: No run directory found in ${RUNS_DIR}." >&2
  exit 1
fi

BEST_CHECKPOINT="${LATEST_RUN}/best.pt"
if [[ ! -f "${BEST_CHECKPOINT}" ]]; then
  echo "ERROR: best.pt not found at ${BEST_CHECKPOINT}." >&2
  exit 1
fi

echo "Running evaluation with ${BEST_CHECKPOINT}"
"${PYTHON_BIN}" -m src.eval_3d --config "${CONFIG_PATH}" --weights "${BEST_CHECKPOINT}"

mkdir -p "${BASELINE_DIR}"

cp "${CONFIG_PATH}" "${BASELINE_DIR}/config_3d_baseline.yaml"
cp "${LATEST_RUN}/train_config_resolved.yaml" "${BASELINE_DIR}/train_config_resolved.yaml"
cp "${LATEST_RUN}/train.log" "${BASELINE_DIR}/train.log"
cp "${LATEST_RUN}/metrics.csv" "${BASELINE_DIR}/metrics.csv"
cp "${LATEST_RUN}/metrics_per_epoch.json" "${BASELINE_DIR}/metrics_per_epoch.json"
cp "${BEST_CHECKPOINT}" "${BASELINE_DIR}/best.pt"

if [[ -d "${LATEST_RUN}/vis" ]]; then
  cp -R "${LATEST_RUN}/vis" "${BASELINE_DIR}/vis"
fi

if [[ -f "${ROOT_DIR}/outputs/metrics_3d.json" ]]; then
  cp "${ROOT_DIR}/outputs/metrics_3d.json" "${BASELINE_DIR}/metrics_3d.json"
fi

git -C "${ROOT_DIR}" rev-parse --short HEAD > "${BASELINE_DIR}/git_commit.txt"
{
  "${PYTHON_BIN}" --version 2>&1
  "${PYTHON_BIN}" - <<'PY'
import importlib
for pkg in ("torch", "monai"):
    try:
        mod = importlib.import_module(pkg)
        print(f"{pkg}=={getattr(mod, '__version__', 'unknown')}")
    except Exception:
        print(f"{pkg} not installed")
PY
} > "${BASELINE_DIR}/env.txt"

nvidia-smi -L > "${BASELINE_DIR}/gpu.txt"

SUMMARY_JSON="${ROOT_DIR}/outputs/metrics_3d.json"
if [[ -f "${SUMMARY_JSON}" ]]; then
  python - <<PY
import json
from pathlib import Path
path = Path("${SUMMARY_JSON}")
data = json.loads(path.read_text())
val = data.get("val", {})
print("Baseline summary:")
print(f"  best checkpoint: ${BEST_CHECKPOINT}")
print(f"  val dice_tumor: {val.get('dice_tumor')}")
print(f"  val foreground_dice: {val.get('foreground_mean_dice')}")
PY
fi

echo "Baseline artifacts copied to ${BASELINE_DIR}"
