# Brain Tumor Segmentation Pipeline (BraTS 2019)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

## Project Overview (Why)
Glioma segmentation on MRI is critical for surgical planning and treatment response tracking, but manual delineation is slow and subjective. This pipeline automates multi-modal MRI segmentation (T1, T1ce, T2, FLAIR) from BraTS 2019 using a U-Net tailored for medical images. It tackles class imbalance and boundary precision with hybrid Dice + categorical crossentropy loss, while robust augmentations and N4 bias correction improve signal consistency across scanners.

## System Architecture
```mermaid
graph TD
    A[NIfTI Volumes] --> B[Preprocessing<br/>Bias Correction / Normalization]
    B --> C[Data Augmentation]
    C --> D[U-Net Model]
    D --> E[Inference API<br/>(FastAPI)]
    E --> F[Output<br/>Segmentation Mask]
```

## Key Features
- N4 Bias Field Correction with SimpleITK to stabilize intensity profiles.
- Custom Data Loader for PNG slices with class-consistent pairing and one-hot masks.
- Hybrid Loss Function (Dice + Categorical Crossentropy) to sharpen tumor boundaries under class imbalance.
- Containerized Inference via FastAPI + Docker, deployable on GCP Vertex AI.
- Headless dependencies for lean containers (`opencv-python-headless`) and `.dockerignore` to keep builds small.

## Getting Started (Local)
1) Clone: `git clone https://github.com/your-org/brain-tumor-segmentation-pipeline.git && cd brain-tumor-segmentation-pipeline`
2) Install deps: `pip install -r requirements.txt`
3) Train: `python train.py --config configs/config.yaml --data-root ./Dataset`
   - Prepare data first with `python -m src.data.prepare_slices --dataset-root /path/to/brats2019 --output-root ./Dataset`
   - Optional bias correction: `python -m src.data.bias_correction --input-dir /path/to/brats2019 --output-dir /path/to/brats2019_preprocessed`

## Docker & Deployment
- Build image: `docker build -t brain-seg:latest .`
- Run container (serving on 8080): `docker run -p 8080:8080 brain-seg:latest`
- FastAPI endpoints:
  - `/health` — readiness check (Vertex AI compatible).
  - `/predict` — returns PNG mask for an uploaded image (`UploadFile`).
  - `/predict-json` — returns class map as JSON.

## Results
**Qualitative Results** (drop in GIFs or side-by-side examples here)

---
Tech stack: TensorFlow/Keras, SimpleITK, Nibabel, Albumentations, FastAPI, Docker, Vertex AI.***
