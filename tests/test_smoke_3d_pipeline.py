import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.networks.nets import UNet

from src.data.msd_task01_3d import MSDTask01Dataset3D, list_msd_task01_cases
from src.eval_3d import main as eval_main


def _write_msd_case(root: Path):
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image = np.random.rand(16, 16, 16, 4).astype("float32")
    label = np.zeros((16, 16, 16), dtype="uint8")
    label[4:8, 4:8, 4:8] = 1

    image_path = images_dir / "case_000.nii.gz"
    label_path = labels_dir / "case_000.nii.gz"
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), str(image_path))
    nib.save(nib.Nifti1Image(label, affine=np.eye(4)), str(label_path))


def test_3d_dataset_model_eval(tmp_path: Path, monkeypatch):
    _write_msd_case(tmp_path)
    cases = list_msd_task01_cases(tmp_path)
    dataset = MSDTask01Dataset3D(
        cases,
        roi_size=(16, 16, 16),
        label_mode="binary",
        num_classes=2,
        pos_ratio=1.0,
        percentiles=(0.5, 99.5),
        mode="train",
        seed=123,
    )

    image, label = dataset[0]
    assert image.shape == (4, 16, 16, 16)
    assert label.shape == (2, 16, 16, 16)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=2,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1,
        norm="batch",
    )
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    assert output.shape[1] == 2

    weights_path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weights_path)

    config = {
        "data": {
            "root": str(tmp_path),
            "train_ratio": 0.0,
            "val_ratio": 1.0,
            "seed": 42,
            "label_mode": "binary",
            "class_names": ["background", "tumor"],
            "roi_size": [16, 16, 16],
            "pos_ratio": 1.0,
            "percentiles": [0.5, 99.5],
        },
        "model": {
            "in_channels": 4,
            "channels": [8, 16],
            "strides": [2],
            "num_res_units": 1,
            "norm": "batch",
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 0.0001,
            "max_epochs": 1,
            "num_workers": 0,
            "seed": 42,
            "deterministic": True,
            "output_dir": "./outputs/runs",
            "limit_train_batches": 1,
            "limit_val_batches": 1,
        },
        "inference": {"roi_size": [16, 16, 16], "overlap": 0.0, "sw_batch_size": 1},
    }
    config_path = tmp_path / "config_3d.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["python", "-m", "src.eval_3d", "--config", str(config_path), "--weights", str(weights_path)],
    )
    eval_main()

    metrics_path = tmp_path / "outputs" / "metrics_3d.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "val" in metrics and "test" in metrics
