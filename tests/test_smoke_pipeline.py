from pathlib import Path

import nibabel as nib
import numpy as np

import imageio

from src.data import prepare_slices


def test_prepare_slices_msd_task01(tmp_path: Path):
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image = np.random.rand(32, 32, 8, 4).astype("float32")
    label = np.zeros((32, 32, 8), dtype="uint8")
    label[8:16, 8:16, 3] = 1

    image_path = images_dir / "brain_000.nii.gz"
    label_path = labels_dir / "brain_000.nii.gz"
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), str(image_path))
    nib.save(nib.Nifti1Image(label, affine=np.eye(4)), str(label_path))

    output_root = tmp_path / "Dataset"
    prepare_slices.prepare_dataset(
        dataset_root=tmp_path,
        output_root=output_root,
        slices_per_volume=2,
        dataset_format="msd_task01",
        channel="flair",
        label_mode="binary",
    )

    pngs = sorted(output_root.rglob("*.png"))
    assert len(pngs) == 4

    mask_paths = sorted((output_root / "test_masks" / "test").glob("*.png"))
    assert mask_paths
    mask = imageio.imread(mask_paths[0])
    assert mask.ndim == 2
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
