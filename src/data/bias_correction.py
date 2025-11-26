import argparse
import glob
import os
import shutil
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk

MODALITIES = ("flair", "t1", "t1ce", "t2")


def correct_bias(in_path, out_path, image_type=sitk.sitkFloat64):
    input_image = sitk.ReadImage(in_path, image_type)
    output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    sitk.WriteImage(output_image, out_path)
    return os.path.abspath(out_path)


def get_image_path(subject_folder, name):
    file_name = os.path.join(subject_folder, f"*{name}.nii.gz")
    matches = glob.glob(file_name)
    if not matches:
        raise FileNotFoundError(f"Could not find modality {name} in {subject_folder}")
    return matches[0]


def normalize_image(in_path, out_path, bias_correction=True):
    if bias_correction:
        correct_bias(in_path, out_path)
    else:
        shutil.copy(in_path, out_path)


def preprocess_brats_folder(
    in_folder: Path,
    out_folder: Path,
    modalities: Iterable[str],
    truth_name: str,
    no_bias_correction_modalities: Iterable[str],
):
    for name in modalities:
        image_image = get_image_path(in_folder, name)
        case_id = os.path.basename(out_folder)
        out_path = os.path.abspath(os.path.join(out_folder, f"{case_id}_{name}.nii.gz"))
        perform_bias_correction = name not in no_bias_correction_modalities
        normalize_image(image_image, out_path, bias_correction=perform_bias_correction)

    truth_image = get_image_path(in_folder, truth_name)
    out_path = os.path.abspath(os.path.join(out_folder, f"{case_id}_truth.nii.gz"))
    shutil.copy(truth_image, out_path)


def preprocess_brats_data(
    brats_folder: Path,
    out_folder: Path,
    overwrite: bool = False,
    no_bias_correction_modalities: Iterable[str] = ("flair",),
    modalities: Iterable[str] = MODALITIES,
):
    for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)), subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                preprocess_brats_folder(
                    Path(subject_folder),
                    Path(new_subject_folder),
                    modalities=modalities,
                    truth_name="seg",
                    no_bias_correction_modalities=no_bias_correction_modalities,
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Run N4 bias-field correction on BraTS data")
    parser.add_argument("--input-dir", required=True, help="Path to raw BraTS dataset")
    parser.add_argument("--output-dir", required=True, help="Where to write corrected data")
    parser.add_argument("--skip-modalities", nargs="*", default=["flair"], help="Modalities to skip bias correction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_brats_data(
        brats_folder=input_dir,
        out_folder=output_dir,
        overwrite=args.overwrite,
        no_bias_correction_modalities=args.skip_modalities,
    )


if __name__ == "__main__":
    main()
