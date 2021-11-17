import numpy as np
import glob, os
import itertools
import random
import shutil
import SimpleITK as sitk

config = dict()
config["modalities"] = ["flair", "t1", "t1ce", "t2"]

def correct_bias(in_path, out_path, image_type=sitk.sitkFloat64):
    input_image = sitk.ReadImage(in_path, image_type)
    output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    sitk.WriteImage(output_image, out_path)
    return os.path.abspath(out_path)

def get_image_path(subject_folder, name):
    file_name = os.path.join(subject_folder, "*" + name + ".nii.gz")
    return glob.glob(file_name)[0]


def normalize_image(in_path, out_path, bias_correction=True):
    if bias_correction:
        correct_bias(in_path, out_path)
    else:
        shutil.copy(in_path, out_path)

def preprocess_brats_folder(in_folder, out_folder, truth_name='seg', no_bias_correction_modalities=None):
    for name in config["modalities"]:
        image_image = get_image_path(in_folder, name)
        case_ID = os.path.basename(out_folder)
        out_path = os.path.abspath(os.path.join(out_folder, "%s_%s.nii.gz"%(case_ID, name)))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
        normalize_image(image_image, out_path, bias_correction=perform_bias_correction)

    truth_image = get_image_path(in_folder, truth_name)
    out_path = os.path.abspath(os.path.join(out_folder, "%s_truth.nii.gz"%(case_ID)))
    shutil.copy(truth_image, out_path)

def preprocess_brats_data(brats_folder, out_folder, overwrite=False, no_bias_correction_modalities=("flair")):
    for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)), subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                preprocess_brats_folder(subject_folder, new_subject_folder, no_bias_correction_modalities=no_bias_correction_modalities)

def main(brats_path, preprocessed_brats):
    preprocess_brats_data(brats_path, preprocessed_brats)

if __name__ == "__main__":
    main(brats_path='/home/rahul/Desktop/brats19/', preprocessed_brats='/home/rahul/Desktop/brats19_Preprocessed/')
