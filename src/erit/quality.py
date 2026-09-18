"""Reject bad phase images before analysis (from legacy/cleanDataBase.py).

An image is rejected when BOTH metrics exceed their thresholds:
    TSM = M*N - (number of pixels above `binary_threshold`)   -> many dark pixels
    STD = standard deviation of the 8-bit phase image          -> very noisy / wrapped
"""

import os
import shutil

import cv2
import numpy as np

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
COMPLEX_FIELD_EXTENSION = ".mat"


def image_metrics(image, binary_threshold=128):
    """Return (tsm_value, std_value) for one grayscale phase image."""
    M, N = image.shape
    _, binary_image = cv2.threshold(image, binary_threshold, 1, cv2.THRESH_BINARY)
    tsm_value = M * N - np.sum(binary_image)
    std_value = np.std(image)
    return tsm_value, std_value


def is_bad_image(image, binary_threshold=128, tsm_threshold=8000, std_threshold=80):
    tsm_value, std_value = image_metrics(image, binary_threshold)
    return bool(tsm_value > tsm_threshold and std_value > std_threshold), tsm_value, std_value


def find_corresponding_complex_field(phase_filename, folder):
    """
    Find the complex field associated with a phase image.

    Example:

    Phase image:
    phase_000003_Hologram_video45_frame_000003.png

    Corresponding complex field (as written by legacy/VortexLegendre/main.m):
    complex_field_000003_Hologram_video45_frame_000003.mat

    The legacy version required the .mat name to end with the full phase name
    ("phase_000003_..."), which never happens with main.m names, so the .mat was
    never moved. Both naming styles are accepted here.
    """
    phase_name_without_extension = os.path.splitext(phase_filename)[0]
    core = phase_name_without_extension[len("phase_"):] if phase_name_without_extension.startswith("phase_") \
        else phase_name_without_extension

    for candidate_filename in os.listdir(folder):
        candidate_path = os.path.join(folder, candidate_filename)

        if os.path.isdir(candidate_path):
            continue

        if not candidate_filename.lower().endswith(COMPLEX_FIELD_EXTENSION):
            continue

        candidate_name_without_extension = os.path.splitext(candidate_filename)[0]

        # Verify that the complex-field filename ends with the complete phase-image name
        if (candidate_name_without_extension.endswith(phase_name_without_extension)
                or candidate_name_without_extension == f"complex_field_{core}"):
            return candidate_filename

    return None


def clean_folder(input_folder, binary_threshold=128, tsm_threshold=8000, std_threshold=80, verbose=True):
    """
    Move rejected phase_*.png images (and their complex field) into <input_folder>/not_good_images.

    outputs:
        List of (filename, tsm_value, std_value, moved) tuples.
    """
    not_good_folder = os.path.join(input_folder, "not_good_images")
    os.makedirs(not_good_folder, exist_ok=True)
    report = []

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if os.path.isdir(file_path):
            continue
        if not filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
            continue
        if not filename.lower().startswith("phase_"):
            continue

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            if verbose:
                print(f"Could not read image: {filename}")
            continue

        bad, tsm_value, std_value = is_bad_image(image, binary_threshold, tsm_threshold, std_threshold)
        report.append((filename, float(tsm_value), float(std_value), bad))

        if not bad:
            if verbose:
                print(f"Kept: {filename} | TSM = {tsm_value:.2f}, STD = {std_value:.2f}")
            continue

        shutil.move(file_path, os.path.join(not_good_folder, filename))
        if verbose:
            print(f"Moved phase image: {filename} | TSM = {tsm_value:.2f}, STD = {std_value:.2f}")

        complex_field_filename = find_corresponding_complex_field(filename, input_folder)

        if complex_field_filename is not None:
            shutil.move(os.path.join(input_folder, complex_field_filename),
                        os.path.join(not_good_folder, complex_field_filename))
            if verbose:
                print(f"Moved complex field: {complex_field_filename}")
        elif verbose:
            print(f"Warning: No corresponding complex field was found for {filename}")

    return report
