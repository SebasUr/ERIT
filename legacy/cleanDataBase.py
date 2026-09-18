import os
import shutil
import cv2
import numpy as np


# Main folder containing phase images and complex fields
input_folder = r"D:\Videos_problem\holo_45\complex_field_video_45"

# Threshold used to binarize the image
binary_threshold = 128

# Threshold for the TSM metric
# TSM = M*N - sum(binary_image)
tsm_threshold = 8000

# Threshold for the standard deviation
std_threshold = 80

# Allowed phase-image extensions
valid_image_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp"
)

# Complex-field extension
complex_field_extension = ".mat"

# Output folder
not_good_folder = os.path.join(input_folder, "not_good_images")
os.makedirs(not_good_folder, exist_ok=True)


def find_corresponding_complex_field(phase_filename, folder):
    """
    Find the complex field associated with a phase image.

    Example:

    Phase image:
    phase_000003_Hologram_video45_frame_000003.png

    Corresponding complex field:
    complex_field_000003_phase_000003_Hologram_video45_frame_000003.mat
    """

    # Remove the image extension
    phase_name_without_extension = os.path.splitext(phase_filename)[0]

    # Search all MAT files inside the folder
    for candidate_filename in os.listdir(folder):

        candidate_path = os.path.join(folder, candidate_filename)

        # Skip folders
        if os.path.isdir(candidate_path):
            continue

        # Only evaluate MAT files
        if not candidate_filename.lower().endswith(complex_field_extension):
            continue

        candidate_name_without_extension = os.path.splitext(
            candidate_filename
        )[0]

        # Verify that the complex-field filename ends with the complete
        # phase-image name
        if candidate_name_without_extension.endswith(
            phase_name_without_extension
        ):
            return candidate_filename

    return None


# Processing loop
for filename in os.listdir(input_folder):

    file_path = os.path.join(input_folder, filename)

    # Skip folders
    if os.path.isdir(file_path):
        continue

    # Skip non-image files
    if not filename.lower().endswith(valid_image_extensions):
        continue

    # Only process phase images
    if not filename.lower().startswith("phase_"):
        continue

    # Read image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Could not read image: {filename}")
        continue

    # Image dimensions
    M, N = image.shape

    # Binarize image
    _, binary_image = cv2.threshold(
        image,
        binary_threshold,
        1,
        cv2.THRESH_BINARY
    )

    # Calculate the TSM metric
    # TSM = total number of pixels - number of white pixels
    tsm_value = M * N - np.sum(binary_image)

    # Calculate the standard deviation of the grayscale image
    std_value = np.std(image)

    # Move the phase image and its corresponding complex field
    # only if both metrics exceed their thresholds
    if tsm_value > tsm_threshold and std_value > std_threshold:

        # Move the phase image
        phase_destination = os.path.join(
            not_good_folder,
            filename
        )

        shutil.move(file_path, phase_destination)

        print(
            f"Moved phase image: {filename} | "
            f"TSM = {tsm_value:.2f}, STD = {std_value:.2f}"
        )

        # Find the associated complex field
        complex_field_filename = find_corresponding_complex_field(
            filename,
            input_folder
        )

        if complex_field_filename is not None:

            complex_field_path = os.path.join(
                input_folder,
                complex_field_filename
            )

            complex_field_destination = os.path.join(
                not_good_folder,
                complex_field_filename
            )

            shutil.move(
                complex_field_path,
                complex_field_destination
            )

            print(
                f"Moved complex field: {complex_field_filename}"
            )

        else:
            print(
                f"Warning: No corresponding complex field was found "
                f"for {filename}"
            )

    else:
        print(
            f"Kept: {filename} | "
            f"TSM = {tsm_value:.2f}, STD = {std_value:.2f}"
        )