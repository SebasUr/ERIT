# Libraries
import cv2
from pathlib import Path
from datetime import datetime


# IMPORTANT:
# Folder containing the hologram video to be processed.
# Replace this path with the location of your input video.
video_path = r"C:\Users\racastaneq\Documents\MEGA\MEGAsync\RACQ\Universities\05 EAFIT\Research projects\2026\Manon\Samples\holo_1.avi"

# IMPORTANT:
# Specify the output folder where all extracted frames will be saved.
# The folder will be created automatically if it does not already exist.
output_folder = r"C:\Users\racastaneq\Documents\MEGA\MEGAsync\RACQ\Universities\05 EAFIT\Research projects\2026\Manon\Samples\holo_1"


# Prefix used for each saved frame
# IMPORTANT:
# Choose a descriptive prefix for the extracted frames.
# Use a unique prefix for each video (e.g., Hologram_video01, Hologram_video02, ..., Hologram_video93)
# so that the frames can be easily identified and associated with their original video.
frame_prefix = "Hologram_video01"

# Recommended formats: "tif", "png", "bmp", "jpg"
output_format = "tif"

# Reconstruction parameters written into the TXT file
reconstruction_parameters = {
    "wavelength_um": 0.633,
    "pixel_size_um": 3.75,
    "magnification": 40,
    "reconstruction_method": "Angular Spectrum",
    "notes": "Frames extracted from AVI video for DHM processing"
}

# main

def extract_frames(video_path, output_folder, frame_prefix, output_format, reconstruction_parameters):
    video_path = Path(video_path)
    output_folder = Path(output_folder)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        filename = f"{frame_prefix}_frame_{frame_count:06d}.{output_format}"
        save_path = output_folder / filename

        cv2.imwrite(str(save_path), frame)

    cap.release()

    txt_path = output_folder / f"{frame_prefix}_metadata.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("VIDEO FRAME EXTRACTION METADATA\n")
        f.write("================================\n\n")

        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Video file: {video_path.name}\n")
        f.write(f"Video path: {video_path}\n")
        f.write(f"Output folder: {output_folder}\n\n")

        f.write("VIDEO INFORMATION\n")
        f.write("-----------------\n")
        f.write(f"Frames per second: {fps}\n")
        f.write(f"Reported total frames: {total_frames_reported}\n")
        f.write(f"Extracted total frames: {frame_count}\n")
        f.write(f"Frame width: {width} px\n")
        f.write(f"Frame height: {height} px\n")
        f.write(f"Output format: .{output_format}\n\n")

        f.write("RECONSTRUCTION PARAMETERS\n")
        f.write("-------------------------\n")
        for key, value in reconstruction_parameters.items():
            f.write(f"{key}: {value}\n")

    print("Extraction completed successfully.")
    print(f"FPS: {fps}")
    print(f"Extracted frames: {frame_count}")
    print(f"Frames saved in: {output_folder}")
    print(f"Metadata saved in: {txt_path}")


extract_frames(
    video_path=video_path,
    output_folder=output_folder,
    frame_prefix=frame_prefix,
    output_format=output_format,
    reconstruction_parameters=reconstruction_parameters
)