"""Split a hologram video into individual frames (from legacy/VideosToFrames.py)."""

from datetime import datetime
from pathlib import Path

import cv2


def extract_frames(video_path, output_folder, frame_prefix, output_format="tif",
                   reconstruction_parameters=None):
    """
    Save every frame of `video_path` as <frame_prefix>_frame_000001.<ext> and write a
    <frame_prefix>_metadata.txt with the video information and reconstruction parameters.

    Use a unique prefix per video (Hologram_video01, Hologram_video02, ...) so that
    hologram -> video -> cow stays traceable.

    outputs:
        Number of extracted frames.
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    reconstruction_parameters = reconstruction_parameters or {}

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
        cv2.imwrite(str(output_folder / filename), frame)

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

    return frame_count
