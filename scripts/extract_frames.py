"""Split a hologram video into frames.   python scripts/extract_frames.py holo_1.avi out/holo_1 Hologram_video01"""
import argparse

from erit.frames import extract_frames

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("video")
ap.add_argument("output_folder")
ap.add_argument("prefix", help="unique per video, e.g. Hologram_video01")
ap.add_argument("--format", default="tif")
ap.add_argument("--wavelength-um", type=float, default=0.633)
ap.add_argument("--pixel-size-um", type=float, default=3.75)
ap.add_argument("--magnification", type=float, default=40)
a = ap.parse_args()

n = extract_frames(a.video, a.output_folder, a.prefix, a.format, {
    "wavelength_um": a.wavelength_um, "pixel_size_um": a.pixel_size_um, "magnification": a.magnification,
    "reconstruction_method": "Angular Spectrum", "notes": "Frames extracted from AVI video for DHM processing"})
print(f"Extracted frames: {n}")
