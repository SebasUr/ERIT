"""Analyse one complex-field .mat (equivalent to legacy RBC_analize/main.py).

    python scripts/analyze_complex_field.py complex_field_000001_Hologram_video01_frame_000001.mat --show
"""
import argparse

from erit.pipeline import analyze_complex_field

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("mat")
ap.add_argument("--wavelength", type=float, default=0.632, help="um")
ap.add_argument("--pixel-size", type=float, default=3.75, help="camera pixel, um")
ap.add_argument("--magnification", type=float, default=40)
ap.add_argument("--delta-n", type=float, default=0.083)
ap.add_argument("--out", default="results")
ap.add_argument("--show", action="store_true")
a = ap.parse_args()

result, df, json_path, csv_path = analyze_complex_field(
    a.mat, a.wavelength, a.pixel_size, a.magnification,
    processing={"delta_n": a.delta_n}, output_directory=a.out, show=a.show)
print(df)
print(result["statistics"])
print("saved:", json_path, csv_path)
