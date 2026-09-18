"""Hologram images -> complex fields with the Python Vortex-Legendre port (replaces legacy main.m).

    python scripts/reconstruct.py data/holo_1 data/holo_1/complex_field_video_1 --preset cow

Writes complex_field_<k>_<name>.mat (variable 'output', like MATLAB) and phase_<k>_<name>.png.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from erit.config import EAFIT_COW, HUMAN_RBC_DATASET
from erit.reconstruction import vortex_legendre

EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def natural_key(p):
    nums = re.findall(r"\d+", p.name)
    return int(nums[-1]) if nums else 0


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("input_folder")
ap.add_argument("output_folder")
ap.add_argument("--preset", choices=["cow", "human"], default="cow")
ap.add_argument("--limit", type=int, default=128, help="Legendre fit half-size (filterRadius in main.m)")
a = ap.parse_args()

acq = EAFIT_COW if a.preset == "cow" else HUMAN_RBC_DATASET
out = Path(a.output_folder)
out.mkdir(parents=True, exist_ok=True)
files = sorted((f for f in Path(a.input_folder).iterdir() if f.suffix.lower() in EXT), key=natural_key)
print(f"Number of holograms found: {len(files)}")

for k, f in enumerate(files, 1):
    print(f"Processing {k} / {len(files)}: {f.name}")
    field = vortex_legendre(np.array(Image.open(f)).astype(float), acq.wavelength, acq.pixel_size, limit=a.limit)
    savemat(out / f"complex_field_{k:06d}_{f.stem}.mat",
            {"output": field, "currentName": f.name, "wavelength": acq.wavelength, "pixelSize": acq.pixel_size})
    ph = np.angle(field)
    Image.fromarray(((ph - ph.min()) / (np.ptp(ph) or 1) * 255).astype(np.uint8)).save(out / f"phase_{k:06d}_{f.stem}.png")
