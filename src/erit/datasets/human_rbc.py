"""Public human RBC hologram dataset (OSF 10.17605/OSF.IO/8P7BA).

Castaneda, Trujillo & Doblas, "A human erythrocytes hologram dataset for learning-based
model training", Data in Brief 54 (2024) 110424.

Layout inside "RBCs Holograms.zip":
    Holograms/{Training,Validation}/Image__<date>__<time>_<i>-<j>[-A|-B].png   256x256 uint8
    Phase/{Training,Validation}/<same name>.png                              256x256 uint8

- 300 full holograms (1920x1200) cropped into 256x256 patches; -A / -B are rotations or
  flips of the same patch (the same cells!) -> use originals only for statistics.
- Phase PNGs are wrapped and min-max normalised to 0..255: not quantitative radians.
- The OSF download is a zip that contains "RBCs Holograms.zip" (Zip64; macOS `unzip`
  fails on it, Python's zipfile works).
"""
import re
import zipfile
from pathlib import Path

from ..config import HUMAN_RBC_DATASET as ACQUISITION  # noqa: F401

NAME_RE = re.compile(r"Image__(?P<date>[\d-]+)__(?P<time>[\d-]+)_(?P<i>\d+)-(?P<j>\d+)(?:-(?P<aug>[A-Z]))?\.png$")


def extract(zip_path, dest, split="Validation"):
    """Extract one split ("Training" / "Validation" / None for all) of the inner zip."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".png") and (split is None or f"/{split}/" in n)]
        z.extractall(dest, names)
    return len(names)


def parse_name(path):
    m = NAME_RE.search(Path(path).name)
    return m.groupdict() if m else None


def list_holograms(root, split="Validation", originals_only=True):
    files = sorted(Path(root, "Holograms", split).glob("*.png"))
    if originals_only:
        files = [f for f in files if (parse_name(f) or {}).get("aug") is None]
    return files


def phase_path(hologram_path):
    p = Path(hologram_path)
    return p.parents[2] / "Phase" / p.parent.name / p.name
