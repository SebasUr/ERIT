"""Move bad phase images (and their .mat) to not_good_images/.   python scripts/clean_database.py <folder>"""
import argparse

from erit.quality import clean_folder

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("folder")
ap.add_argument("--binary-threshold", type=int, default=128)
ap.add_argument("--tsm-threshold", type=float, default=8000)
ap.add_argument("--std-threshold", type=float, default=80)
a = ap.parse_args()
clean_folder(a.folder, a.binary_threshold, a.tsm_threshold, a.std_threshold)
