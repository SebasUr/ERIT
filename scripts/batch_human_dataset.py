"""Run the full chain on the public human RBC dataset and collect one row per cell.

    python scripts/batch_human_dataset.py data/human_rbc --split Validation --workers 4
    python scripts/batch_human_dataset.py data/human_rbc --extract-from "RBCs Holograms.zip"

Uses 1 BLAS thread per worker: without it every worker's SVD spawns one thread per core
and the machine thrashes (seen: ~86 s/image instead of ~0.2 s).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import pandas as pd  # noqa: E402

from erit.config import HUMAN_RBC_DATASET  # noqa: E402
from erit.datasets import human_rbc  # noqa: E402
from erit.pipeline import analyze_hologram  # noqa: E402

PROCESSING = {"min_area_um2": 20, "max_area_um2": 120, "delta_n": 0.06, "phase_mode": "quantitative"}


def one(path):
    warnings.filterwarnings("ignore")
    try:
        a = analyze_hologram(str(path), HUMAN_RBC_DATASET, PROCESSING)
    except Exception as e:  # keep going, report at the end
        return [{"file": path.name, "error": repr(e)}]
    prof = {r["rbc_id"]: r for r in a["profiles"]["rbcs"]}
    hv = {r["rbc_id"]: r for r in a["height_volume"]["rbcs"]}
    rows = []
    for g in a["geometry"]["rbcs"]:
        i = g["rbc_id"]
        rows.append({"file": path.name, "rbc": i, "area_um2": g["area_um2"],
                     "diam_um": g["equivalent_diameter_um"], "minor_major": g["minor_axis_um"] / g["major_axis_um"],
                     "touches_border": g["touches_border"], "dphi_rad": prof.get(i, {}).get("delta_phase"),
                     "height_um": hv.get(i, {}).get("representative_height_um"),
                     "volume_um3": hv.get(i, {}).get("volume_um3")})
    return rows or [{"file": path.name}]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing Holograms/ and Phase/")
    ap.add_argument("--split", default="Validation")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, help="only the first N images")
    ap.add_argument("--extract-from", help="path to 'RBCs Holograms.zip' to extract the split first")
    ap.add_argument("--out", default="results/human_rbc_cells.csv")
    a = ap.parse_args()

    if a.extract_from:
        print("extracted", human_rbc.extract(a.extract_from, a.root, a.split), "files")
    files = human_rbc.list_holograms(a.root, a.split)[: a.n]
    rows, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for k, rr in enumerate(pool.imap_unordered(one, files, chunksize=4), 1):
            rows += rr
            if k % 200 == 0 or k == len(files):
                print(f"{k}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df.dropna(subset=["dphi_rad"])
    ok = ok[~ok.touches_border.astype(bool)]
    print(f"images {len(files)} | cells measured (not on border) {len(ok)} | "
          f"median diameter {ok.diam_um.median():.2f} um, area {ok.area_um2.median():.1f} um2, "
          f"dphi {ok.dphi_rad.median():.2f} rad | saved {a.out}")
