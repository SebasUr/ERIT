# legacy/ — original code (do not edit)

This folder is an exact copy of the team's `Codes/` folder as received on 2026-09-17.
It is kept untouched as the reference: the `erit` package in `src/` was rebuilt from it,
and any difference in behaviour must be explainable against these files.

| File | Language | What it does |
|---|---|---|
| `VideosToFrames.py` | Python | Splits a hologram video (.avi) into individual frames |
| `VortexLegendre/main.m` | MATLAB | Batch reconstruction: hologram image -> complex field `.mat` + phase `.png` |
| `VortexLegendre/vortexLegendre.m`, `functions_vortexLegendre.m` | MATLAB | The Vortex + Legendre reconstruction method |
| `VortexLegendre/unwrap_phase.m`, `phase_unwrap.m` | MATLAB | Phase unwrapping helpers |
| `cleanDataBase.py` | Python | Moves "bad" phase images (TSM / std thresholds) to `not_good_images/` |
| `RBC_analize/main.py` | Python | Analysis of one complex field: segmentation, geometry, phase profiles, height, volume -> JSON/CSV |
| `RBC_analize/functions_hematological.py` | Python | All analysis functions used by `main.py` |
| `RBC_analize/data_management.py` | Python | Build/save the JSON and CSV results |
| `RBC_analize/unwrapping.py` | Python | Weighted least-squares phase unwrapping (Ghiglia & Romero) |
| `RBC_analize/*.m` | MATLAB | Earlier, interactive (manual cell selection) versions of the analysis |

`functions_anemia.asv` is a MATLAB autosave file; it is kept only because it came with the originals.
