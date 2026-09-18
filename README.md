# ERIT — RBC analysis with Digital Holographic Microscopy

Can a digital holographic microscope (DHM) reproduce a blood test (hemogram: RBC count,
MCV, RDW, ...)? This repository holds the processing code of the EAFIT Applied Optics Group
project *"Assessing Digital Holographic Microscopy for Quantitative Hematological Analysis"*
(R. Castañeda, C. Trujillo, J. Zapata; students M. Lecossois, S. Moreno).

## Layout

```
legacy/            original team code, untouched (reference; see legacy/README.md)
src/erit/          the same code reorganised as an installable Python package
    frames.py          video -> frames
    quality.py         reject bad phase images
    io.py              read complex field (.mat)
    phase.py           wrapped / unwrapped phase
    unwrapping.py      weighted least-squares unwrapping (Ghiglia & Romero)
    segmentation.py    find the cells
    geometry.py        area, diameter, axes of each cell
    profiles.py        two phase profiles and delta-phi per cell
    height_volume.py   phase -> height -> volume
    results.py         JSON / CSV output
    plotting.py        figures
    pipeline.py        one call = legacy RBC_analize/main.py (+ analyze_hologram for raw images)
    config.py          acquisition parameters (cow setup, human dataset)
    reconstruction/    Python port of the MATLAB Vortex + Legendre reconstruction
    datasets/          helpers for the public human RBC dataset (OSF 8P7BA)
scripts/           command-line entry points (no hardcoded paths)
tests/             checks, including "new package == legacy code" on synthetic data
data/              local data, git-ignored (never committed)
```

The hologram reconstruction (Vortex + Legendre) exists both in MATLAB
(`legacy/VortexLegendre/main.m`) and as a Python port (`erit.reconstruction`,
`scripts/reconstruct.py`), so the whole chain runs without MATLAB.

## Install

```bash
conda create -n erit python=3.11
conda activate erit
pip install -e ".[dev]"
pytest
```

## Use

```bash
# 1. video -> frames
python scripts/extract_frames.py holo_1.avi data/holo_1 Hologram_video01
# 2. frames -> complex fields (or run legacy/VortexLegendre/main.m in MATLAB)
python scripts/reconstruct.py data/holo_1 data/holo_1/complex_field_video_1 --preset cow
# 3. drop bad reconstructions
python scripts/clean_database.py data/holo_1/complex_field_video_1
# 4. analyse one complex field
python scripts/analyze_complex_field.py data/holo_1/complex_field_video_1/complex_field_000001_Hologram_video01_frame_000001.mat --show

# public human RBC dataset: extract the validation split and measure every cell (~2.5 min)
python scripts/batch_human_dataset.py data/human_rbc --extract-from "path/to/RBCs Holograms.zip"
```

## Differences from the legacy code

| | legacy | erit |
|---|---|---|
| Phase used for measurements | wrapped phase, arbitrary offset | `quantitative_phase`: offset removed, unwrapped, cells positive (`phase_mode="legacy_wrapped"` keeps the old behaviour) |
| `read_complex_field(field_key=...)` | never assigns the field (crash) | fixed |
| Cells cut by the image border | counted like any other | flagged `touches_border` |
| `cleanDataBase` with `main.m` file names | never finds the `.mat` | finds it |

## Known limitations (open)

- Absolute height/volume depend on `delta_n` (refractive-index difference cell vs medium), which is
  not calibrated. On the human dataset (dn = 0.06) cells come out ~5 um tall and ~230 fL, about 2-3x
  the physiological values, while diameter (~8.1 um) and area (~52 um2) are correct.
- Delta-phi uses the profile minimum, which falls on the cell edge (profiles only extend ~3 px into
  the background), not the background mean.
- Not implemented yet: A/B/C image classification, avoiding the same cell counted in consecutive
  frames, per-frame Excel report, hemoglobin / MCH, comparison with each cow's hemogram.
