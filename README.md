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
    pipeline.py        one call = legacy RBC_analize/main.py
scripts/           command-line entry points (no hardcoded paths)
tests/             checks, including "new package == legacy code" on synthetic data
data/              local data, git-ignored (never committed)
```

The hologram reconstruction itself (Vortex + Legendre) is still MATLAB-only:
`legacy/VortexLegendre/main.m`.

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
# 2. frames -> complex fields: run legacy/VortexLegendre/main.m in MATLAB
# 3. drop bad reconstructions
python scripts/clean_database.py data/holo_1/complex_field_video_1
# 4. analyse one complex field
python scripts/analyze_complex_field.py data/holo_1/complex_field_video_1/complex_field_000001_Hologram_video01_frame_000001.mat --show
```
