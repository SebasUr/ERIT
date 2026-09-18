"""ERIT — quantitative analysis of red blood cells (RBCs) with digital holographic microscopy.

Processing chain (one module per step):

    frames        video (.avi)            -> hologram frames
    [MATLAB]      hologram                -> complex field (.mat)      legacy/VortexLegendre
    quality       phase image             -> keep / reject
    io            .mat                    -> complex field + acquisition parameters
    phase         complex field           -> wrapped / unwrapped phase
    segmentation  phase                   -> labelled RBC mask
    geometry      mask                    -> area, diameter, axes ...
    profiles      phase + mask            -> two profiles and delta phi per RBC
    height_volume phase + mask + profiles -> height and volume per RBC
    results       everything              -> JSON / CSV
    pipeline      runs io -> results for one hologram (equivalent to legacy main.py)
"""

__version__ = "0.1.0"
