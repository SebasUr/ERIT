"""End-to-end analysis of one hologram.

    analyze_complex_field(.mat)  same steps as legacy/RBC_analize/main.py
    analyze_hologram(image)      reconstruct with the Python Vortex-Legendre port, then analyse

phase_mode selects which phase image feeds segmentation / profiles / height / volume:
    "quantitative"    (default) piston removed + unwrapped + cells positive  -> erit.phase.quantitative_phase
    "legacy_wrapped"  the raw wrapped phase, exactly like legacy main.py. With it, cells whose
                      phase wraps, or a background near +-pi, make Otsu segment the background.
"""
from pathlib import Path

import numpy as np

from . import geometry, height_volume, io, phase, plotting, profiles, results, segmentation

DEFAULT_PROCESSING = {
    "unwrap_method": "skimage",
    "segmentation_method": "otsu",
    "min_area_um2": 10,
    "max_area_um2": None,
    "separate_touching": True,
    "polarity": "bright",
    "delta_n": 0.083,
    "phase_mode": "quantitative",
}


def phase_for_analysis(complex_field, phase_mode="quantitative", unwrap_method="skimage", show=False):
    if phase_mode == "quantitative":
        return phase.quantitative_phase(complex_field)
    if phase_mode == "legacy_wrapped":
        return phase.extract_phase(complex_field, unwrap_method=unwrap_method, show=show)["phase"]
    raise ValueError(f"Unknown phase_mode '{phase_mode}'. Use 'quantitative' or 'legacy_wrapped'.")


def analyze_phase(phase_image, dxy, wavelength, processing=None, show=False):
    """Segmentation -> geometry -> profiles -> height/volume on an already prepared phase map."""
    p = {**DEFAULT_PROCESSING, **(processing or {})}
    seg = segmentation.segment_rbc(phase_image, dxy=dxy, method=p["segmentation_method"],
                                   min_area_um2=p["min_area_um2"], max_area_um2=p["max_area_um2"],
                                   separate_touching=p["separate_touching"], polarity=p["polarity"])
    geo = geometry.extract_rbc_geometry(seg["labeled_mask"], dxy=dxy)
    prof = profiles.phase_profiles(phase_image, seg["labeled_mask"], dxy=dxy)
    hv = height_volume.calculate_rbc_height_volume(phase_image=phase_image, labeled_mask=seg["labeled_mask"],
                                                   profile_result=prof, wavelength=wavelength,
                                                   delta_n=p["delta_n"], dxy=dxy)
    if show:
        plotting.show_segmentation_results(phase_image, seg)
        plotting.show_phase_profiles_overview(phase_image, seg["labeled_mask"], prof)
        plotting.plot_phase_profiles(prof, profiles_per_figure=16)
    return {"segmentation": seg, "geometry": geo, "profiles": prof, "height_volume": hv, "processing": p}


def _save(source_file, parameters, analysis, output_directory):
    hologram_result = results.build_hologram_result(
        source_file=source_file, parameters=parameters, segmentation_result=analysis["segmentation"],
        geometry_result=analysis["geometry"], profile_result=analysis["profiles"],
        height_volume_result=analysis["height_volume"], processing_parameters=analysis["processing"])
    json_path = results.save_hologram_json(hologram_result, output_directory)
    df = results.hologram_json_to_dataframe(json_path)
    csv_path = results.export_dataframe_to_csv(df, json_path)
    return hologram_result, df, json_path, csv_path


def analyze_complex_field(mat_path, wavelength, pixel_size, magnification,
                          processing=None, output_directory="results", show=False):
    """
    Run the full analysis on one complex-field .mat file and write <name>.json and <name>.csv.

    outputs:
        (hologram_result dict, per-RBC DataFrame, json_path, csv_path)
    """
    p = {**DEFAULT_PROCESSING, **(processing or {})}
    field = io.read_complex_field(mat_path, wavelength=wavelength, pixel_size=pixel_size,
                                  magnification=magnification)
    phase_image = phase_for_analysis(field["complex_field"], p["phase_mode"], p["unwrap_method"], show)
    analysis = analyze_phase(phase_image, field["parameters"]["dxy"], wavelength, p, show)
    return _save(field["source_file"], field["parameters"], analysis, output_directory)


def analyze_hologram(hologram, acquisition, processing=None, output_directory=None, source_file="hologram"):
    """
    Reconstruct a raw hologram (2D array or image path) with the Vortex-Legendre port and analyse it.

    acquisition: erit.config.Acquisition. If output_directory is given, JSON/CSV are written
    and (hologram_result, df, json_path, csv_path) is returned; otherwise the analysis dict.
    """
    from .reconstruction import vortex_legendre

    if isinstance(hologram, (str, Path)):
        from PIL import Image
        source_file = str(Path(hologram).resolve())
        hologram = np.array(Image.open(hologram)).astype(float)

    p = {**DEFAULT_PROCESSING, **(processing or {})}
    field = vortex_legendre(hologram, acquisition.wavelength, acquisition.pixel_size)
    phase_image = phase_for_analysis(field, p["phase_mode"], p["unwrap_method"])
    analysis = analyze_phase(phase_image, acquisition.dxy, acquisition.wavelength, p)
    analysis["phase_image"] = phase_image
    if output_directory is None:
        return analysis
    parameters = {"wavelength": acquisition.wavelength, "pixel_size": acquisition.pixel_size,
                  "magnification": acquisition.magnification, "dxy": acquisition.dxy}
    return _save(source_file, parameters, analysis, output_directory)
