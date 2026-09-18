"""End-to-end analysis of one reconstructed hologram (same steps as legacy/RBC_analize/main.py)."""

from . import geometry, height_volume, io, phase, plotting, profiles, results, segmentation

DEFAULT_PROCESSING = {
    "unwrap_method": "skimage",
    "segmentation_method": "otsu",
    "min_area_um2": 10,
    "max_area_um2": None,
    "separate_touching": True,
    "polarity": "bright",
    "delta_n": 0.083,
}


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
    dxy = field["parameters"]["dxy"]

    phase_result = phase.extract_phase(field["complex_field"], unwrap_method=p["unwrap_method"], show=show)
    # NOTE: like legacy main.py, the *wrapped* phase is used for everything below.
    phase_image = phase_result["phase"]

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

    hologram_result = results.build_hologram_result(
        source_file=field["source_file"], parameters=field["parameters"], segmentation_result=seg,
        geometry_result=geo, profile_result=prof, height_volume_result=hv, processing_parameters=p)
    json_path = results.save_hologram_json(hologram_result, output_directory)
    df = results.hologram_json_to_dataframe(json_path)
    csv_path = results.export_dataframe_to_csv(df, json_path)
    return hologram_result, df, json_path, csv_path
