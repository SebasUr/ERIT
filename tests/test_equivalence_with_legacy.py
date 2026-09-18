"""The erit package must give exactly the same numbers as the legacy code."""
import json

import numpy as np

from erit.pipeline import analyze_complex_field

PARAMS = dict(wavelength=0.632, pixel_size=3.75, magnification=40)


def run_legacy_main(fha, dm, mat_path, out_dir):
    """Same calls as legacy/RBC_analize/main.py, without the figures."""
    result = fha.read_complex_field(file_path=str(mat_path), **PARAMS)
    phase_result = fha.extract_phase(result["complex_field"], unwrap_method="skimage", show=False)
    dxy = result["parameters"]["dxy"]
    seg = fha.segment_rbc(phase_result["phase"], dxy=dxy, method="otsu", min_area_um2=10,
                          max_area_um2=None, separate_touching=True, polarity="bright")
    geo = fha.extract_rbc_geometry(seg["labeled_mask"], dxy=dxy)
    prof = fha.phase_profiles(phase_result["phase"], seg["labeled_mask"], dxy=dxy)
    hv = fha.calculate_rbc_height_volume(phase_image=phase_result["phase"], labeled_mask=seg["labeled_mask"],
                                         profile_result=prof, wavelength=0.632, delta_n=0.083, dxy=dxy)
    res = dm.build_hologram_result(result["source_file"], result["parameters"], seg, geo, prof, hv,
                                   {"unwrap_method": "skimage", "segmentation_method": "otsu",
                                    "min_area_um2": 10, "max_area_um2": None, "separate_touching": True,
                                    "polarity": "bright", "delta_n": 0.083})
    return dm.hologram_json_to_dataframe(dm.save_hologram_json(res, out_dir)), res


def test_pipeline_matches_legacy_main(legacy, synthetic_field, tmp_path):
    fha, dm = legacy
    mat_path, _ = synthetic_field

    df_legacy, res_legacy = run_legacy_main(fha, dm, mat_path, tmp_path / "legacy")
    res_new, df_new, json_path, _ = analyze_complex_field(mat_path, **PARAMS, output_directory=tmp_path / "new")

    assert len(df_new) == len(df_legacy) > 0
    np.testing.assert_array_equal(df_new.to_numpy(dtype=float), df_legacy.to_numpy(dtype=float))
    assert res_new["statistics"] == res_legacy["statistics"]
    assert json.load(open(json_path))["summary"] == res_legacy["summary"]


def test_synthetic_cells_are_found(synthetic_field, tmp_path):
    mat_path, _ = synthetic_field
    res, df, _, _ = analyze_complex_field(mat_path, **PARAMS, output_directory=tmp_path)
    assert res["summary"]["num_segmented_rbcs"] == 5
    assert df["Equivalent_Diameter_um"].between(4, 6).all()   # 2 * 27 px * 0.09375 um/px ~ 5 um
