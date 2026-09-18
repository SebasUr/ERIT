"""Behaviour that differs from the legacy code on purpose."""
import numpy as np
import pytest
from scipy.io import savemat

from erit import geometry, io, phase
from erit.pipeline import analyze_complex_field
from erit.quality import find_corresponding_complex_field


def test_read_complex_field_with_explicit_key(tmp_path, legacy):
    f = tmp_path / "f.mat"
    savemat(f, {"myfield": np.exp(1j * np.ones((8, 8))), "other": np.zeros((8, 8))})
    r = io.read_complex_field(f, 0.632, 3.75, 40, field_key="myfield")
    assert r["shape"] == (8, 8)
    fha, _ = legacy
    with pytest.raises(AttributeError):   # legacy bug: complex_field stays None
        fha.read_complex_field(str(f), 0.632, 3.75, 40, field_key="myfield")


def test_quality_finds_mat_written_by_main_m(tmp_path):
    (tmp_path / "complex_field_000003_Hologram_video45_frame_000003.mat").touch()
    (tmp_path / "complex_field_000013_Hologram_video45_frame_000013.mat").touch()
    found = find_corresponding_complex_field("phase_000003_Hologram_video45_frame_000003.png", tmp_path)
    assert found == "complex_field_000003_Hologram_video45_frame_000003.mat"


def test_border_flag():
    m = np.zeros((50, 50), int)
    m[0:10, 20:30] = 1       # touches top
    m[20:30, 20:30] = 2      # interior
    flags = {r["rbc_id"]: r["touches_border"] for r in geometry.extract_rbc_geometry(m, 0.1)["rbcs"]}
    assert flags == {1: True, 2: False}


@pytest.mark.parametrize("piston", [0.0, 2.8, -3.0])
@pytest.mark.parametrize("sign", [1, -1])
def test_quantitative_phase_recovers_cells(synthetic_field, piston, sign):
    _, true_phase = synthetic_field
    field = np.exp(1j * (sign * true_phase + piston))
    q = phase.quantitative_phase(field)
    assert np.corrcoef(q.ravel(), true_phase.ravel())[0, 1] > 0.99
    assert abs(np.median(q)) < 0.05


def test_legacy_wrapped_mode_breaks_when_background_is_near_pi(synthetic_field, tmp_path):
    """Why the default changed: with a piston of ~pi the legacy chain segments the background."""
    _, true_phase = synthetic_field
    f = tmp_path / "shifted.mat"
    savemat(f, {"output": np.exp(1j * (true_phase + 2.8))})
    kw = dict(wavelength=0.632, pixel_size=3.75, magnification=40)
    legacy_res, *_ = analyze_complex_field(f, **kw, output_directory=tmp_path / "a",
                                           processing={"phase_mode": "legacy_wrapped"})
    fixed_res, fixed_df, *_ = analyze_complex_field(f, **kw, output_directory=tmp_path / "b")
    assert fixed_res["summary"]["num_segmented_rbcs"] == 5
    assert legacy_res["summary"]["num_segmented_rbcs"] != 5
