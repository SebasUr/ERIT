import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.show = lambda *a, **k: None

LEGACY = Path(__file__).resolve().parents[1] / "legacy" / "RBC_analize"


@pytest.fixture(scope="session")
def legacy():
    """Import the untouched legacy modules."""
    sys.path.insert(0, str(LEGACY))
    import data_management
    import functions_hematological
    yield functions_hematological, data_management
    sys.path.remove(str(LEGACY))


@pytest.fixture
def synthetic_field(tmp_path):
    """
    A 256x256 complex field with 5 RBC-like cells (flat background at phase 0,
    cells with ~1.5 rad bumps), saved as the MATLAB reconstruction does: variable 'output'.
    """
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:256, :256]
    phase = np.zeros((256, 256))
    for cy, cx in [(60, 60), (60, 190), (140, 120), (200, 50), (200, 200)]:
        r = np.hypot(yy - cy, xx - cx) / 27.0          # radius ~27 px ~ 2.5 um at dxy=0.094
        cell = np.clip(1 - r**2, 0, None)
        phase += 1.5 * cell**0.5 * (0.7 + 0.3 * r**2)  # rim thicker than centre
    phase += rng.normal(0, 0.03, phase.shape)
    field = np.exp(1j * phase)
    path = tmp_path / "complex_field_000001_Hologram_video01_frame_000001.mat"
    savemat(path, {"output": field})
    return path, phase
