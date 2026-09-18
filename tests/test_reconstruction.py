"""Vortex-Legendre port on a simulated off-axis hologram."""
import numpy as np

from erit.config import HUMAN_RBC_DATASET as ACQ
from erit.phase import quantitative_phase
from erit.pipeline import analyze_phase
from erit.reconstruction import vortex_legendre


def simulated_hologram(true_phase, carrier=(-70, 60)):
    N = true_phase.shape[0]
    y, x = np.mgrid[:N, :N]
    ref = np.exp(2j * np.pi * (carrier[1] * x + carrier[0] * y) / N)
    obj = np.exp(1j * true_phase)
    return np.abs(obj + ref) ** 2


def test_reconstruction_recovers_phase(synthetic_field):
    _, true_phase = synthetic_field
    field = vortex_legendre(simulated_hologram(true_phase), ACQ.wavelength, ACQ.pixel_size)
    q = quantitative_phase(field)
    inner = (slice(20, -20), slice(20, -20))   # spatial filter blurs the borders
    assert np.corrcoef(q[inner].ravel(), true_phase[inner].ravel())[0, 1] > 0.95


def test_full_chain_on_simulated_hologram(synthetic_field):
    _, true_phase = synthetic_field
    field = vortex_legendre(simulated_hologram(true_phase), ACQ.wavelength, ACQ.pixel_size)
    a = analyze_phase(quantitative_phase(field), ACQ.dxy, ACQ.wavelength)
    assert a["segmentation"]["num_rbcs"] == 5
