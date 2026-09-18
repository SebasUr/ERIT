"""Acquisition parameters of each microscope / dataset.

dxy (object-plane pixel size) = pixel_size / magnification.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Acquisition:
    wavelength: float      # um
    pixel_size: float      # camera pixel pitch, um
    magnification: float

    @property
    def dxy(self):
        return self.pixel_size / self.magnification


# EAFIT setup used for the cow videos (legacy main.m / main.py; VideosToFrames writes 0.633)
EAFIT_COW = Acquisition(wavelength=0.632, pixel_size=3.75, magnification=40)

# Public human RBC dataset, Castaneda, Trujillo & Doblas, Data in Brief 54 (2024) 110424
HUMAN_RBC_DATASET = Acquisition(wavelength=0.532, pixel_size=5.86, magnification=40)
