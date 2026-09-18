"""Reading reconstructed complex fields (.mat files written by the MATLAB reconstruction)."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def read_complex_field(file_path, wavelength, pixel_size, magnification,
                        field_key=None):
    """
    Read a complex optical field from a MATLAB file and package it with
    the acquisition parameters.

    outputs:
        Dictionary containing the complex field, acquisition parameters,
        field shape, and source file path.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Complex-field file not found: {file_path}")

    mat = loadmat(str(file_path))

    # Drop MATLAB metadata keys (__header__, __version__, __globals__)
    data_keys = [k for k in mat.keys() if not k.startswith("__")]

    complex_field = None

    if field_key is not None:
        if field_key not in mat:
            raise KeyError(
                f"'{field_key}' not found in file. Available keys: {data_keys}"
            )
        complex_field = np.asarray(mat[field_key]).astype(np.complex128)

    else:
        # 1. A variable that is already a 2D complex array
        for key in data_keys:
            arr = np.asarray(mat[key])
            if np.iscomplexobj(arr) and arr.ndim == 2:
                complex_field = arr.astype(np.complex128)
                break

        # 2. Separate real and imaginary parts
        if complex_field is None:
            real_key = next(
                (k for k in data_keys if k.lower() in ("real", "re")), None
            )
            imag_key = next(
                (k for k in data_keys if k.lower() in ("imag", "im")), None
            )
            if real_key and imag_key:
                real_part = np.asarray(mat[real_key])
                imag_part = np.asarray(mat[imag_key])
                complex_field = (real_part + 1j * imag_part).astype(np.complex128)

        # 3. Separate amplitude and phase
        if complex_field is None:
            amp_key = next(
                (k for k in data_keys if k.lower() in ("amp", "amplitude")), None
            )
            phase_key = next(
                (k for k in data_keys if k.lower() in ("phase", "phi")), None
            )
            if amp_key and phase_key:
                amp = np.asarray(mat[amp_key])
                phase = np.asarray(mat[phase_key])
                complex_field = (amp * np.exp(1j * phase)).astype(np.complex128)

        if complex_field is None:
            raise ValueError(
                "Could not auto-detect the complex field in this file. "
                f"Available keys: {data_keys}. "
                "Pass field_key explicitly to select the correct variable."
            )

    # Effective pixel size in the object plane
    dxy = pixel_size / magnification

    # All scalar parameters saved together in one vector
    parameters = {
        "wavelength": wavelength,
        "pixel_size": pixel_size,
        "magnification": magnification,
        "dxy": dxy,
    }

    return {
        "complex_field": complex_field,
        "parameters": parameters,
        "shape": complex_field.shape,
        "source_file": str(file_path.resolve()),
    }
