"""Phase extraction from the complex field: wrapped phase and optional unwrapping."""

import numpy as np
from skimage.restoration import unwrap_phase as _skimage_unwrap_phase

from .plotting import show_phase_results
from .unwrapping import phase_unwrap as _custom_unwrap_phase


def extract_phase(complex_field, unwrap_method=None, show=True):
    """
    Compute the wrapped and optionally unwrapped phase from a complex optical field.

    unwrap_method : str or None, optional (default None)
        Which unwrapping method to use:
            None       -> only compute the wrapped phase.
            "skimage"  -> scikit-image's unwrap_phase.
            "custom"   -> custom weighted phase-unwrapping method.
            "both"     -> apply both skimage and custom methods.

    show : bool, optional (default True)
        If True, display the computed phase images.

    outputs:
        'phase': 2D numpy array, wrapped phase (rad)
        'phase_unwrapped_skimage': 2D numpy array or None
        'phase_unwrapped_custom': 2D numpy array or None
        'unwrap_method': selected unwrapping method
    """
    complex_field = np.asarray(complex_field)

    if not np.iscomplexobj(complex_field):
        raise ValueError("complex_field must be a complex-valued array.")

    # Wrapped phase
    phase = np.angle(complex_field)

    # Initialize unwrapped-phase outputs
    phase_unwrapped_skimage = None
    phase_unwrapped_custom = None

    if unwrap_method is None:
        pass

    elif unwrap_method == "skimage":
        phase_unwrapped_skimage = _skimage_unwrap_phase(phase)

    elif unwrap_method == "custom":
        phase_unwrapped_custom = _custom_unwrap_phase(
            phase,
            weight=np.abs(complex_field)
        )

    elif unwrap_method == "both":
        phase_unwrapped_skimage = _skimage_unwrap_phase(phase)

        phase_unwrapped_custom = _custom_unwrap_phase(
            phase,
            weight=np.abs(complex_field)
        )

    else:
        raise ValueError(
            f"Unknown unwrap_method '{unwrap_method}'. "
            "Use None, 'skimage', 'custom', or 'both'."
        )

    results = {
        "phase": phase,
        "phase_unwrapped_skimage": phase_unwrapped_skimage,
        "phase_unwrapped_custom": phase_unwrapped_custom,
        "unwrap_method": unwrap_method,
    }

    if show:
        show_phase_results(results)

    return results


def remove_piston(complex_field, n_iter=3, tol=0.6):
    """
    Bring the background phase to 0.

    The reconstruction (legacy main.m runs with NoPistonCompensation=true) leaves an arbitrary
    constant phase offset, so the background can sit near +-pi and wrap. The offset is estimated
    from the whole field, then refined using only background-like pixels (|phase| < tol).
    """
    U = np.asarray(complex_field) * np.exp(-1j * np.angle(np.sum(complex_field)))
    for _ in range(n_iter):
        bg = np.abs(np.angle(U)) < tol
        if bg.sum() < 100:
            break
        U = U * np.exp(-1j * np.angle(U[bg].sum()))
    return U


def quantitative_phase(complex_field):
    """
    Phase map ready for measurement: piston removed, unwrapped, background at 0 and cells positive.

    The sign of the reconstructed phase depends on which diffraction order (+1 or -1) the
    spatial filter picked. RBCs delay light (n_cell > n_medium) and cover less area than the
    background, so the correct sign is the one giving a positive-skewed histogram.
    """
    from scipy.stats import skew

    phase = _skimage_unwrap_phase(np.angle(remove_piston(complex_field)))
    phase = phase - np.median(phase)
    if skew(phase.ravel()) < 0:
        phase = -phase
    return phase - np.median(phase)
