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
