"""Convert phase to cell height and volume given the refractive-index difference."""

import numpy as np


def calculate_rbc_height_volume(phase_image, labeled_mask, profile_result, wavelength, delta_n, dxy):
    """
    Calculate representative RBC heights from phase profiles and RBC volumes from the full phase map.
    Representative height: Each RBC height is calculated from the two phase-profile
            h1 = wavelength * delta_phase_1 / (2*pi*delta_n)
            h2 = wavelength * delta_phase_2 / (2*pi*delta_n)
            representative_height = (h1 + h2) / 2

    Volume: The mean background phase is first calculated using all pixels
        labeled as background. A pixel-wise phase difference is then

        calculated for each RBC: delta_phi(x,y) = phi(x,y) - mean_background_phase

        The corresponding height map is: h(x,y) = wavelength * delta_phi(x,y) / (2*pi*delta_n)

        Finally, the RBC volume is calculated as: volume = sum(h(x,y) * dxy^2)

    delta_n : float
        Refractive-index difference between the RBC and surrounding
        medium: delta_n = n_RBC - n_medium

    outputs:
        'background_phase': Mean phase of all background pixels.
        'rbcs': List containing, for each RBC:
                rbc_id, height_profile_1_um, height_profile_2_um, representative_height_um, volume_um3

        'height_statistics': Mean, standard deviation, minimum, and maximum of the representative RBC heights.

        'volume_statistics': Mean, standard deviation, minimum, and maximum RBC volume.

        'height_map':Full pixel-wise height map. Background pixels are zero.
    """
    phase_image = np.asarray(phase_image, dtype=np.float64)

    labeled_mask = np.asarray(labeled_mask)

    # Validate inputs
    if wavelength <= 0:
        raise ValueError("wavelength must be greater than zero.")

    if dxy <= 0:
        raise ValueError("dxy must be greater than zero.")

    if abs(delta_n) < 1e-12:
        raise ValueError("delta_n must be different from zero.")

    # Mean background phase
    background_mask = (labeled_mask == 0)

    if not np.any(background_mask):
        raise ValueError("No background pixels were found in labeled_mask.")

    background_phase = float(np.mean(phase_image[background_mask]))

    # Conversion factor from phase to height
    phase_to_height = (wavelength / (2 * np.pi * delta_n))

    # Full delta-phase map relative to the mean background
    delta_phase_map = (phase_image - background_phase)

    # Full height map
    # Height values are retained only inside segmented RBCs.
    height_map = np.zeros_like(phase_image, dtype=np.float64)
    rbc_pixels = (labeled_mask > 0)

    height_map[rbc_pixels] = (delta_phase_map[rbc_pixels] * phase_to_height)

    # Pixel area in the object plane
    pixel_area_um2 = (dxy ** 2)

    # Organize phase-profile information by RBC ID
    profile_by_id = {int(rbc["rbc_id"]): rbc for rbc in profile_result["rbcs"]}
    rbcs = []

    # Process each RBC with valid phase profiles
    for rbc_id, profile_data in profile_by_id.items():
        # Check that this RBC also exists in the labeled mask
        rbc_mask = (labeled_mask == rbc_id)

        if not np.any(rbc_mask):
            continue

        # Representative height from Profile 1
        delta_phase_1 = float(profile_data["delta_phase_1"])
        height_profile_1_um = float(delta_phase_1 * phase_to_height)

        # Representative height from Profile 2
        delta_phase_2 = float(profile_data["delta_phase_2"])
        height_profile_2_um = float(delta_phase_2 * phase_to_height)

        # Mean representative height
        representative_height_um = float((height_profile_1_um + height_profile_2_um) / 2.0)

        # Volume from the full RBC height map
        rbc_height_values = (height_map[rbc_mask])
        volume_um3 = float(np.sum(rbc_height_values) * pixel_area_um2)

        # Store RBC information
        rbcs.append({
            "rbc_id": int(rbc_id),
            "height_profile_1_um": height_profile_1_um, "height_profile_2_um": height_profile_2_um,
            "representative_height_um": representative_height_um, "volume_um3": volume_um3,
        })

    # Statistics
    def _stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Representative height statistics
    representative_heights = [rbc["representative_height_um"] for rbc in rbcs]
    height_statistics = _stats(representative_heights)

    # Volume statistics
    volumes = [rbc["volume_um3"] for rbc in rbcs]
    volume_statistics = _stats(volumes)

    return {
        "background_phase": background_phase,
        "rbcs": rbcs,
        "height_statistics": height_statistics,
        "volume_statistics": volume_statistics,
        "height_map": height_map,
    }
