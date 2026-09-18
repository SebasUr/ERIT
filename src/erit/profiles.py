"""Two centre-crossing phase profiles per RBC and their phase difference (delta phi)."""

import numpy as np
from scipy.ndimage import map_coordinates
from skimage import measure


def phase_profiles(phase_image, labeled_mask, dxy, background_margin_px=6, extension_search_factor=3.0, num_angle_candidates=36,
    region_fraction=0.05, step_px=0.5):
    """
    Extract two center-crossing phase profiles per RBC and calculate
    the corresponding delta phase measurements.
        delta_phase_1 -> Profile 1
        delta_phase_2 -> Profile 2

    The final phase shift is:
        delta_phase = (delta_phase_1 + delta_phase_2) / 2


    background_margin_px : int, optional (default 6)
        Minimum amount of background required at each end of a
        valid profile.

    extension_search_factor : float, optional (default 3.0)
        Maximum profile-search distance relative to the equivalent
        RBC radius.

    num_angle_candidates : int, optional (default 36)
        Number of candidate orientations evaluated between 0 and pi.

    region_fraction : float, optional (default 0.05)
        Fraction of the phase dynamic range used to define the
        high- and low-phase regions.

    outputs:
        'num_rbcs': Number of RBCs with two valid profiles.

        'num_excluded': Number of RBCs excluded because two valid profiles could not be obtained.

        'rbcs': Phase-profile information for each valid RBC.

        'statistics': Mean, standard deviation, minimum, and maximum values across all valid RBCs.
    """
    phase_image = np.asarray(phase_image, dtype=np.float64)
    labeled_mask = np.asarray(labeled_mask)

    # Validate inputs
    if dxy <= 0:
        raise ValueError("dxy must be greater than zero.")

    if not 0 < region_fraction < 1:
        raise ValueError("region_fraction must be between 0 and 1.")

    if step_px <= 0:
        raise ValueError("step_px must be greater than zero.")

    regions = measure.regionprops(labeled_mask)

    rbcs = []
    num_excluded = 0
    angle_candidates = np.linspace(0, np.pi, num_angle_candidates, endpoint=False)

    # Trace one direction from the RBC centroid
    def _trace_direction(cy, cx, label, direction, sign, max_extension_px):
        coords = []
        distances = []
        background_count = 0
        t = 0.0

        while abs(t) <= max_extension_px:
            t_next = t + sign * step_px
            r = cy + t_next * direction[0]
            c = cx + t_next * direction[1]
            r_round = int(round(r))
            c_round = int(round(c))

            # Image boundary reached
            if (
                r_round < 0
                or r_round >= labeled_mask.shape[0]
                or c_round < 0
                or c_round >= labeled_mask.shape[1]
            ):
                return coords, distances, False
            lbl = labeled_mask[r_round, c_round]

            # Another RBC was reached
            if lbl != 0 and lbl != label:
                return coords, distances, False
            coords.append((r, c))
            distances.append(t_next)

            # Enough background reached
            if lbl == 0:
                background_count += 1
                if background_count >= background_margin_px:
                    return coords, distances, True
            t = t_next

        return coords, distances, False

    # Angular distance between two directions
    def _angular_distance(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    # Analyze one phase profile
    def _analyze_profile(phase_values):
        phase_values = np.asarray(phase_values, dtype=float)
        max_phase = float(np.max(phase_values))
        min_phase = float(np.min(phase_values))
        phase_range = (max_phase - min_phase)

        # Regions close to the maximum and minimum phase values
        high_threshold = (max_phase - region_fraction * phase_range)
        low_threshold = (min_phase + region_fraction * phase_range)

        high_mask = (phase_values >= high_threshold)
        low_mask = (phase_values <= low_threshold)

        high_region_mean_phase = float(np.mean(phase_values[high_mask]))
        low_region_mean_phase = float(np.mean(phase_values[low_mask]))

        delta_phase = float(abs(high_region_mean_phase - low_region_mean_phase))

        return {
            "max_phase": max_phase,
            "min_phase": min_phase,
            "high_threshold": float(high_threshold),
            "low_threshold": float(low_threshold),
            "high_region_mean_phase": high_region_mean_phase,
            "low_region_mean_phase": low_region_mean_phase,
            "delta_phase": delta_phase,
        }

    # Analyze each RBC
    for region in regions:
        label = int(region.label)

        cy, cx = (region.centroid)

        radius_px = (region.equivalent_diameter_area / 2.0)

        max_extension_px = max(radius_px * extension_search_factor,
            radius_px + background_margin_px + 2)

        # Find all valid profile directions
        valid_angles = []
        for theta in angle_candidates:
            direction = np.array([np.sin(theta), np.cos(theta)])

            coords_pos, dist_pos, ok_pos = (
                _trace_direction( cy, cx, label, direction, +1, max_extension_px))

            coords_neg, dist_neg, ok_neg = (
                _trace_direction(cy, cx, label, direction, -1, max_extension_px))

            if ok_pos and ok_neg:
                coords_neg = list(reversed(coords_neg))
                dist_neg = list(reversed(dist_neg))
                coords_full = np.array(coords_neg + coords_pos)
                dist_full = np.array(dist_neg + dist_pos)

                valid_angles.append((theta, coords_full, dist_full))

        # Two valid profiles are required
        if len(valid_angles) < 2:
            num_excluded += 1
            continue

        # Profile 1
        theta_1, coords_1, dist_1 = (valid_angles[0])

        # Profile 2
        # Choose the valid direction closest to perpendicular to Profile 1.
        best_score = -1
        theta_2 = None
        coords_2 = None
        dist_2 = None

        for theta, coords, dist in valid_angles[1:]:
            score = _angular_distance(theta, theta_1)

            if score > best_score:
                best_score = score
                theta_2 = theta
                coords_2 = coords
                dist_2 = dist

        # Extract phase values using bilinear interpolation
        phase_1 = map_coordinates(phase_image, [coords_1[:, 0], coords_1[:, 1]], order=1)
        phase_2 = map_coordinates(phase_image, [coords_2[:, 0], coords_2[:, 1]], order=1)

        # Analyze both profiles independently
        analysis_1 = _analyze_profile(phase_1)
        analysis_2 = _analyze_profile(phase_2)

        delta_phase_1 = (analysis_1["delta_phase"])
        delta_phase_2 = (analysis_2["delta_phase"])

        # Mean phase shift from both profiles
        delta_phase = float((delta_phase_1 + delta_phase_2) / 2.0)

        # Store Profile 1
        profile_1 = {
            "distance_um": (dist_1 - dist_1.min()) * dxy, "phase": phase_1,
            "coords_rc": coords_1, "angle_rad": float(theta_1),
            "max_phase": analysis_1["max_phase"], "min_phase": analysis_1["min_phase"],
            "high_threshold": analysis_1["high_threshold"], "low_threshold": analysis_1["low_threshold"],
            "high_region_mean_phase": analysis_1["high_region_mean_phase"],
            "low_region_mean_phase": analysis_1["low_region_mean_phase"], "delta_phase": delta_phase_1,
        }

        # Store Profile 2
        profile_2 = {
            "distance_um": (dist_2 - dist_2.min()) * dxy, "phase": phase_2,
            "coords_rc": coords_2, "angle_rad": float(theta_2),
            "max_phase": analysis_2["max_phase"], "min_phase":analysis_2["min_phase"],
            "high_threshold": analysis_2["high_threshold"], "low_threshold": analysis_2["low_threshold"],
            "high_region_mean_phase": analysis_2["high_region_mean_phase"],
            "low_region_mean_phase": analysis_2["low_region_mean_phase"], "delta_phase": delta_phase_2,
        }

        # Store RBC information
        rbcs.append({
            "rbc_id": label, "centroid_rc": (float(cy), float(cx)),
            "profile_1": profile_1, "profile_2": profile_2,
            "delta_phase_1": delta_phase_1, "delta_phase_2": delta_phase_2,
            "delta_phase": delta_phase,
        })

    # Statistics across all valid RBCs
    def _stats(key):
        values = np.array([rbc[key] for rbc in rbcs], dtype=float)

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

    statistics = {
        "delta_phase_1": _stats("delta_phase_1"),
        "delta_phase_2": _stats("delta_phase_2"),
        "delta_phase": _stats("delta_phase"),
    }

    return {
        "num_rbcs": len(rbcs),
        "num_excluded": num_excluded,
        "rbcs": rbcs,
        "statistics": statistics,
    }
