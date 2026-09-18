
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from skimage.restoration import unwrap_phase as _skimage_unwrap_phase
from scipy.ndimage import distance_transform_edt, maximum_filter, map_coordinates
from skimage.segmentation import watershed
from skimage import measure, morphology
from unwrapping import phase_unwrap as _custom_unwrap_phase


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


def show_phase_results(results):
    """
    Display the wrapped phase and the available unwrapped phase results.
    The wrapped phase is displayed in the range [-pi, pi].
    """
    phase = results["phase"]
    phase_skimage = results["phase_unwrapped_skimage"]
    phase_custom = results["phase_unwrapped_custom"]

    images = [phase]
    titles = ["Wrapped Phase"]
    is_wrapped = [True]

    if phase_skimage is not None:
        images.append(phase_skimage)
        titles.append("Unwrapped Phase - Skimage")
        is_wrapped.append(False)

    if phase_custom is not None:
        images.append(phase_custom)
        titles.append("Unwrapped Phase - Custom")
        is_wrapped.append(False)

    # Find common limits for all available unwrapped phase images
    unwrapped_images = [
        image for image, wrapped in zip(images, is_wrapped)
        if not wrapped
    ]

    if unwrapped_images:
        unwrap_vmin = min(np.min(image) for image in unwrapped_images)
        unwrap_vmax = max(np.max(image) for image in unwrapped_images)

    n_images = len(images)

    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(5 * n_images, 4)
    )

    if n_images == 1:
        axes = [axes]

    for ax, image, title, wrapped in zip(
        axes, images, titles, is_wrapped
    ):

        if wrapped:
            im = ax.imshow(image, cmap="gray", vmin=-np.pi, vmax=np.pi)
        else:
            im = ax.imshow(image, cmap="gray", vmin=unwrap_vmin, vmax=unwrap_vmax)

        ax.set_title(title)
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Phase [rad]")

        if wrapped:
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels(
                [r"$-\pi$", "0", r"$\pi$"]
            )

    plt.tight_layout()
    plt.show()


def segment_rbc(phase_image, dxy, method="otsu", manual_threshold=0.5,
                 min_area_um2=50, max_area_um2=None, separate_touching=True,
                 polarity="bright"):
    """
    Segment individual RBCs from a wrapped or unwrapped phase image.

    method : str, optional (default "otsu")
        Thresholding method:
            "otsu"     -> automatic Otsu thresholding.
            "manual"   -> user-defined threshold.
            "adaptive" -> adaptive Gaussian thresholding.

    min_area_um2 : int, optional (default 50)
        Minimum RBC area, in um^2.

    max_area:um2 : int or None, optional (default None)
        Maximum RBC area, in um^2.
        If None, no upper-area restriction is applied.

    separate_touching : bool, optional (default True)
        If True, apply distance-transform and watershed segmentation
        to separate touching RBCs.

    polarity : str, optional (default "bright")
        Defines the expected RBC contrast:
            "bright" -> RBCs have higher values than the background.
            "dark"   -> RBCs have lower values than the background.

    outputus:
        'binary_mask': Boolean mask containing only accepted RBCs.
        'labeled_mask': Integer mask where each RBC has a unique label.
        'num_rbcs': Number of detected RBCs.
        'rbcs': List containing basic information for each RBC label, center position, pixel area, equivalent diameter, and bounding box.
        'threshold_value': Threshold used on the normalized 8-bit image.
        'method': Thresholding method used.
        'polarity': RBC polarity used during segmentation.
    """
    phase_image = np.asarray(phase_image, dtype=np.float64)

    if polarity not in ("bright", "dark"):
        raise ValueError(f"Unknown polarity '{polarity}'. Use 'bright' or 'dark'.")

    # Normalize phase image to 8-bit for thresholding
    p_min, p_max = phase_image.min(), phase_image.max()
    if p_max - p_min < 1e-12:
        raise ValueError("phase_image has near-zero dynamic range and cannot be segmented.")
    image_8u = ((phase_image - p_min) / (p_max - p_min) * 255).astype(np.uint8)

    threshold_type = cv2.THRESH_BINARY if polarity == "bright" else cv2.THRESH_BINARY_INV

    # Convert physical area limits (um^2) to pixels using dxy (um/px)
    pixel_area = dxy ** 2
    min_area_px = int(np.ceil(min_area_um2 / pixel_area))
    max_area_px = int(np.floor(max_area_um2 / pixel_area)) if max_area_um2 is not None else None

    # --- Thresholding ---
    if method == "otsu":
        threshold_value, binary = cv2.threshold(image_8u, 0, 255, threshold_type + cv2.THRESH_OTSU)
    elif method == "manual":
        if not 0 <= manual_threshold <= 1:
            raise ValueError("manual_threshold must be between 0 and 1.")
        threshold_value = manual_threshold * 255
        _, binary = cv2.threshold(image_8u, threshold_value, 255, threshold_type)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(image_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       threshold_type, 15, 5)
        threshold_value = None
    else:
        raise ValueError(f"Unknown thresholding method '{method}'. Use 'otsu', 'manual', or 'adaptive'.")

    binary_mask = binary > 0

    # Remove small noise / small holes (below the min-area threshold)
    binary_mask = morphology.remove_small_objects(binary_mask, max_size=max(min_area_px - 1, 0))
    binary_mask = morphology.remove_small_holes(binary_mask, max_size=max(min_area_px - 1, 0))

    # --- Separate touching RBCs using distance transform + watershed ---
    if separate_touching and np.any(binary_mask):
        distance = distance_transform_edt(binary_mask)
        # Window size for local-maxima search, derived from the minimum RBC
        # area so it scales with real cell size instead of a hardcoded pixel count.
        min_diameter_px = max(int(round(2 * np.sqrt(min_area_px / np.pi))), 3)
        local_maxima = maximum_filter(distance, size=min_diameter_px) == distance
        local_maxima = local_maxima & (distance > 5)
        markers = measure.label(local_maxima)

        if markers.max() > 1:
            candidate_labels = watershed(-distance, markers, mask=binary_mask)
        else:
            candidate_labels = measure.label(binary_mask, connectivity=2)
    else:
        candidate_labels = measure.label(binary_mask, connectivity=2)

    # --- Filter by area and build the final labeled mask ---
    regions = measure.regionprops(candidate_labels)

    labeled_mask = np.zeros_like(candidate_labels, dtype=np.int32)
    rbcs = []
    new_label = 1
    for region in regions:
        area = region.area
        if area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue

        rbc_mask = candidate_labels == region.label
        labeled_mask[rbc_mask] = new_label

        rbcs.append({
            "label": new_label,
            "centroid": region.centroid,  # (row, col)
            "area_px": int(area),
            "area_um2": float(area * pixel_area),
            "equivalent_diameter_px": float(region.equivalent_diameter_area),
            "bbox": region.bbox,
        })
        new_label += 1

    binary_mask = labeled_mask > 0
    num_rbcs = len(rbcs)

    return {
        "binary_mask": binary_mask,
        "labeled_mask": labeled_mask,
        "num_rbcs": num_rbcs,
        "rbcs": rbcs,
        "threshold_value": threshold_value,
        "method": method,
        "polarity": polarity,
        "min_area_px": min_area_px,
        "max_area_px": max_area_px,
    }


def show_segmentation_results(phase_image, segmentation_result):
    """
    Display the main results of the RBC segmentation process.

    Shows:
        1. Original phase image.
        2. Final binary RBC mask.
        3. Detected RBCs with contours and labels.
    """
    phase_image = np.asarray(phase_image)

    binary_mask = segmentation_result["binary_mask"]
    labeled_mask = segmentation_result["labeled_mask"]
    num_rbcs = segmentation_result["num_rbcs"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original phase image
    im = axes[0].imshow(
        phase_image,
        cmap="gray"
    )
    axes[0].set_title("Phase Image")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], label="Phase [rad]")

    # Binary mask
    axes[1].imshow(
        binary_mask,
        cmap="gray"
    )
    axes[1].set_title("Binary RBC Mask")
    axes[1].axis("off")

    # Detected RBCs
    axes[2].imshow(
        phase_image,
        cmap="gray"
    )

    for label in range(1, num_rbcs + 1):

        # Individual RBC mask
        rbc_mask = labeled_mask == label

        # Find RBC contour
        contours = measure.find_contours(
            rbc_mask,
            level=0.5
        )

        for contour in contours:
            axes[2].plot(
                contour[:, 1],
                contour[:, 0],
                linewidth=1.5
            )

        # Temporary center only for displaying the RBC label
        y_coords, x_coords = np.where(rbc_mask)

        if len(x_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            axes[2].text(
                center_x,
                center_y,
                str(label),
                fontsize=8,
                ha="center",
                va="center"
            )

    axes[2].set_title(
        f"Detected RBCs: {num_rbcs}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def extract_rbc_geometry(labeled_mask, dxy):
    """
    Extract geometrical properties from individually labeled RBCs.

    outputs:
        Dictionary containing:
            'num_rbcs': Number of analyzed RBCs.
            'rbcs': List with the geometrical properties of each RBC.
            'area_statistics': Mean, standard deviation, minimum, and maximum RBC area.
        """

    labeled_mask = np.asarray(labeled_mask)
    if labeled_mask.ndim != 2:
        raise ValueError("labeled_mask must be a 2D array.")

    if dxy <= 0:
        raise ValueError("dxy must be greater than zero.")

    regions = measure.regionprops(labeled_mask)

    rbcs = []
    for region in regions:
        # Centroid returned by regionprops is (row, column) = (y, x)
        center_y_px, center_x_px = region.centroid

        # Area
        area_px = float(region.area)
        area_um2 = area_px * (dxy ** 2)

        # Equivalent diameter and radius
        equivalent_diameter_px = float(
            region.equivalent_diameter_area
        )

        equivalent_diameter_um = (
            equivalent_diameter_px * dxy
        )

        equivalent_radius_um = (
            equivalent_diameter_um / 2
        )

        # Major and minor axes
        major_axis_um = (
            float(region.axis_major_length) * dxy
        )

        minor_axis_um = (
            float(region.axis_minor_length) * dxy
        )

        # Orientation
        orientation_rad = float(region.orientation)
        orientation_deg = np.degrees(orientation_rad)

        # Bounding box
        min_row, min_col, max_row, max_col = region.bbox

        bbox_px = {
            "x_min": int(min_col),
            "y_min": int(min_row),
            "x_max": int(max_col),
            "y_max": int(max_row),
        }

        bbox_um = {
            "x_min": float(min_col * dxy),
            "y_min": float(min_row * dxy),
            "x_max": float(max_col * dxy),
            "y_max": float(max_row * dxy),
        }

        rbcs.append({
            "rbc_id": int(region.label),

            "center_x_px": float(center_x_px),
            "center_y_px": float(center_y_px),

            "center_x_um": float(center_x_px * dxy),
            "center_y_um": float(center_y_px * dxy),

            "area_px": area_px,
            "area_um2": area_um2,

            "equivalent_radius_um": equivalent_radius_um,
            "equivalent_diameter_um": equivalent_diameter_um,

            "major_axis_um": major_axis_um,
            "minor_axis_um": minor_axis_um,

            "orientation_rad": orientation_rad,
            "orientation_deg": orientation_deg,

            "bbox_px": bbox_px,
            "bbox_um": bbox_um,
        })

    # Area statistics for this hologram
    areas = np.array(
        [rbc["area_um2"] for rbc in rbcs],
        dtype=float
    )

    if len(areas) > 0:
        area_statistics = {
            "mean_um2": float(np.mean(areas)),
            "std_um2": float(np.std(areas)),
            "min_um2": float(np.min(areas)),
            "max_um2": float(np.max(areas)),
        }
    else:
        area_statistics = {
            "mean_um2": None,
            "std_um2": None,
            "min_um2": None,
            "max_um2": None,
        }
    return {
        "num_rbcs": len(rbcs),
        "rbcs": rbcs,
        "area_statistics": area_statistics,
    }


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


def show_phase_profiles_overview(phase_image, labeled_mask, profile_result):
    """
    Display the phase image with RBC contours, RBC IDs, and the two
    phase-profile lines used for each valid RBC.

    outputs: matplotlib.figure.Figure Overview figure.
    """
    phase_image = np.asarray(phase_image)
    labeled_mask = np.asarray(labeled_mask)
    rbcs = (profile_result["rbcs"])
    num_excluded = (profile_result["num_excluded"])
    regions = measure.regionprops(labeled_mask)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 9))
    image = ax.imshow(phase_image, cmap="viridis")

    # Draw RBC contours and IDs
    for region in regions:
        label = int(region.label)
        contours = measure.find_contours(labeled_mask == label, 0.5)

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.8)

        cy, cx = (
            region.centroid)
        ax.text(
            cx, cy, str(label), fontsize=7, ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.15", "alpha": 0.5}
        )

    # Draw the two phase-profile lines
    for rbc in rbcs:
        profile_1 = (rbc["profile_1"])
        profile_2 = (rbc["profile_2"])
        coords_1 = (profile_1["coords_rc"])
        coords_2 = (profile_2["coords_rc"])

        # Profile 1
        ax.plot(coords_1[:, 1], coords_1[:, 0], linewidth=1.2)

        # Profile 2
        ax.plot(coords_2[:, 1], coords_2[:, 0], linewidth=1.2)

    # Figure settings
    ax.set_title("Phase Profile Measurements\n" f"{profile_result['num_rbcs']} RBCs analyzed, " f"{num_excluded}excluded")
    ax.axis("off")

    fig.colorbar(image, ax=ax, label="Phase [rad]", fraction=0.046)
    fig.tight_layout()
    plt.show()

    return fig


def plot_phase_profiles(profile_result, profiles_per_figure=20):
    """
    Display the phase profiles of all valid RBCs in batches.
    Each subplot shows: Profile 1, Profile 2
        - Maximum and minimum of both Profiles, High- and low-region mean phase values
        - delta_phase_1, delta_phase_2, Mean delta_phase

    outcomes:
         List containing the generated matplotlib Figure objects.
    """
    if profiles_per_figure < 1:
        raise ValueError("profiles_per_figure must be greater than zero.")

    rbcs = (profile_result["rbcs"])
    figures = []

    if len(rbcs) == 0:
        return figures

    # Number of batches
    num_batches = int(np.ceil(len(rbcs) / profiles_per_figure))

    # Generate each batch
    for batch_idx in range(num_batches):
        start = (batch_idx * profiles_per_figure)
        end = min(start + profiles_per_figure, len(rbcs))
        batch_rbcs = (rbcs[start:end])

        # Figure organization
        ncols = 4
        nrows = int(np.ceil(len(batch_rbcs) / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.7 * nrows), squeeze=False)
        axes = (axes.flatten())

        # Plot each RBC
        for ax, rbc in zip(axes, batch_rbcs):
            p1 = (rbc["profile_1"])
            p2 = (rbc["profile_2"])

            # Plot both phase profiles
            line_1, = ax.plot(p1["distance_um"], p1["phase"], label="Profile 1")
            line_2, = ax.plot(p2["distance_um"], p2["phase"], label="Profile 2")

            # Find extrema positions - Profile 1
            idx_max_1 = int(np.argmax(p1["phase"]))
            idx_min_1 = int(np.argmin(p1["phase"]))

            # Maximum - Profile 1
            ax.plot(p1["distance_um"][idx_max_1], p1["phase"][idx_max_1], "o", color=line_1.get_color())

            # Minimum - Profile 1
            ax.plot(p1["distance_um"][idx_min_1], p1["phase"][idx_min_1], "v", color=line_1.get_color())

            # Find extrema positions - Profile 2
            idx_max_2 = int(np.argmax(p2["phase"]))
            idx_min_2 = int(np.argmin(p2["phase"]))

            # Maximum - Profile 2
            ax.plot(p2["distance_um"][idx_max_2], p2["phase"][idx_max_2], "o", color=line_2.get_color())

            # Minimum - Profile 2
            ax.plot(p2["distance_um"][idx_min_2],p2["phase"][idx_min_2], "v", color=line_2.get_color())

            # Mean high- and low-phase regions - Profile 1
            ax.axhline(p1["high_region_mean_phase"], linestyle="--", linewidth=0.8, color=line_1.get_color())
            ax.axhline(p1["low_region_mean_phase"], linestyle=":", linewidth=0.8, color=line_1.get_color())

            # Mean high- and low-phase regions - Profile 2
            ax.axhline(p2["high_region_mean_phase"], linestyle="--", linewidth=0.8, color=line_2.get_color())
            ax.axhline(p2["low_region_mean_phase"], linestyle=":", linewidth=0.8, color=line_2.get_color())

            # Plot information
            ax.set_title(
                f"RBC {rbc['rbc_id']}\n"
                rf"$\Delta\phi_1$ = "
                f"{rbc['delta_phase_1']:.3f} rad | "
                rf"$\Delta\phi_2$ = "
                f"{rbc['delta_phase_2']:.3f} rad\n"
                rf"$\overline{{\Delta\phi}}$ = "
                f"{rbc['delta_phase']:.3f} rad",
                fontsize=9
            )

            ax.set_xlabel("Distance [um]")
            ax.set_ylabel("Phase [rad]")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)

        # Hide unused axes in the last batch
        for ax in axes[len(batch_rbcs):]:
            ax.axis("off")

        # Figure title
        fig.suptitle("RBC Phase Profiles " f"({start + 1}-{end})", fontsize=14)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figures.append(fig)

    plt.show()

    return figures


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