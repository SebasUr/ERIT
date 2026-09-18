"""RBC segmentation: threshold the phase image, split touching cells, filter by area."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage import measure, morphology
from skimage.segmentation import watershed


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
