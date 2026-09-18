"""2D geometry of each segmented RBC (area, diameter, axes, orientation, bbox)."""

import numpy as np
from skimage import measure


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
