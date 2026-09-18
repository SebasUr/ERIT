"""Collect per-RBC results of one hologram into JSON / DataFrame / CSV."""

import json
from pathlib import Path

import pandas as pd


def build_hologram_result(source_file, parameters, segmentation_result, geometry_result, profile_result,
    height_volume_result, processing_parameters=None):
    """
    Combine all processed RBC information for one hologram into a structured dictionary ready to be saved as JSON.
    """

    geometry_by_id = {int(rbc["rbc_id"]): rbc for rbc in geometry_result["rbcs"]}
    phase_by_id = {int(rbc["rbc_id"]): rbc for rbc in profile_result["rbcs"]}
    height_volume_by_id = {int(rbc["rbc_id"]): rbc for rbc in height_volume_result["rbcs"]}

    all_ids = sorted(set(geometry_by_id) | set(phase_by_id) | set(height_volume_by_id))
    rbcs = {}

    for rbc_id in all_ids:
        geometry = geometry_by_id.get(rbc_id)
        phase = phase_by_id.get(rbc_id)
        height_volume = height_volume_by_id.get(rbc_id)
        rbc_entry = {}

        # Geometry
        if geometry is not None:
            rbc_entry["geometry"] = {
                "center_x_px": geometry["center_x_px"],
                "center_y_px": geometry["center_y_px"],
                "center_x_um": geometry["center_x_um"],
                "center_y_um": geometry["center_y_um"],
                "area_px": geometry["area_px"],
                "area_um2": geometry["area_um2"],
                "equivalent_radius_um": geometry["equivalent_radius_um"],
                "equivalent_diameter_um": geometry["equivalent_diameter_um"],
                "major_axis_um": geometry["major_axis_um"],
                "minor_axis_um": geometry["minor_axis_um"],
                "orientation_deg": geometry["orientation_deg"],
                "bbox_px": geometry["bbox_px"],
                "bbox_um": geometry["bbox_um"],
            }

        # Phase
        if phase is not None:
            rbc_entry["phase"] = {
                "delta_phase_1_rad": phase["delta_phase_1"],
                "delta_phase_2_rad": phase["delta_phase_2"],
                "delta_phase_rad": phase["delta_phase"],
            }

        # Height and volume
        if height_volume is not None:
            rbc_entry["height_volume"] = {
                "height_profile_1_um": height_volume["height_profile_1_um"],
                "height_profile_2_um": height_volume["height_profile_2_um"],
                "representative_height_um": height_volume["representative_height_um"],
                "volume_um3": height_volume["volume_um3"],
            }

        rbcs[str(rbc_id)] = rbc_entry

    result = {
        "source_file": str(source_file),
        "parameters": parameters,
        "processing_parameters": processing_parameters or {},
        "summary": {
            "num_segmented_rbcs": segmentation_result["num_rbcs"],
            "num_profiled_rbcs": profile_result["num_rbcs"],
            "num_excluded_profiles": profile_result["num_excluded"],
        },

        "rbcs": rbcs,
        "statistics": {
            "area": geometry_result["area_statistics"],
            "phase": profile_result["statistics"],
            "height": height_volume_result["height_statistics"],
            "volume": height_volume_result["volume_statistics"],
        }
    }

    return result


def save_hologram_json(hologram_result, output_directory="results"):
    """
    Save one hologram result dictionary as a formatted JSON file.
    """
    # Convert output directory to Path
    output_directory = Path(output_directory)

    # Create output directory if it does not exist
    output_directory.mkdir(parents=True, exist_ok=True)

    # Get the original source filename
    source_file = Path(hologram_result["source_file"])

    # Generate JSON filename from the original source filename
    json_name = source_file.with_suffix(".json").name

    # Final output path
    output_path = output_directory / json_name

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(hologram_result, file, indent=4, ensure_ascii=False)

    return str(output_path.resolve())


def hologram_json_to_dataframe(json_path):
    """
    Read one hologram JSON file and convert the per-RBC results
    into a pandas DataFrame.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []

    for rbc_id, rbc in data["rbcs"].items():
        geometry = rbc.get("geometry", {})
        phase = rbc.get("phase", {})
        height_volume = rbc.get("height_volume", {})
        rows.append({
            "RBC_ID": int(rbc_id),
            "Center_X_um": geometry.get("center_x_um"), "Center_Y_um": geometry.get("center_y_um"),
            "Area_um2": geometry.get("area_um2"),
            "Equivalent_Radius_um": geometry.get("equivalent_radius_um"),
            "Equivalent_Diameter_um": geometry.get("equivalent_diameter_um"),
            "Major_Axis_um": geometry.get("major_axis_um"),
            "Minor_Axis_um": geometry.get("minor_axis_um"),
            "Orientation_deg": geometry.get("orientation_deg"),
            "Delta_Phase_1_rad": phase.get("delta_phase_1_rad"),
            "Delta_Phase_2_rad": phase.get("delta_phase_2_rad"),
            "Delta_Phase_rad": phase.get("delta_phase_rad"),
            "Height_Profile_1_um": height_volume.get("height_profile_1_um"),
            "Height_Profile_2_um": height_volume.get("height_profile_2_um"),
            "Representative_Height_um": height_volume.get("representative_height_um"),
            "Volume_um3": height_volume.get("volume_um3"),
        })

    return pd.DataFrame(rows)


def export_dataframe_to_csv(dataframe, json_path):
    """
    Export a pandas DataFrame containing RBC measurements to a CSV file.

    outputs:
           Absolute path of the saved CSV file.
    """
    # Convert JSON path to Path
    json_path = Path(json_path)

    # Generate CSV path using the same filename
    csv_path = json_path.with_suffix(".csv")

    # Export DataFrame
    dataframe.to_csv(csv_path, index=False)

    return str(csv_path.resolve())
