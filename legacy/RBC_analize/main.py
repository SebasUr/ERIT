import functions_hematological as fha
import data_management as dm
import json


result = fha.read_complex_field(
    file_path="complex_field_000001_Hologram_video01_frame_000001.mat",
    wavelength=0.632,  # um
    pixel_size=3.75,  # um
    magnification=40,
    # field_key="output"  # uncomment if auto-detection fails
)
print("Field shape:", result["shape"])
print("Parameters [wavelength, pixel_size, magnification, dx]:")
print(result["parameters"])


phase_result = fha.extract_phase(result["complex_field"], unwrap_method="skimage", show=True)
segmentation_result = fha.segment_rbc(phase_result["phase"], dxy=result["parameters"]["dxy"],
                                      method="otsu", min_area_um2=10, max_area_um2=None, separate_touching=True,
                                        polarity="bright")

fha.show_segmentation_results(phase_result["phase"], segmentation_result)

geometry_result = fha.extract_rbc_geometry(segmentation_result["labeled_mask"],dxy=result["parameters"]["dxy"])

profile_result = fha.phase_profiles(phase_result["phase"], segmentation_result["labeled_mask"], dxy=result["parameters"]["dxy"])
fha.show_phase_profiles_overview(phase_result["phase"], segmentation_result["labeled_mask"], profile_result)
fha.plot_phase_profiles(profile_result, profiles_per_figure=16)

height_volume_result = fha.calculate_rbc_height_volume(
                        phase_image=phase_result["phase"],
                        labeled_mask=segmentation_result["labeled_mask"],
                        profile_result=profile_result,
                        wavelength=result["parameters"]["wavelength"],
                        delta_n=0.083,
                        dxy=result["parameters"]["dxy"]
                    )

# Organize all results
hologram_result = dm.build_hologram_result(
    source_file=result["source_file"],
    parameters=result["parameters"],
    segmentation_result=segmentation_result,
    geometry_result=geometry_result,
    profile_result=profile_result,
    height_volume_result=height_volume_result,

    processing_parameters={
        "unwrap_method": "skimage",
        "segmentation_method": "otsu",
        "min_area_um2": 10,
        "max_area_um2": None,
        "separate_touching": True,
        "polarity": "bright",
        "delta_n": 0.083,
    }
)

# Save master JSON
json_path = dm.save_hologram_json(hologram_result)

# Convert directly to table
df = dm.hologram_json_to_dataframe(
    json_path
)

print(df)

# Export DataFrame to CSV
csv_path = dm.export_dataframe_to_csv(df,json_path)

with open(json_path, "r", encoding="utf-8") as file:
    saved_data = json.load(file)

print(saved_data["statistics"])