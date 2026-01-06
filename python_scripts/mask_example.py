import dioptrin
import numpy as np
import os

def main():
    print("Dioptrin Mask Example:")

    # Define paths to test data
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    poni_path = os.path.join(project_root, "test_data", "v1.poni")
    image_path = os.path.join(project_root, "test_data", "CeO2_Pilatus1M.tif")

    if not os.path.exists(poni_path):
        print(f"Error: PONI file not found at {poni_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please ensure 'test_data/CeO2_Pilatus1M.tif' exists.")
        return

    # 1. Create an Integrator instance
    integrator = dioptrin.Integrator(
        poni_path,
        method="pixel_split",
        unit="2th_deg"
    )
    print("\nIntegrator created.")

    # 2. Perform 1D integration without a mask
    print("\nPerforming 1D integration WITHOUT mask...")
    result_no_mask = integrator.integrate1d(image_path, n_points=1000)
    print(f"  Intensity without mask (first 5 points): {result_no_mask.intensity[:5]}")
    print(f"  Intensity without mask (last 5 points): {result_no_mask.intensity[-5:]}")

    # 3. Create a dummy mask (e.g., mask out a central rectangular region)
    # The image is 1043 rows x 981 columns
    image_rows = 1043
    image_cols = 981
    dummy_mask_array = np.zeros((image_rows, image_cols), dtype=np.uint8)

    # Mask a central region: set pixels to 1 in the mask means they are masked
    start_row, end_row = image_rows // 4, image_rows // 2
    start_col, end_col = image_cols // 4, image_cols // 2
    dummy_mask_array[start_row:end_row, start_col:end_col] = 1
    print(f"\nCreated a dummy mask array, masking region [{start_row}:{end_row}, {start_col}:{end_col}].")

    # 4. Set the mask using a NumPy array
    print("Setting mask from NumPy array...")
    integrator.set_mask(dummy_mask_array)
    print("  Mask set successfully from array.")

    # 5. Perform 1D integration with the mask
    print("\nPerforming 1D integration WITH mask (from array)...")
    result_with_array_mask = integrator.integrate1d(image_path, n_points=1000)
    print(f"  Intensity with array mask (first 5 points): {result_with_array_mask.intensity[:5]}")
    print(f"  Intensity with array mask (last 5 points): {result_with_array_mask.intensity[-5:]}")
    print("  Note: Intensity values should be different from no mask due to masked pixels.")

    # 6. Clear the mask
    print("\nClearing the mask...")
    integrator.clear_mask()
    print("  Mask cleared successfully.")

    # 7. Perform 1D integration after clearing the mask (should be same as no mask)
    print("\nPerforming 1D integration AFTER clearing mask (should be like no mask)...")
    result_after_clear = integrator.integrate1d(image_path, n_points=1000)
    print(f"  Intensity after clear (first 5 points): {result_after_clear.intensity[:5]}")
    print(f"  Intensity after clear (last 5 points): {result_after_clear.intensity[-5:]}")

    # Verify that clearing the mask worked
    assert np.allclose(result_no_mask.intensity, result_after_clear.intensity), \
        "Intensity after clearing mask does not match intensity without mask!"
    print("  Verification successful: Intensity after clearing mask matches no mask.")

    # (Optional) Demonstrate setting mask from a TIFF file if one exists or can be created easily.
    # For simplicity, we'll skip creating a dummy TIFF here, but the API supports it.
    print("\nMask example complete.")

if __name__ == "__main__":
    main()
