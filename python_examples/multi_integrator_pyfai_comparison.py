"""Compare Dioptrin MultiIntegrator with pyFAI MultiGeometry.

Uses two synthetic detector geometries with the same test image to
verify that Dioptrin's multi-detector merging produces results
consistent with pyFAI's MultiGeometry.

Requirements: pyFAI, fabio
"""

import numpy as np
import os
import sys


def main():
    try:
        import pyFAI
        from pyFAI.multi_geometry import MultiGeometry
        import fabio
    except ImportError:
        print("This example requires pyFAI and fabio:")
        print("  pip install pyFAI fabio")
        sys.exit(1)

    import dioptrin

    print("Dioptrin MultiIntegrator vs pyFAI MultiGeometry Comparison")
    print("=" * 60)
    print(f"pyFAI version:   {pyFAI.version}")
    print(f"Dioptrin threads: {dioptrin.rayon_num_threads()}")

    # --- Load test image ---
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    image_path = os.path.join(project_root, "test_data", "CeO2_Pilatus1M.tif")
    poni_path = os.path.join(project_root, "test_data", "v1.poni")

    if not os.path.exists(image_path):
        print(f"\nError: Test image not found at {image_path}")
        print("Run `cargo test` first to download the test data.")
        sys.exit(1)

    image_fabio = fabio.open(image_path).data
    # When passing NumPy arrays, no flip needed — Dioptrin only flips
    # when loading from TIFF file internally. With arrays, the user
    # controls orientation, so we pass the same data as pyFAI.
    image_dioptrin = np.ascontiguousarray(image_fabio, dtype=np.float64)

    print(f"\nImage: {image_path}")
    print(f"Image shape: {image_fabio.shape}")

    # --- Create two detector geometries ---
    # We use the same PONI file but modify poni2 to simulate
    # two detectors side by side (shifted horizontally).
    ai1 = pyFAI.load(poni_path)

    # Second detector: shift beam center horizontally
    ai2 = pyFAI.load(poni_path)
    ai2.poni2 = ai1.poni2 + 0.04  # shift 40mm in X

    print(f"\nDetector 1: poni2 = {ai1.poni2:.6f} m")
    print(f"Detector 2: poni2 = {ai2.poni2:.6f} m (shifted +40mm)")
    print(f"Wavelength: {ai1.wavelength:.4e} m")

    # --- Determine radial range ---
    n_points = 1000
    unit = "2th_deg"

    # Quick integration to find the range each detector covers
    r1 = ai1.integrate1d(image_fabio, n_points, unit=unit)
    r2 = ai2.integrate1d(image_fabio, n_points, unit=unit)

    radial_min = min(r1.radial[0], r2.radial[0])
    radial_max = max(r1.radial[-1], r2.radial[-1])
    print(f"\nDetector 1 range: [{r1.radial[0]:.4f}, {r1.radial[-1]:.4f}] deg")
    print(f"Detector 2 range: [{r2.radial[0]:.4f}, {r2.radial[-1]:.4f}] deg")
    print(f"Combined range:   [{radial_min:.4f}, {radial_max:.4f}] deg")

    # --- pyFAI MultiGeometry ---
    print(f"\n--- pyFAI MultiGeometry (n_points={n_points}) ---")
    multi_pyfai = MultiGeometry(
        [ai1, ai2],
        unit=unit,
        radial_range=(radial_min, radial_max),
    )
    # Disable pyFAI's extra solid angle scaling (pixel1*pixel2/dist²) so
    # both libraries report in the same units. Dioptrin's per-pixel
    # normalization already handles solid angle correction internally.
    result_pyfai = multi_pyfai.integrate1d(
        [image_fabio, image_fabio],
        npt=n_points,
        correctSolidAngle=False,
        error_model="poisson",
    )
    bins_pyfai = result_pyfai.radial
    int_pyfai = result_pyfai.intensity
    sig_pyfai = result_pyfai.sigma
    print(f"Output bins: {len(bins_pyfai)}")

    # --- Dioptrin MultiIntegrator ---
    print(f"\n--- Dioptrin MultiIntegrator (n_points={n_points}) ---")

    # Build PONI dicts matching pyFAI's geometry
    poni_dict1 = {
        "pixel1": ai1.pixel1,
        "pixel2": ai1.pixel2,
        "distance": ai1.dist,
        "poni1": ai1.poni1,
        "poni2": ai1.poni2,
        "rot1": ai1.rot1,
        "rot2": ai1.rot2,
        "rot3": ai1.rot3,
        "wavelength": ai1.wavelength,
    }
    poni_dict2 = {
        "pixel1": ai2.pixel1,
        "pixel2": ai2.pixel2,
        "distance": ai2.dist,
        "poni1": ai2.poni1,
        "poni2": ai2.poni2,
        "rot1": ai2.rot1,
        "rot2": ai2.rot2,
        "rot3": ai2.rot3,
        "wavelength": ai2.wavelength,
    }

    d_int1 = dioptrin.Integrator.from_poni_dict(
        poni_dict1,
        method="pixel_split",
        unit="2th_deg",
        radial_range=(radial_min, radial_max),
    )
    d_int2 = dioptrin.Integrator.from_poni_dict(
        poni_dict2,
        method="pixel_split",
        unit="2th_deg",
        radial_range=(radial_min, radial_max),
    )

    multi_dioptrin = dioptrin.MultiIntegrator(integrators=[d_int1, d_int2])
    result_dioptrin = multi_dioptrin.integrate1d(
        images=[image_dioptrin, image_dioptrin],
        n_points=n_points,
        errors=True,
    )
    bins_dioptrin = np.array(result_dioptrin.radial)
    int_dioptrin = np.array(result_dioptrin.intensity)
    sig_dioptrin = np.array(result_dioptrin.errors)

    print(f"Output bins: {len(bins_dioptrin)}")

    # --- Compare results ---
    print(f"\n--- Comparison ---")

    # Bin positions
    bin_diff = np.abs(bins_pyfai - bins_dioptrin)
    print(f"Bin positions max diff:  {bin_diff.max():.6e} deg")
    print(f"Bin positions mean diff: {bin_diff.mean():.6e} deg")

    # Intensity comparison (only where both have signal)
    both_nonzero = (int_pyfai != 0) & (int_dioptrin != 0)
    n_both = np.sum(both_nonzero)
    print(f"\nBins with signal in both: {n_both} / {n_points}")

    if n_both > 0:
        # Correlation (scale-independent)
        corr = np.corrcoef(int_pyfai[both_nonzero], int_dioptrin[both_nonzero])[0, 1]
        print(f"Intensity correlation: {corr:.10f}")

        # Note: pyFAI and Dioptrin may differ in absolute scale due to
        # solid angle normalization conventions. Correlation and peak
        # positions are the meaningful comparison metrics.

        # Compare after normalizing both to unit max
        p_norm = int_pyfai / int_pyfai.max()
        d_norm = int_dioptrin / int_dioptrin.max()
        norm_diff = np.abs(p_norm[both_nonzero] - d_norm[both_nonzero])
        print(f"Normalized diff (0-1 scale): max={norm_diff.max():.4e}, mean={norm_diff.mean():.4e}, median={np.median(norm_diff):.4e}")

        # Error shape comparison (normalize sigma by respective intensity)
        both_sigma = both_nonzero & (sig_pyfai != 0) & (sig_dioptrin != 0) & (int_pyfai != 0) & (int_dioptrin != 0)
        n_sigma = np.sum(both_sigma)
        if n_sigma > 0:
            # Compare relative error (sigma/intensity) between the two
            rel_err_pyfai = sig_pyfai[both_sigma] / int_pyfai[both_sigma]
            rel_err_dioptrin = sig_dioptrin[both_sigma] / int_dioptrin[both_sigma]
            rel_err_diff = np.abs(rel_err_pyfai - rel_err_dioptrin)
            print(f"\nRelative error (sigma/I) comparison ({n_sigma} bins):")
            print(f"  pyFAI mean:    {rel_err_pyfai.mean():.6f}")
            print(f"  Dioptrin mean: {rel_err_dioptrin.mean():.6f}")
            print(f"  Diff:  max={rel_err_diff.max():.4e}, mean={rel_err_diff.mean():.4e}")

    # --- Peak position comparison ---
    print(f"\n--- Peak positions ---")
    # Find peaks in both (CeO2 has strong diffraction peaks)
    pyfai_peak_idx = np.argmax(int_pyfai[both_nonzero])
    dioptrin_peak_idx = np.argmax(int_dioptrin[both_nonzero])
    pyfai_peak_pos = bins_pyfai[both_nonzero][pyfai_peak_idx]
    dioptrin_peak_pos = bins_dioptrin[both_nonzero][dioptrin_peak_idx]
    print(f"Strongest peak (pyFAI):    {pyfai_peak_pos:.4f} deg")
    print(f"Strongest peak (Dioptrin): {dioptrin_peak_pos:.4f} deg")
    print(f"Peak position difference:  {abs(pyfai_peak_pos - dioptrin_peak_pos):.4f} deg")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if n_both > 0 and corr > 0.95:
        print(f"PASSED: Correlation {corr:.6f} > 0.95")
    elif n_both > 0:
        print(f"WARNING: Correlation {corr:.6f} is below 0.95")
    else:
        print("WARNING: No overlapping non-zero bins found")


if __name__ == "__main__":
    main()
