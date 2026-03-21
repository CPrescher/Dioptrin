"""Multi-detector integration example.

Demonstrates using MultiIntegrator to merge integration results from
multiple detectors into a single diffraction pattern. Uses synthetic
data with two simulated detectors at different distances.
"""

import dioptrin
import numpy as np


def main():
    print("Dioptrin MultiIntegrator Example")
    print("=" * 50)

    # --- Simulate two detectors with different geometries ---
    # Both detectors observe the same sample but are at different distances.
    # We use PONI dicts so no .poni files are needed.

    poni_det1 = {
        "pixel1": 0.000172,    # 172 um pixels (Pilatus-like)
        "pixel2": 0.000172,
        "distance": 0.2,       # 200 mm sample-detector distance
        "poni1": 0.08,         # beam center
        "poni2": 0.08,
        "wavelength": 1e-10,   # 1 Angstrom
    }

    poni_det2 = {
        "pixel1": 0.000172,
        "pixel2": 0.000172,
        "distance": 0.4,       # 400 mm - further away, covers different q range
        "poni1": 0.08,
        "poni2": 0.08,
        "wavelength": 1e-10,
    }

    # Determine a good shared radial range by checking each detector's range
    print("\n1. Determining radial range from individual detectors...")
    det1_temp = dioptrin.Integrator.from_poni_dict(poni_det1, unit="q_A^-1")
    det2_temp = dioptrin.Integrator.from_poni_dict(poni_det2, unit="q_A^-1")

    img1 = np.random.poisson(100, (195, 487)).astype(np.float64)
    img2 = np.random.poisson(100, (195, 487)).astype(np.float64)

    r1 = det1_temp.integrate1d(img1, n_points=500)
    r2 = det2_temp.integrate1d(img2, n_points=500)

    q_min = min(r1.radial[0], r2.radial[0])
    q_max = max(r1.radial[-1], r2.radial[-1])
    print(f"   Detector 1 q range: [{r1.radial[0]:.2f}, {r1.radial[-1]:.2f}] A^-1")
    print(f"   Detector 2 q range: [{r2.radial[0]:.2f}, {r2.radial[-1]:.2f}] A^-1")
    print(f"   Combined q range:   [{q_min:.2f}, {q_max:.2f}] A^-1")

    # --- Create MultiIntegrator with shared radial range ---
    print("\n2. Creating MultiIntegrator...")
    int1 = dioptrin.Integrator.from_poni_dict(
        poni_det1,
        method="pixel_split",
        unit="q_A^-1",
        radial_range=(q_min, q_max),
    )
    int2 = dioptrin.Integrator.from_poni_dict(
        poni_det2,
        method="pixel_split",
        unit="q_A^-1",
        radial_range=(q_min, q_max),
    )

    multi = dioptrin.MultiIntegrator(integrators=[int1, int2])
    print("   MultiIntegrator created with 2 detectors")

    # --- Single frame integration ---
    print("\n3. Integrating a single frame set...")
    n_points = 1000
    result = multi.integrate1d(images=[img1, img2], n_points=n_points)
    print(f"   Output bins: {len(result.radial)}")
    print(f"   q range: [{result.radial[0]:.4f}, {result.radial[-1]:.4f}] A^-1")

    # Find bins with actual signal
    nonzero = [i for i, v in enumerate(result.intensity) if v > 0]
    if nonzero:
        print(f"   Non-zero bins: {len(nonzero)} / {n_points}")
        print(f"   Mean intensity (non-zero bins): {np.mean([result.intensity[i] for i in nonzero]):.2f}")

    # --- With error estimates ---
    print("\n4. Integrating with error estimates...")
    result_err = multi.integrate1d(images=[img1, img2], n_points=n_points, errors=True)
    if result_err.errors:
        nonzero_errs = [result_err.errors[i] for i in nonzero] if nonzero else []
        if nonzero_errs:
            print(f"   Mean sigma (non-zero bins): {np.mean(nonzero_errs):.4f}")
            print(f"   Mean SNR (non-zero bins): {np.mean([result_err.intensity[i] / result_err.errors[i] for i in nonzero if result_err.errors[i] > 0]):.2f}")

    # --- Verify: single detector via MultiIntegrator matches plain Integrator ---
    print("\n5. Verifying single-detector equivalence...")
    int_single = dioptrin.Integrator.from_poni_dict(
        poni_det1,
        method="pixel_split",
        unit="q_A^-1",
        radial_range=(q_min, q_max),
    )
    result_plain = int_single.integrate1d(img1, n_points=n_points)

    multi_single = dioptrin.MultiIntegrator(integrators=[int_single])
    result_multi = multi_single.integrate1d(images=[img1], n_points=n_points)

    max_diff = max(
        abs(a - b)
        for a, b in zip(result_plain.intensity, result_multi.intensity)
    )
    print(f"   Max intensity difference (plain vs multi): {max_diff:.2e}")
    assert max_diff < 1e-10, "Single-detector equivalence check failed!"
    print("   PASSED")

    # --- Verify: two identical detectors produce same intensity as one ---
    print("\n6. Verifying two-identical-detectors invariance...")
    int_a = dioptrin.Integrator.from_poni_dict(
        poni_det1, method="pixel_split", unit="q_A^-1", radial_range=(q_min, q_max),
    )
    int_b = dioptrin.Integrator.from_poni_dict(
        poni_det1, method="pixel_split", unit="q_A^-1", radial_range=(q_min, q_max),
    )
    multi_double = dioptrin.MultiIntegrator(integrators=[int_a, int_b])
    result_double = multi_double.integrate1d(images=[img1, img1], n_points=n_points)

    max_diff = max(
        abs(a - b)
        for a, b in zip(result_plain.intensity, result_double.intensity)
    )
    print(f"   Max intensity difference (single vs double): {max_diff:.2e}")
    assert max_diff < 1e-10, "Two-identical-detectors invariance check failed!"
    print("   PASSED (signal doubles, norm doubles, ratio unchanged)")

    # --- Batch processing ---
    print("\n7. Batch processing multiple frame sets...")
    n_frames = 5
    image_sets = []
    for i in range(n_frames):
        frame1 = np.random.poisson(100 + i * 10, (195, 487)).astype(np.float64)
        frame2 = np.random.poisson(100 + i * 10, (195, 487)).astype(np.float64)
        image_sets.append([frame1, frame2])

    # Need to recreate integrators since they were consumed
    int_batch1 = dioptrin.Integrator.from_poni_dict(
        poni_det1, method="pixel_split", unit="q_A^-1", radial_range=(q_min, q_max),
    )
    int_batch2 = dioptrin.Integrator.from_poni_dict(
        poni_det2, method="pixel_split", unit="q_A^-1", radial_range=(q_min, q_max),
    )
    multi_batch = dioptrin.MultiIntegrator(integrators=[int_batch1, int_batch2])

    results = multi_batch.batch1d(images=image_sets, n_points=n_points)
    print(f"   Processed {len(results)} frame sets")
    for i, res in enumerate(results):
        nonzero_count = sum(1 for v in res.intensity if v > 0)
        print(f"   Frame {i}: {nonzero_count} non-zero bins")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
