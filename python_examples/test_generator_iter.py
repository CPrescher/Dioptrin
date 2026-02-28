"""Test generator/iterator support for batch1d_iter and batch2d_iter.

Verifies that generators, iterators, and various input types work correctly
with the streaming batch APIs.
"""
import numpy as np
import dioptrin


PONI_PATH = "test_data/v1.poni"
TIFF_PATH = "test_data/CeO2_Pilatus1M.tif"


def file_path_generator(paths):
    """Generator that yields file paths one at a time."""
    for p in paths:
        yield p


def array_generator(arrays):
    """Generator that yields numpy arrays one at a time."""
    for a in arrays:
        yield a


def mixed_generator(paths, arrays):
    """Generator that yields a mix of paths and arrays."""
    for p in paths:
        yield p
    for a in arrays:
        yield a


def main() -> None:
    integrator = dioptrin.Integrator(PONI_PATH, method="pixel_split")

    # Get reference results from a list (existing API)
    files = [TIFF_PATH, TIFF_PATH]
    ref_results = []
    for r in integrator.batch1d_iter(files, n_points=500):
        assert r.is_ok(), f"Reference failed: {r.error}"
        ref_results.append(r)

    ref_intensity_sum = sum(sum(r.result.intensity) for r in ref_results)
    print(f"Reference (list): {len(ref_results)} results, sum(I)={ref_intensity_sum:.2f}")

    # --- Test 1: Generator of file paths (1D) ---
    print("\n--- Test 1: Generator of file paths (batch1d_iter) ---")
    gen_results = []
    it = integrator.batch1d_iter(file_path_generator(files), n_points=500)
    assert it.total == 0, f"Expected total=0 for generator, got {it.total}"
    assert it.progress is None, f"Expected progress=None for generator, got {it.progress}"
    for r in it:
        assert r.is_ok(), f"Generator path failed: {r.error}"
        gen_results.append(r)
    gen_intensity_sum = sum(sum(r.result.intensity) for r in gen_results)
    assert len(gen_results) == 2, f"Expected 2 results, got {len(gen_results)}"
    assert abs(gen_intensity_sum - ref_intensity_sum) < 1e-6, \
        f"Intensity mismatch: {gen_intensity_sum} vs {ref_intensity_sum}"
    print(f"  OK: {len(gen_results)} results, sum(I)={gen_intensity_sum:.2f}")
    print(f"  completed={it.completed}")

    # --- Test 2: Generator of numpy arrays (1D) ---
    print("\n--- Test 2: Generator of numpy arrays (batch1d_iter) ---")
    frame_array = np.random.rand(516, 1554).astype(np.float64) * 100
    frame_arrays = [frame_array, frame_array, frame_array]

    arr_results = []
    for r in integrator.batch1d_iter(array_generator(frame_arrays), n_points=500):
        assert r.is_ok(), f"Generator array failed: {r.error}"
        arr_results.append(r)

    assert len(arr_results) == 3, f"Expected 3 results, got {len(arr_results)}"
    # All results should be identical since same array
    for i in range(1, len(arr_results)):
        diff = max(abs(a - b) for a, b in
                   zip(arr_results[0].result.intensity, arr_results[i].result.intensity))
        assert diff < 1e-10, f"Array results differ at index {i}: max_diff={diff}"
    print(f"  OK: {len(arr_results)} results, all identical")

    # --- Test 3: Empty generator ---
    print("\n--- Test 3: Empty generator ---")
    empty_results = list(integrator.batch1d_iter(iter([]), n_points=500))
    assert len(empty_results) == 0, f"Expected 0 results, got {len(empty_results)}"
    print(f"  OK: 0 results from empty generator")

    # --- Test 4: Generator with errors=True (1D) ---
    print("\n--- Test 4: Generator with errors=True (batch1d_iter) ---")
    err_results = []
    for r in integrator.batch1d_iter(file_path_generator(files), n_points=500, errors=True):
        assert r.is_ok(), f"Generator errors failed: {r.error}"
        assert r.result.errors is not None, "Expected errors in result"
        err_results.append(r)
    assert len(err_results) == 2
    print(f"  OK: {len(err_results)} results with errors")

    # --- Test 5: Generator of file paths (2D) ---
    print("\n--- Test 5: Generator of file paths (batch2d_iter) ---")
    results_2d = []
    for r in integrator.batch2d_iter(file_path_generator(files), n_radial=50, n_azimuthal=36):
        assert r.is_ok(), f"2D generator failed: {r.error}"
        results_2d.append(r)
    assert len(results_2d) == 2
    assert results_2d[0].result.n_radial == 50
    assert results_2d[0].result.n_azimuthal == 36
    print(f"  OK: {len(results_2d)} 2D results ({results_2d[0].result.n_radial}x{results_2d[0].result.n_azimuthal})")

    # --- Test 6: Generator of numpy arrays (2D) ---
    print("\n--- Test 6: Generator of numpy arrays (batch2d_iter) ---")
    results_2d_arr = []
    for r in integrator.batch2d_iter(array_generator(frame_arrays[:2]), n_radial=50, n_azimuthal=36):
        assert r.is_ok(), f"2D array generator failed: {r.error}"
        results_2d_arr.append(r)
    assert len(results_2d_arr) == 2
    print(f"  OK: {len(results_2d_arr)} 2D results from array generator")

    # --- Test 7: Backward compat - list of paths still works ---
    print("\n--- Test 7: Backward compatibility ---")
    list_results = list(integrator.batch1d_iter(files, n_points=500))
    assert len(list_results) == 2
    list_sum = sum(sum(r.result.intensity) for r in list_results)
    assert abs(list_sum - ref_intensity_sum) < 1e-6
    print(f"  OK: list of paths still works, sum(I)={list_sum:.2f}")

    # --- Test 8: List of arrays still works ---
    print("\n--- Test 8: List of arrays backward compat ---")
    list_arr_results = list(integrator.batch1d_iter(frame_arrays[:2], n_points=500))
    assert len(list_arr_results) == 2
    print(f"  OK: list of arrays still works")

    print("\nAll generator/iterator tests passed!")


if __name__ == "__main__":
    main()
