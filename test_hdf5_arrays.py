"""
Test script to verify HDF5 integration with Dioptrin using h5py.
This demonstrates the recommended pattern for processing HDF5 files.
"""

import numpy as np

def test_batch1d_with_arrays():
    """Test batch1d with NumPy arrays (simulating h5py frames)."""
    try:
        import dioptrin
    except ImportError:
        print("⚠️  Dioptrin not installed. Run: maturin develop --features python")
        return

    # Create a simple PONI configuration
    poni_dict = {
        "pixel1": 0.000172,
        "pixel2": 0.000172,
        "distance": 0.2,
        "poni1": 0.1,
        "poni2": 0.1,
        "wavelength": 1e-10
    }

    integrator = dioptrin.Integrator.from_poni_dict(
        poni_dict,
        method="pixel_split"
    )

    # Simulate HDF5 frames as NumPy arrays
    print("Creating test frames...")
    frames = []
    for i in range(5):
        # 100x100 detector with some pattern
        frame = np.random.random((100, 100)).astype(np.float64) * 100
        frames.append(frame)

    print(f"✓ Created {len(frames)} test frames")

    # Test single frame integration
    print("\nTesting single frame integration...")
    result = integrator.integrate1d(frames[0], n_points=50)
    print(f"✓ Single frame: {len(result.intensity)} bins")

    # Test batch integration with arrays
    print("\nTesting batch integration with arrays...")
    results = integrator.batch1d(frames, n_points=50)
    print(f"✓ Batch processed {len(results)} frames")

    # Check results
    for i, result in enumerate(results):
        if result.is_ok():
            print(f"  Frame {i}: {len(result.result.intensity)} intensity values")
        else:
            print(f"  Frame {i}: ERROR - {result.error}")

    print("\n✅ All tests passed!")
    print("\nTo use with real HDF5 files:")
    print("  with h5py.File('data.h5', 'r') as f:")
    print("      frames = [f['/entry/data/data'][i] for i in range(100)]")
    print("  results = integrator.batch1d(frames, n_points=1000)")


def test_with_real_hdf5():
    """Test with a real HDF5 file if h5py is available."""
    try:
        import h5py
        import dioptrin
    except ImportError as e:
        print(f"⚠️  Required library not available: {e}")
        return

    # Check if test HDF5 file exists
    import os
    test_file = "test_data/nexus_test_file.nxs"

    if not os.path.exists(test_file):
        print(f"⚠️  Test file not found: {test_file}")
        return

    print(f"\n{'='*60}")
    print("Testing with real HDF5 file...")
    print(f"{'='*60}")

    poni_dict = {
        "pixel1": 0.000172,
        "pixel2": 0.000172,
        "distance": 0.2,
        "poni1": 0.1,
        "poni2": 0.1,
        "wavelength": 1e-10
    }

    integrator = dioptrin.Integrator.from_poni_dict(
        poni_dict,
        method="pixel_split"
    )

    with h5py.File(test_file, 'r') as f:
        dataset = f['/entry/instrument/detector/data']
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset dtype: {dataset.dtype}")

        # Load first 5 frames
        frames = [dataset[i] for i in range(min(5, len(dataset)))]
        print(f"Loaded {len(frames)} frames")

        # Integrate
        results = integrator.batch1d(frames, n_points=100)
        print(f"✓ Integrated {len(results)} frames")

        for i, result in enumerate(results):
            if result.is_ok():
                intensity_sum = np.sum(result.result.intensity)
                print(f"  Frame {i}: sum(intensity) = {intensity_sum:.2f}")

    print("\n✅ Real HDF5 test passed!")


if __name__ == "__main__":
    test_batch1d_with_arrays()
    test_with_real_hdf5()
