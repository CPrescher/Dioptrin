"""
Example: Memory-efficient batch processing of large HDF5 files using chunking.

This demonstrates how to process thousands of HDF5 frames without loading
everything into memory at once.
"""

import h5py
import numpy as np
from pathlib import Path


def batch_process_hdf5_chunked(
    hdf5_path: str,
    dataset_path: str,
    integrator,
    n_points: int = 1000,
    chunk_size: int = 100,
    errors: bool = False
):
    """
    Process a large HDF5 dataset in chunks using batch1d.

    Args:
        hdf5_path: Path to HDF5 file
        dataset_path: Path to dataset within HDF5 file (e.g., '/entry/data/data')
        integrator: Dioptrin Integrator instance
        n_points: Number of radial bins
        chunk_size: Number of frames to process per batch (adjust based on RAM)
        errors: Whether to compute error estimates

    Returns:
        List of all batch results
    """
    all_results = []

    with h5py.File(hdf5_path, 'r') as f:
        dataset = f[dataset_path]
        n_frames = len(dataset)

        print(f"Dataset: {dataset.shape}, dtype: {dataset.dtype}")
        print(f"Processing {n_frames} frames in chunks of {chunk_size}")
        print()

        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk_num = start // chunk_size + 1
            total_chunks = (n_frames + chunk_size - 1) // chunk_size

            print(f"Chunk {chunk_num}/{total_chunks}: Loading frames {start}-{end-1}...")

            # Load chunk into memory
            frames = [dataset[i] for i in range(start, end)]

            print(f"  Integrating {len(frames)} frames...")

            # Process chunk in parallel
            results = integrator.batch1d(frames, n_points=n_points, errors=errors)
            all_results.extend(results)

            # Calculate some statistics for this chunk
            successful = sum(1 for r in results if r.is_ok())
            failed = len(results) - successful

            print(f"  ✓ Chunk complete: {successful} succeeded, {failed} failed")
            print(f"  Overall progress: {end}/{n_frames} frames ({100*end/n_frames:.1f}%)")
            print()

    return all_results


def batch_process_hdf5_iter_chunked(
    hdf5_path: str,
    dataset_path: str,
    integrator,
    n_points: int = 1000,
    chunk_size: int = 100,
    errors: bool = False
):
    """
    Process a large HDF5 dataset in chunks using batch1d_iter for streaming results.

    This version uses the iterator API which yields results as they complete,
    useful for real-time processing or progress updates.

    Args:
        hdf5_path: Path to HDF5 file
        dataset_path: Path to dataset within HDF5 file
        integrator: Dioptrin Integrator instance
        n_points: Number of radial bins
        chunk_size: Number of frames to process per batch
        errors: Whether to compute error estimates

    Yields:
        BatchResult objects as they complete
    """
    with h5py.File(hdf5_path, 'r') as f:
        dataset = f[dataset_path]
        n_frames = len(dataset)
        total_chunks = (n_frames + chunk_size - 1) // chunk_size

        print(f"Dataset: {dataset.shape}, dtype: {dataset.dtype}")
        print(f"Processing {n_frames} frames in {total_chunks} chunks of {chunk_size}")
        print()

        global_frame_idx = 0

        for chunk_num, start in enumerate(range(0, n_frames, chunk_size), 1):
            end = min(start + chunk_size, n_frames)

            print(f"Chunk {chunk_num}/{total_chunks}: Loading frames {start}-{end-1}...")

            # Load chunk into memory
            frames = [dataset[i] for i in range(start, end)]

            print(f"  Integrating {len(frames)} frames with streaming API...")

            # Process chunk with iterator (results stream as they complete)
            batch_iter = integrator.batch1d_iter(
                frames,
                n_points=n_points,
                errors=errors
            )

            # Process results as they arrive
            for result in batch_iter:
                # You can process each result immediately without waiting for the whole chunk
                if result.is_ok():
                    # Example: save to disk, update database, etc.
                    intensity_sum = np.sum(result.result.intensity)
                    print(f"    Frame {global_frame_idx}: sum(I) = {intensity_sum:.2e}", end='\r')
                else:
                    print(f"    Frame {global_frame_idx}: ERROR - {result.error}")

                global_frame_idx += 1
                yield result

            print(f"\n  ✓ Chunk {chunk_num} complete")
            print(f"  Overall progress: {end}/{n_frames} frames ({100*end/n_frames:.1f}%)")
            print()


def example_usage():
    """Example demonstrating both chunking approaches."""
    try:
        import dioptrin
    except ImportError:
        print("❌ Dioptrin not installed. Run: maturin develop --features python")
        return

    # Setup integrator
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

    # Check for test file
    test_file = Path("test_data/nexus_test_file.nxs")

    if not test_file.exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("Creating a synthetic HDF5 file for demonstration...")

        # Create a synthetic HDF5 file
        demo_file = Path("demo_large.h5")
        with h5py.File(demo_file, 'w') as f:
            # Simulate 500 frames of 200x200 detector
            print("  Generating 500 synthetic frames...")
            data = np.random.random((500, 200, 200)).astype(np.float32) * 1000
            f.create_dataset('/entry/data/data', data=data)

        hdf5_path = str(demo_file)
        dataset_path = '/entry/data/data'
    else:
        hdf5_path = str(test_file)
        dataset_path = '/entry/instrument/detector/data'

    print("="*70)
    print("Example 1: batch1d with chunking (all results at end)")
    print("="*70)
    print()

    results = batch_process_hdf5_chunked(
        hdf5_path,
        dataset_path,
        integrator,
        n_points=100,
        chunk_size=50  # Small chunks for demo
    )

    print(f"✅ Processed {len(results)} frames total")
    successful = sum(1 for r in results if r.is_ok())
    print(f"   {successful} successful, {len(results) - successful} failed")
    print()

    print("="*70)
    print("Example 2: batch1d_iter with chunking (streaming results)")
    print("="*70)
    print()

    # Process with streaming API
    result_count = 0
    for result in batch_process_hdf5_iter_chunked(
        hdf5_path,
        dataset_path,
        integrator,
        n_points=100,
        chunk_size=50
    ):
        result_count += 1
        # You could save results to disk here, update a progress bar, etc.

    print(f"✅ Streamed {result_count} results")
    print()

    print("="*70)
    print("Memory Efficiency Tips:")
    print("="*70)
    print("- chunk_size=100:  ~100 frames × 3262×3108 × 4 bytes ≈ 4 GB RAM")
    print("- chunk_size=50:   ~50 frames × 3262×3108 × 4 bytes ≈ 2 GB RAM")
    print("- chunk_size=20:   ~20 frames × 3262×3108 × 4 bytes ≈ 800 MB RAM")
    print()
    print("Adjust chunk_size based on your available RAM and frame size.")
    print("Larger chunks = fewer Python/Rust transitions = slightly faster.")


if __name__ == "__main__":
    example_usage()
