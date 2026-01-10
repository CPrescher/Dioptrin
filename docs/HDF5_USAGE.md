# Using Dioptrin with HDF5 Files

Dioptrin doesn't include HDF5 reading in its Rust core, but you can easily integrate HDF5 files using Python's `h5py` library. This approach is cleaner, more flexible, and avoids dependency issues.

## Why No Built-in HDF5 Support?

- **h5py is ubiquitous**: Already installed in every scientific Python environment
- **Handles all compression**: bitshuffle, blosc, lz4, etc. - no plugin hassles
- **Zero-copy integration**: NumPy arrays pass efficiently to Rust
- **Simpler dependencies**: Keeps the Rust library lightweight

## Installation

```bash
pip install h5py dioptrin
```

## Usage Patterns

### Single Frame Integration

```python
import h5py
import dioptrin

# Load integrator
integrator = dioptrin.Integrator("calibration.poni", method="pixel_split")

# Read and integrate a single frame
with h5py.File('data.h5', 'r') as f:
    frame = f['/entry/data/data'][42]  # Frame 42
    result = integrator.integrate1d(frame, n_points=1000, errors=True)

print(result.radial, result.intensity, result.errors)
```

### Batch Processing (Recommended)

For multiple frames, load them into a list and use `batch1d` for parallel processing:

```python
import h5py
import dioptrin

integrator = dioptrin.Integrator("calibration.poni")

# Load all frames (h5py handles decompression)
with h5py.File('data.h5', 'r') as f:
    dataset = f['/entry/data/data']
    frames = [dataset[i] for i in range(len(dataset))]

# Parallel integration in Rust
results = integrator.batch1d(frames, n_points=1000)

for result in results:
    if result.is_ok():
        print(f"Frame {result.filename}: {len(result.result.intensity)} points")
```

### Processing Large Datasets in Chunks

For very large HDF5 files that don't fit in memory, process in chunks:

```python
import h5py
import dioptrin

integrator = dioptrin.Integrator("calibration.poni")

chunk_size = 100  # Adjust based on available RAM
all_results = []

with h5py.File('large_data.h5', 'r') as f:
    dataset = f['/entry/data/data']
    n_frames = len(dataset)

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)

        # Load chunk
        frames = [dataset[i] for i in range(start, end)]

        # Integrate in parallel
        results = integrator.batch1d(frames, n_points=1000)
        all_results.extend(results)

        print(f"Processed {end}/{n_frames} frames")
```

### Streaming Results with Iterator API

Use `batch1d_iter` to process results as they complete (useful for real-time processing):

```python
import h5py
import dioptrin

integrator = dioptrin.Integrator("calibration.poni")

chunk_size = 100

with h5py.File('large_data.h5', 'r') as f:
    dataset = f['/entry/data/data']
    n_frames = len(dataset)

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        frames = [dataset[i] for i in range(start, end)]

        # Stream results as they complete
        batch_iter = integrator.batch1d_iter(frames, n_points=1000)

        for result in batch_iter:
            if result.is_ok():
                # Process immediately: save to disk, update database, etc.
                save_result(result)
            print(f"Progress: {batch_iter.completed}/{batch_iter.total}")
```

**See `examples/hdf5_chunked_batch.py` for a complete working example.**

### Auto-Detecting NeXus Paths

```python
import h5py

def find_nexus_data(filepath):
    """Find the data path in a NeXus HDF5 file."""
    common_paths = [
        '/entry/data/data',
        '/entry/instrument/detector/data',
        '/entry/data/data_000001',
        '/entry1/data/data',
        '/entry_0000/instrument/detector/data',
    ]

    with h5py.File(filepath, 'r') as f:
        for path in common_paths:
            if path in f:
                return path

    raise ValueError(f"Could not find data in {filepath}")

# Usage
data_path = find_nexus_data('data.nxs')
with h5py.File('data.nxs', 'r') as f:
    frames = [f[data_path][i] for i in range(10)]

results = integrator.batch1d(frames, n_points=1000)
```

## Performance Characteristics

### What's Fast
- **Integration**: Fully parallelized in Rust across all CPU cores
- **Array passing**: Near-zero-copy from NumPy to Rust via PyO3

### What's Sequential
- **HDF5 reading**: h5py's file reading is sequential (HDF5 C library limitation)
- **Decompression**: Happens during h5py read

### Typical Timings (100 frames, 3262×3108 pixels)

| Operation | Time | Parallel |
|-----------|------|----------|
| HDF5 read (uncompressed) | ~0.5s | No |
| HDF5 read (bitshuffle) | ~2s | No |
| Integration (pixel_split) | ~30s | Yes |
| **Total** | ~32s | Mixed |

**Key insight**: Integration dominates runtime, and that's fully parallelized. The sequential HDF5 reading is only ~6% of total time.

## Array Requirements

- **Type**: `float64` (will auto-convert from int16, uint16, etc.)
- **Layout**: C-contiguous (row-major) - h5py arrays are already this
- **Shape**: 2D array `[height, width]`

## Comparison with pyFAI

If you're migrating from pyFAI, the workflow is similar:

**pyFAI:**
```python
import fabio
img = fabio.open('data.h5')
result = ai.integrate1d(img.data, 1000)
```

**Dioptrin:**
```python
import h5py
with h5py.File('data.h5', 'r') as f:
    frame = f['/entry/data/data'][0]
result = integrator.integrate1d(frame, 1000)
```

The main difference: you handle file I/O explicitly with h5py, which gives you more control.

## Mixing File Types

You can mix TIFF files and HDF5 arrays in the same workflow:

```python
# Batch process TIFF files
tiff_results = integrator.batch1d(tiff_paths, n_points=1000)

# Batch process HDF5 frames
with h5py.File('data.h5', 'r') as f:
    frames = [f['/entry/data/data'][i] for i in range(100)]
hdf5_results = integrator.batch1d(frames, n_points=1000)

# Combine results
all_results = tiff_results + hdf5_results
```

## Troubleshooting

### "Array must be C-contiguous"

h5py arrays are normally C-contiguous, but if you've sliced or transposed them:

```python
frame = dataset[42]  # Good
frame = np.ascontiguousarray(dataset[42])  # Force contiguous if needed
```

### Compression Not Supported

If h5py can't read your file:

```bash
# macOS
brew install hdf5-plugins

# Linux
sudo apt-get install libhdf5-plugins
```

Or export uncompressed:

```python
import h5py

with h5py.File('compressed.h5', 'r') as f_in:
    data = f_in['/entry/data/data'][:]

with h5py.File('uncompressed.h5', 'w') as f_out:
    f_out.create_dataset('/entry/data/data', data=data)
```

## Advanced: Custom Data Sources

The `batch1d` method accepts any list of 2D float64 NumPy arrays. This means you can integrate data from:

- HDF5 files (via h5py)
- NetCDF files (via netCDF4)
- Memory-mapped arrays
- Dask arrays (call `.compute()` first)
- Your custom file format

```python
import netCDF4

with netCDF4.Dataset('data.nc', 'r') as f:
    frames = [f['detector_data'][i] for i in range(100)]

results = integrator.batch1d(frames, n_points=1000)
```

## Summary

✅ **Do**: Use h5py to read HDF5 files, pass arrays to Dioptrin
✅ **Do**: Use `batch1d()` for parallel integration of multiple frames
✅ **Do**: Process large datasets in chunks to manage memory
❌ **Don't**: Expect built-in HDF5 reading in the Rust library
❌ **Don't**: Worry about compression - h5py handles it
