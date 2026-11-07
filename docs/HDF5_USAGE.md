# Using Dioptrin with HDF5 Files

[Manual index](index.md) · [Performance and CPU backends](PERFORMANCE.md)

This guide describes the current Python workflow: read HDF5 data with `h5py`,
then pass arrays to Dioptrin for integration. Dioptrin's Python integrator does
not open HDF5 paths directly. Native reading for Rust consumers is provided by
the separate [dioptrin-io library](https://github.com/CPrescher/dioptrin-io).

Reading, decompression and integration are separate stages. The reader and data
layout determine how decoding can run in parallel; the integration worker count
alone does not establish the throughput of the whole pipeline.

## Installation

```bash
pip install h5py dioptrin
```

For files using plugin-provided filters such as Bitshuffle/LZ4, also install
the matching filter implementation and register it before reading. For example:

```bash
pip install hdf5plugin
```

Then use `import hdf5plugin` before reading compressed datasets. See the
[filter documentation](https://hdf5plugin.readthedocs.io/en/stable/usage.html#read-compressed-datasets).

## Usage Patterns

### Single Frame Integration

```python
import h5py
import numpy as np
import dioptrin

# Load integrator
integrator = dioptrin.Integrator("calibration.poni", method="pixel_split")

# Read and integrate a single frame
with h5py.File('data.h5', 'r') as f:
    frame = np.ascontiguousarray(f['/entry/data/data'][42], dtype=np.float64)
    result = integrator.integrate1d(frame, n_points=1000, errors=True)

print(result.radial, result.intensity, result.errors)
```

### Batch Processing a Small Group

For a small group of frames, use `batch1d` for parallel integration. This example
processes the first eight frames. For a complete large file, use the bounded
chunked approach below.

```python
import h5py
import numpy as np
import dioptrin

integrator = dioptrin.Integrator("calibration.poni")

# h5py reads and decompresses each selected frame
with h5py.File('data.h5', 'r') as f:
    dataset = f['/entry/data/data']
    # Prepare the contiguous float64 arrays expected by the Python API
    frames = [np.ascontiguousarray(dataset[i], dtype=np.float64)
              for i in range(min(8, len(dataset)))]

# Parallel integration in Rust
results = integrator.batch1d(frames, n_points=1000)

for result in results:
    if result.is_ok():
        print(f"Frame {result.filename}: {len(result.result.intensity)} points")
```

### Processing Large Datasets in Chunks

For large HDF5 files, process bounded groups. One 3262 × 3108 float64 frame is
about 77.35 MiB; eight such arrays alone use about 619 MiB. Decoding, lookup tables,
worker queues and retained results add memory, so this is not an RSS limit.

```python
import h5py
import numpy as np
import dioptrin

integrator = dioptrin.Integrator("calibration.poni")

chunk_size = 8  # Adjust based on available RAM
all_results = []

with h5py.File('large_data.h5', 'r') as f:
    dataset = f['/entry/data/data']
    n_frames = len(dataset)

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)

        # Load chunk and convert to float64
        frames = [np.ascontiguousarray(dataset[i], dtype=np.float64) for i in range(start, end)]

        # Integrate in parallel
        results = integrator.batch1d(frames, n_points=1000)
        all_results.extend(results)

        print(f"Processed {end}/{n_frames} frames")
```

### Streaming Results with Iterator API

Use `batch1d_iter` to process results as they complete (useful for real-time processing):

```python
import h5py
import numpy as np
import dioptrin

integrator = dioptrin.Integrator("calibration.poni")

chunk_size = 8

with h5py.File('large_data.h5', 'r') as f:
    dataset = f['/entry/data/data']
    n_frames = len(dataset)

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        # Convert to float64
        frames = [np.ascontiguousarray(dataset[i], dtype=np.float64) for i in range(start, end)]

        # Stream results as they complete
        batch_iter = integrator.batch1d_iter(frames, n_points=1000)

        for result in batch_iter:
            if result.is_ok():
                # Process immediately: save to disk, update database, etc.
                save_result(result)
            print(f"Progress: {batch_iter.completed}/{batch_iter.total}")
```

**See `python_examples/hdf5_chunked_batch.py` for a complete working example.**

### Auto-Detecting NeXus Paths

```python
import h5py
import numpy as np
import dioptrin

def find_nexus_data(filepath):
    """Find the data path in a NeXus HDF5 file."""
    common_paths = [
        '/entry/data/data',
        '/entry/instrument/detector/data',
        '/entry/data/data_000001',
        '/entry1/data/data',
        '/entry_0000/instrument/detector/data',
        '/entry_0000/instrument/eiger/data',
    ]

    with h5py.File(filepath, 'r') as f:
        for path in common_paths:
            if path in f:
                return path

    raise ValueError(f"Could not find data in {filepath}")

# Usage
integrator = dioptrin.Integrator('calibration.poni')
data_path = find_nexus_data('data.nxs')
with h5py.File('data.nxs', 'r') as f:
    frames = [np.ascontiguousarray(f[data_path][i], dtype=np.float64)
              for i in range(min(8, len(f[data_path])))]

results = integrator.batch1d(frames, n_points=1000)
```

## Performance Characteristics

h5py serializes its HDF5 API calls through an interpreter-wide lock, including
calls for different files or datasets. A pipeline can overlap h5py reading with
non-h5py processing, but extra h5py threads alone do not make those API calls
parallel. Actual overlap with integration depends on the application's scheduling
and bindings. See [h5py's threading documentation](https://docs.h5py.org/en/stable/threads.html).

Dioptrin can integrate frames in parallel after data are available. The native
indexed reader in dioptrin-io follows a different path: it indexes supported
immutable chunks through HDF5 once, then uses parallel positional reads and
checked decoding outside that mutex. Applications must explicitly use that
reader; it is separate from the Python workflow shown here.

Measure decoding, conversion/copies, mask creation, integration and normalization
with the intended data and settings. Integration does not always dominate the
total. The [integration performance chapter](PERFORMANCE.md) covers masks,
float64 arithmetic and shared plans. The
[dioptrin-io performance guide](https://dioptrin-io.readthedocs.io/en/latest/PERFORMANCE.html)
covers reader backends, worker counts, sparse/dense images and Windows read/decode rates.

## Array Requirements

- **Type**: `float64` for the current Python array API
- **Layout**: C-contiguous (row-major)
- **Shape**: 2D array `[height, width]`

HDF5 count arrays often use integer types. Prepare float64 for the current
Python interface; this requirement does not apply to dioptrin-io's native pixel
buffers or Dioptrin's typed Rust integration paths.

```python
frame = np.ascontiguousarray(dataset[i], dtype=np.float64)
```

`np.ascontiguousarray` avoids an additional conversion copy when its input
already has the requested dtype and layout. Reading a dataset slice still
materializes an array. The single-frame Python integration API borrows a
contiguous float64 slice; application preparation and batch/stream queues can
have other copy and memory costs.

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
import numpy as np
with h5py.File('data.h5', 'r') as f:
    frame = np.ascontiguousarray(f['/entry/data/data'][0], dtype=np.float64)
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
    dataset = f['/entry/data/data']
    frames = [np.ascontiguousarray(dataset[i], dtype=np.float64)
              for i in range(min(8, len(dataset)))]
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

The compression filter must be available before the first data read. For
Bitshuffle/LZ4 and other filters supported by hdf5plugin:

```python
import h5py
import hdf5plugin  # registers its supported filters with h5py's HDF5 library
```

Other filters may require another matching plugin. A file cannot be read or
converted to an uncompressed copy until its original compression is supported.
See the [plugin usage guide](https://hdf5plugin.readthedocs.io/en/stable/usage.html).

## Advanced: Custom Data Sources

The `batch1d` method accepts any list of 2D float64 NumPy arrays. This means you can integrate data from:

- HDF5 files (via h5py)
- NetCDF files (via netCDF4)
- Memory-mapped arrays
- Dask arrays (call `.compute()` first)
- Your custom file format

```python
import netCDF4
import numpy as np

with netCDF4.Dataset('data.nc', 'r') as f:
    dataset = f['detector_data']
    frames = [np.ascontiguousarray(dataset[i], dtype=np.float64)
              for i in range(min(8, len(dataset)))]

results = integrator.batch1d(frames, n_points=1000)
```

## Choosing the workflow

Use h5py with the required filter plugins for the Python examples in this guide.
Pass contiguous float64 arrays to Dioptrin and choose chunk/queue sizes from
available memory. Use the separate dioptrin-io reader APIs when building a native
Rust input pipeline. Compare complete pipelines with matched data and integration
settings before assigning a speedup to the decoder or integrator.
