# Examples

This directory contains example scripts demonstrating Dioptrin usage.

## Running Examples

All examples require the Python bindings to be built first:

```bash
# Build Python package
maturin develop --features python

# Run an example
python examples/hdf5_chunked_batch.py
```

## Available Examples

### HDF5 Processing

**`hdf5_chunked_batch.py`** - Memory-efficient batch processing of large HDF5 files

Demonstrates:
- Processing 1000s of HDF5 frames without loading all into memory
- Chunking strategy for bounded memory usage
- Both `batch1d()` and `batch1d_iter()` APIs
- Real-time progress updates with streaming API

**Requirements:** `h5py` (install with `pip install h5py`)

**Usage:**
```bash
python examples/hdf5_chunked_batch.py
```

Creates a demo HDF5 file if test data is not available.

---

**`test_hdf5_arrays.py`** - Simple test/demo for HDF5 array integration

Demonstrates:
- Basic integration with NumPy arrays
- Working with real HDF5 test files
- Single frame and batch processing

**Requirements:** `h5py` (install with `pip install h5py`)

**Usage:**
```bash
python examples/test_hdf5_arrays.py
```

## Testing Examples

To verify all examples work correctly:

```bash
# From repository root
python test_examples.py
```

This is also run in CI to catch breaking changes.

## Adding New Examples

When adding a new example:

1. Create the script in `examples/`
2. Add it to `test_examples.py` in the `examples` list
3. Add documentation to this README
4. Ensure it runs successfully in CI

Examples should:
- Be self-contained and runnable without user intervention
- Include docstrings explaining what they demonstrate
- Handle missing test data gracefully (create synthetic data if needed)
- Print clear output showing what's happening
