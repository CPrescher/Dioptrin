# Python Examples

This directory contains Python example scripts demonstrating Dioptrin usage.

**Note:** Rust examples (`.rs` files) are in the `examples/` directory and are run with `cargo run --example <name>`.

## Running Python Examples

All examples require the Python bindings to be built first:

```bash
# Build Python package
maturin develop --features python

# Run an example
python python_examples/hdf5_chunked_batch.py
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

---

**`test_hdf5_arrays.py`** - Simple test/demo for HDF5 array integration

Demonstrates:
- Basic integration with NumPy arrays
- Working with real HDF5 test files
- Single frame and batch processing

**Requirements:** `h5py` (install with `pip install h5py`)

### Core Functionality Tests

**`geometry_test_check.py`** - Test detector geometry calculations

**`integration_test_check_1d.py`** - Test 1D integration methods

**`integration_test_check_2d.py`** - Test 2D integration methods

**`mask_example.py`** - Demonstration of pixel masking

### Batch Processing

**`batch_api_smoke_test.py`** - Quick smoke test for batch API

**`batch_api_speed_test.py`** - Performance benchmarks for batch processing

### Multi-Detector Integration

**`multi_integrator_example.py`** - Multi-detector merging with `MultiIntegrator`

Demonstrates:
- Determining a shared radial range across detectors
- Creating a `MultiIntegrator` from multiple `Integrator` instances
- Single and batch integration with merged output
- Error propagation across detectors
- Verification: single-detector equivalence, two-identical-detectors invariance

### Comparison with pyFAI

**`compare_to_pyfai.py`** - Compare Dioptrin results with pyFAI

**`batch_pyfai_speed_test.py`** - Speed comparison with pyFAI batch processing

**Requirements:** `pyFAI` (install with `pip install pyfai`)

## Testing Examples

To verify all examples work correctly:

```bash
# From repository root
python test_examples.py
```

This automatically discovers and runs all `.py` files in the `python_examples/` directory.
It's also run in CI to catch breaking changes.

## Adding New Examples

When adding a new example:

1. Create the script in `python_examples/` (it will be auto-discovered by `test_examples.py`)
2. Add documentation to this README
3. Ensure it runs successfully in CI

Examples should:
- Be self-contained and runnable without user intervention
- Include docstrings explaining what they demonstrate
- Handle missing test data gracefully (create synthetic data if needed)
- Print clear output showing what's happening
- Not start with underscore (those are skipped by the test runner)
