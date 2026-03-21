# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dioptrin is a high-performance Rust library for fast 1D/2D diffraction integration in X-ray crystallography. It uses Compressed Sparse Row (CSR) lookup tables for efficient integration, supporting counting, pixel-splitting, and supersampled methods. Optional Python bindings via PyO3.

## Build and Test Commands

```bash
cargo build                        # Debug build
cargo build --release              # Optimized release build
cargo test                         # Run all tests
cargo test -- --nocapture          # Tests with output
cargo bench --bench integrate      # Integration benchmarks
cargo bench --bench build          # LUT building benchmarks
cargo run --example integrate1d    # 1D integration example
cargo run --release --example timing1d  # Performance timing

# Python bindings
maturin develop --features python  # Build Python extension
python test_examples.py            # Test all Python examples
```

## Architecture

**Core modules in `src/`:**
- `frame.rs` - TIFF image loading with automatic format conversion and coordinate flip
- `geometry.rs` - Detector geometry calculations (two-theta, azimuth, solid angle) using Rayon parallelization
- `poni.rs` - PONI calibration file parsing (v1 and v2 formats)
- `mask.rs` - Binary pixel masking
- `integration/` - Main `Integrator` API with builder pattern, LUT caching via RefCell, corrections
  - `Integrator` supports optional `radial_range` for explicit bin grids (required for MultiIntegrator)
  - `integrate1d_raw` methods return intermediate sums (sum_signal, sum_norm, sum_variance) before division
- `lut/` - CSR lookup table builders (counting, pixel-split, supersampled) and integration functions
  - `_with_range` variants of 1D builders accept explicit radial min/max
  - `IntermediateResult1d`/`IntermediateResult2d` structs for raw integration output
- `multi.rs` - `MultiIntegrator` for merging results from multiple detectors (similar to pyFAI's MultiGeometry)
- `math/` - Sparse matrix implementations (CSR, COO), histogram utilities
- `python/` - PyO3 bindings (enabled with `--features python`)
  - Accepts both file paths and NumPy arrays for integration
  - `batch1d`/`batch2d` methods work with lists of TIFF paths or NumPy arrays
  - `MultiIntegrator` class for multi-detector merging from Python

**Key patterns:**
- Builder pattern for Integrator configuration
- Interior mutability (RefCell) for LUT caching in `&self` methods
- Lazy geometry array computation, cached and updated on dimension change
- Multi-detector merging: sum raw intermediates across detectors, divide once at the end

## Test Data

Tests use fixtures in `test_data/`:
- `v1.poni`, `v2.poni` - calibration files
- `CeO2_Pilatus1M.tif` - auto-downloaded from GitHub on first test run
- `test_reference/` - expected outputs for validation
- `test_output/` - generated outputs (do not commit)

## HDF5 Support

Dioptrin does NOT include HDF5 reading in Rust. Instead, users read HDF5 files with Python's `h5py` and pass NumPy arrays:

```python
import h5py
import numpy as np
with h5py.File('data.h5', 'r') as f:
    dataset = f['/entry/data/data']
    frames = [dataset[i].astype(np.float64) for i in range(100)]
results = integrator.batch1d(frames, n_points=1000)  # Parallel integration
```

**Rationale:** h5py handles all compression formats (bitshuffle, blosc, etc.) and is already installed in scientific Python environments. This keeps the Rust library lightweight and avoids dependency/plugin issues. See `docs/HDF5_USAGE.md` for details.

## Code Style

- Rust 2021 edition, standard `cargo fmt`
- Commit prefixes: `feat:`, `fix:`, `refactor:`, `doc:`, `bench:` (lowercase)
- Tests alongside source (e.g., `src/integration/tests.rs`)
