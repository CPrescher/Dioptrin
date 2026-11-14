# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dioptrin is a high-performance Rust library for fast 1D/2D diffraction integration in X-ray crystallography. It uses sparse lookup tables (built as CSR; fast 1D execution defaults to a CSC layout, selectable with `sparse_layout`) supporting counting, pixel-splitting, pyFAI-compatible full splitting (`pyfai_full`), supersampled and legacy 1D histogram methods. Also includes geometry calibration and Python bindings via PyO3.

## Build and Test Commands

```bash
cargo build                        # Debug build
cargo build --release              # Optimized release build
cargo test                         # Run all tests
cargo test -- --nocapture          # Tests with output
cargo bench --bench integrate      # Integration benchmarks
cargo bench --bench build          # LUT building benchmarks
cargo bench --bench integrator     # Integrator API benchmarks
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
- `poni.rs` - PONI calibration file parsing (v1 `PixelSize1/2`; v2/v2.1 `Detector_config`, of which only `pixel1`/`pixel2` are read)
- `fit2d.rs` - Fit2D <-> PONI geometry conversion
- `detector.rs`, `detector/` - Detector models (bundled catalog, binning, orientation, Fit2D spline distortion) used optionally by `Geometry`
- `mask.rs` - Binary pixel masking
- `integration.rs`, `integration/` - Main `Integrator` API with builder pattern, thread-safe LUT caching (`RwLock`, `cache.rs`), corrections, fixed/dynamic mask modes, shared 2D grids (`grid.rs`)
  - `Integrator` supports optional `radial_range` for explicit bin grids (required for MultiIntegrator)
  - `integrate1d_raw` methods return intermediate sums (sum_signal, sum_norm, sum_variance) before division
- `lut/` - CSR lookup table builders (counting, pixel-split, supersampled, `pyfai_1d`/`pyfai_split` for `pyfai_full`) and integration functions; `prepared.rs`/`csc.rs` hold the execution layouts, `shared_grid.rs` the fractional splitting on explicit shared 2D edges
  - `_with_range` variants of 1D builders accept explicit radial min/max
  - `IntermediateResult1d`/`IntermediateResult2d` structs for raw integration output
- `multi.rs` - `MultiIntegrator` for merging 1D patterns and shared-grid 2D cakes from multiple detectors (similar to pyFAI's MultiGeometry)
- `batch.rs`, `batch/` - Parallel batch processing of many frames with one shared `Integrator`
- `calibration/` - Geometry refinement and automatic calibration (calibrants, peak search, ellipse fits, ring identification)
- `math/` - Sparse matrix implementations (CSR, COO), histogram utilities
- `python/` - PyO3 bindings (enabled with `--features python`)
  - Accepts both file paths and NumPy arrays for integration
  - `batch1d`/`batch1d_iter` accept lists of TIFF paths or NumPy arrays; `batch2d` takes TIFF paths only, arrays go through `batch2d_iter`
  - `MultiIntegrator` class for multi-detector merging from Python

**Key patterns:**
- Builder pattern for Integrator configuration
- Interior mutability (`RwLock`) for LUT and geometry caching in `&self` methods; `Integrator` is `Send + Sync` and shared across batch workers
- Lazy geometry array computation, cached and updated on dimension change
- Multi-detector merging: sum raw intermediates across detectors, divide once at the end

## Test Data

Tests use fixtures in `test_data/`:
- `v1.poni`, `v2.poni` - calibration files
- `CeO2_Pilatus1M.tif` - committed; downloaded from GitHub if missing
- `LaB6_40keV_MarCCD.*`, `ceo2_side.*`, `calibrants/` - calibration fixtures
- `detector_reference.*` - detector-model reference (`tests/detector_reference.rs`)
- `nexus_test_file.nxs` - HDF5 input for the Python HDF5 examples
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

- Rust 2024 edition, standard `cargo fmt`
- Commit prefixes: `feat:`, `fix:`, `refactor:`, `doc:`, `bench:` (lowercase)
- Tests alongside source (e.g., `src/integration/tests.rs`); integration tests in `tests/`, Python tests in `python_examples/`
