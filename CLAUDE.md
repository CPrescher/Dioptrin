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
maturin develop                    # Build Python extension
```

## Architecture

**Core modules in `src/`:**
- `frame.rs` - TIFF image loading with automatic format conversion and coordinate flip
- `geometry.rs` - Detector geometry calculations (two-theta, azimuth, solid angle) using Rayon parallelization
- `poni.rs` - PONI calibration file parsing (v1 and v2 formats)
- `mask.rs` - Binary pixel masking
- `integration/` - Main `Integrator` API with builder pattern, LUT caching via RefCell, corrections
- `lut/` - CSR lookup table builders (counting, pixel-split, supersampled) and integration functions
- `math/` - Sparse matrix implementations (CSR, COO), histogram utilities
- `python.rs` - PyO3 bindings (enabled with `--features python`)

**Key patterns:**
- Builder pattern for Integrator configuration
- Interior mutability (RefCell) for LUT caching in `&self` methods
- Lazy geometry array computation, cached and updated on dimension change

## Test Data

Tests use fixtures in `test_data/`:
- `v1.poni`, `v2.poni` - calibration files
- `CeO2_Pilatus1M.tif` - auto-downloaded from GitHub on first test run
- `test_reference/` - expected outputs for validation
- `test_output/` - generated outputs (do not commit)

## Code Style

- Rust 2021 edition, standard `cargo fmt`
- Commit prefixes: `feat:`, `fix:`, `refactor:`, `doc:`, `bench:` (lowercase)
- Tests alongside source (e.g., `src/integration/tests.rs`)
