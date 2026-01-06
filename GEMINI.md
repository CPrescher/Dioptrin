# Gemini Code Understanding

## Project Overview

This project, **Dioptrin**, is a high-performance Rust library designed for the fast 1D and 2D azimuthal integration of 2D X-ray diffraction data. It is optimized for speed through techniques like compressed sparse row (CSR) lookup tables and parallel processing using the Rayon library.

The core functionality is written in Rust, but it provides comprehensive Python bindings via PyO3 and Maturin, making it easy to integrate into scientific Python workflows. Key features include:

-   Multiple integration methods: pixel-splitting, counting, histogramming, and supersampling.
-   Support for various radial units (`2th_rad`, `2th_deg`, `q_A^-1`, `q_nm^-1`).
-   Corrections for polarization and solid angle.
-   Batch processing capabilities to efficiently handle multiple files in parallel.
-   Calculation of error estimates.

## Building and Running

The project is a hybrid Rust and Python project.

### Rust

The core library and examples are managed with Cargo.

**Building the library:**

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release
```

**Running examples:**

The `examples/` directory contains several useful examples.

```bash
# Run a specific example (e.g., 1D integration)
cargo run --example integrate1d

# Run a benchmark example in release mode for accurate timing
cargo run --release --example benchmark_pipeline
```

**Running benchmarks:**

The project includes benchmarks using `criterion`.

```bash
# Run integration benchmarks
cargo bench --bench integrate

# Run LUT build benchmarks
cargo bench --bench build
```

### Python

The Python bindings are built using `maturin`. It is recommended to use a virtual environment. The project uses `uv` for managing python environments.

**Setting up the environment:**

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies from uv.lock
pip install -r requirements.txt
```

**Building and installing the Python package:**

```bash
# Build the Python bindings and install in the current environment
maturin develop

# For an optimized build
maturin develop --release
```

**Running Python scripts:**

After building, you can run Python scripts that import `dioptrin`.

```python
import dioptrin
import numpy as np

# Example from README.md
integrator = dioptrin.Integrator(
    "test_data/v1.poni",
    method="pixel_split",
    unit="2th_deg"
)

# 1D integration from a TIFF file
result = integrator.integrate1d("test_data/CeO2_Pilatus1M.tif", n_points=1000)
print("Radial bins:", result.radial)
print("Intensity:", result.intensity)
```

## Testing

### Rust

The Rust test suite is run using Cargo.

```bash
cargo test
```

### Python

The `python_scripts/` directory contains various scripts for testing and comparing results against the `pyFAI` library. These can be run as standard Python scripts after the `dioptrin` package has been built and installed.

For example, to compare 1D integration results:

```bash
python python_scripts/integration_test_check_1d.py
```

## Development Conventions

-   **Formatting:** The project enforces standard Rust formatting using `rustfmt`. Before committing, ensure your code is formatted with `cargo fmt --all`.
-   **Linting:** The CI pipeline runs `cargo clippy` to catch common errors and improve code quality. It is recommended to run `cargo clippy --all-targets -- -D warnings` locally to check for issues.
-   **Dependencies:** Python dependencies are managed in `pyproject.toml` and locked with `uv.lock`.
