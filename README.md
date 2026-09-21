# Dioptrin — Fast 1D/2D diffraction integration with CSR LUTs

Dioptrin provides fast 1D/2D diffraction integration using compressed sparse row (CSR) lookup tables.
It supports counting, pixel-splitting, and supersampled integration methods, and integrates in terms
of two-theta (tth) and azimuthal coordinates. The `Integrator` handles geometry updates,
polarization/solid-angle corrections, and optional masking for each frame.

## 1D integration example

```rust
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator};
use dioptrin::poni::Poni;

fn main() {
    let poni = Poni::load("test_data/v1.poni").expect("load poni");
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).expect("load frame");
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let mut integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_polarization_factor(0.99);

    let n_points = 1250;
    let (tth, intensity) = integrator.integrate1d(&frame, n_points);

    println!("1D bins: {}", tth.len());
    println!("1D intensity values: {}", intensity.len());
}
```

## 1D integration with errors

```rust
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator};
use dioptrin::poni::Poni;

fn main() {
    let poni = Poni::load("test_data/v1.poni").expect("load poni");
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).expect("load frame");
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let mut integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_polarization_factor(0.99);

    let n_points = 1250;
    let (tth, intensity, sigma) = integrator.integrate1d_with_errors(&frame, n_points);

    println!("1D bins: {}", tth.len());
    println!("1D intensity values: {}", intensity.len());
    println!("1D sigma values: {}", sigma.len());
}
```

## Azimuthal standard deviation (spottiness)

For single-detector 1D integration, request the spread of corrected pixel
intensities within each radial bin independently of Poisson uncertainties:

```python
result = integrator.integrate1d(image, 1000, errors=True, azimuthal_std=True)
mean = result.intensity
spread = result.azimuthal_std
poisson_uncertainty = result.errors

# Also available for TIFF paths, array batches, and streaming generators:
results = integrator.batch1d(images, 1000, azimuthal_std=True)
for item in integrator.batch1d_iter(frame_generator(), 1000, azimuthal_std=True):
    if item.is_ok():
        process(item.result.azimuthal_std)
```

In Rust, use `Integrator::integrate1d_with_azimuthal_std(&frame, n_points, errors)`
or `integrate1d_with_azimuthal_std_from_slice`. These return
`dioptrin::lut::AzimuthalIntegrationResult`. The batch task is
`dioptrin::batch::Integration1DWithAzimuthalStd`.

For raw signal `s`, solid-angle/polarization normalization `n`, and pixel fraction
`c`, the corrected intensity is `x = s/n` and its weight is `w = c*n`:

```text
mean     = sum(w*x) / sum(w) = sum(c*s) / sum(c*n)
variance = sum(w*(x-mean)^2) / sum(w)
azimuthal_std = sqrt(variance)
```

This is a **population standard deviation**, not RMS or an error bar on the
mean. It uses the same weighting as the integrated intensity. Counting,
pixel-split and supersampled methods reuse the cached CSR lookup table.
The legacy Histogram method retains its equal weighting of pre-corrected pixels
and automatic radial range. The spread uses a shifted two-pass calculation to
avoid cancellation for large backgrounds with small variations. Poisson errors
continue to use squared pixel fractions and are calculated independently.

Masked or nonfinite signals, nonpositive/nonfinite normalizations and zero-weight
contributions are excluded. Empty bins have **NaN spread** (intensity and errors
remain zero); one contributing pixel has zero spread. Without the option,
`result.azimuthal_std` is `None` and the existing integration path is unchanged.
This option is not currently exposed by `MultiIntegrator` or 2D integration.

The statistic is pixel-weighted, not an equal-angle average of a cake image.
Noise, radial gradients and artifacts can contribute alongside texture; changing
binning or masks can change the result. pyFAI's `error_model="azimuthal"` agrees
for equal weights, but its variance propagation uses squared normalization and
pixel-fraction weights. Its `std` is **not a drop-in equivalent** under unequal
weights. Backends must use the formula above for a consistent spottiness measure.

Validation: `python -m unittest discover -s python_examples -p test_azimuthal_std.py`.
The optional pyFAI comparison requires the `test` extras. Benchmark the enabled
and disabled paths with `python python_examples/bench_azimuthal_std.py`.

## 2D integration example (azimuthal range)

```rust
use std::f64::consts::PI;

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator};
use dioptrin::poni::Poni;

fn main() {
    let poni = Poni::load("test_data/v1.poni").expect("load poni");
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).expect("load frame");
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let mut integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_polarization_factor(0.99);

    let n_tth = 1250;
    let n_azimuthal = 360;
    let azimuthal_step = 2.0 * PI / n_azimuthal as f64;
    let azimuthal_start = 0.0;

    let cake = integrator.integrate2d_with_azimuthal_range(&frame, n_tth, n_azimuthal, azimuthal_start, azimuthal_step);

    println!("2D bins: {}", cake.len());
}
```

## Batch processing multiple files

The `batch` module provides efficient parallel processing of multiple files with a shared integrator.
This minimizes memory usage by sharing geometry arrays and LUT cache across workers.

```rust
use std::sync::Arc;
use dioptrin::batch::{process_files, Integration1D, Integration1DWithErrors};
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::poni::Poni;

fn main() {
    // Create shared integrator (geometry + LUT cache shared across all workers)
    let poni = Poni::load("test_data/v1.poni").expect("load poni");
    let geometry = Geometry::new(1043, 981, poni);
    let integrator = Arc::new(
        Integrator::new(geometry)
            .with_method(IntegrationMethod::PixelSplit)
            .with_radial_unit(RadialUnit::TthDeg)
    );

    // List of files to process
    let files = vec![
        "image1.tif".to_string(),
        "image2.tif".to_string(),
        "image3.tif".to_string(),
    ];

    // Process with 4 workers, 2 threads per worker (8 total threads)
    let task = Integration1D { n_bins: 3000 };
    let results = process_files(&files, integrator, task, 4, 2);

    // Handle results
    for (filename, result) in results {
        match result {
            Ok((radial, intensity)) => {
                println!("{}: {} bins integrated", filename, radial.len());
            }
            Err(e) => eprintln!("{}: error - {}", filename, e),
        }
    }
}
```

**Available integration tasks:**
- `Integration1D { n_bins }` - 1D integration → `(radial, intensity)`
- `Integration1DWithErrors { n_bins }` - 1D with errors → `(radial, intensity, errors)`
- `Integration2D { n_radial, n_azimuthal }` - 2D integration → `intensity_flat`
- `Integration2DWithErrors { n_radial, n_azimuthal }` - 2D with errors → `(intensity_flat, errors_flat)`
- `Integration2DWithAzimuthalRange` - 2D with azimuthal restriction
- `Integration2DWithAzimuthalRangeAndErrors` - 2D with range and errors

For custom processing logic, implement the `IntegrationTask` trait on your own struct.

## Raw LUT integration example

If you want direct control over LUT building and integration, you can use the LUT APIs directly.
This is useful when you want to reuse a LUT across multiple frames or manage caching yourself.

```rust
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::RadialUnit;
use dioptrin::lut::{build_lut2d_ps_csr, integrate2d};
use dioptrin::poni::Poni;

fn main() {
    let poni = Poni::load("test_data/v1.poni").expect("load poni");
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).expect("load frame");
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();
    let lut2d = build_lut2d_ps_csr(&geometry, &tth, &azimuthal, 1250, 360, RadialUnit::TthRad);

    let intensity = frame.data.clone();
    let norm = vec![1.0; intensity.len()];
    let cake = integrate2d(&intensity, &lut2d, None, &norm);

    println!("2D bins: {}", cake.len());
}
```

## Examples

Runnable examples are provided in the `examples/` directory:

### 1D Integration
```
cargo run --example integrate1d [image.tif] [calibration.poni]
```
Loads a PONI file and TIFF image, performs 1D azimuthal integration, and prints the results.

### 2D Integration
```
cargo run --example integrate2d [image.tif] [calibration.poni]
```
Performs 2D integration with radial and azimuthal binning.

### Timing Benchmark
```
cargo run --release --example timing1d [image.tif] [calibration.poni]
```
Measures 1D integration performance with a warm-up phase and 100 iterations.

### Thread Scaling Benchmark
```
cargo run --release --example benchmark_threads [image.tif] [calibration.poni]
```
Benchmarks 1D and 2D integration performance across different thread counts (1 to system maximum), showing timing and speedup for each configuration.

### Pipeline Throughput Benchmark
```
cargo run --release --example benchmark_pipeline [image.tif] [calibration.poni]
```
Measures real throughput of the full pipeline (load file + integrate) with different combinations of parallel workers and threads per worker. Helps find the optimal configuration for batch processing.

### Batch Processing API
```
cargo run --release --example batch_api [image.tif] [calibration.poni]
```
Demonstrates the `batch` module API for processing multiple files in parallel. Shows different integration methods (1D, 2D, with/without errors) and compares sequential vs. parallel performance with various worker/thread configurations.

All examples default to `test_data/CeO2_Pilatus1M.tif` and `test_data/v1.poni` if no arguments are provided.

## Python Bindings

Dioptrin provides Python bindings via PyO3. Build with `maturin develop` (or `maturin develop --release` for optimized builds).

### Basic Usage

```python
import dioptrin

# Create an integrator from a PONI calibration file
integrator = dioptrin.Integrator(
    "calibration.poni",
    method="pixel_split",      # "counting", "pixel_split", "histogram", or "supersampled"
    polarization_factor=0.99,  # optional
    unit="2th_deg"             # "2th_rad", "2th_deg", "q_A^-1", or "q_nm^-1"
)

# 1D integration from TIFF file
result = integrator.integrate1d("pattern.tif", n_points=1000)
radial, intensity = result.radial, result.intensity

# 1D integration with error estimates
result = integrator.integrate1d("pattern.tif", n_points=1000, errors=True)
errors = result.errors

# 1D integration from NumPy array
import numpy as np
data = np.array(image_data, dtype=np.float64)
result = integrator.integrate1d(data, n_points=1000)

# 2D integration from TIFF file
result = integrator.integrate2d("pattern.tif", n_radial=100, n_azimuthal=360)
cake = np.array(result.intensity).reshape(result.n_azimuthal, result.n_radial)

# 2D integration from NumPy array
result = integrator.integrate2d(data, n_radial=100, n_azimuthal=360)
```

### Batch Processing

For processing multiple files, use the batch methods which efficiently share geometry and LUT cache:

```python
files = ["image1.tif", "image2.tif", "image3.tif"]

# Batch 1D integration - returns list of BatchResult
results = integrator.batch1d(files, n_points=1000)
for r in results:
    if r.is_ok():
        print(f"{r.filename}: {len(r.result.intensity)} bins")
        # r.result.radial, r.result.intensity available
    else:
        print(f"{r.filename}: error - {r.error}")

# Batch 1D with error estimates
results = integrator.batch1d(files, n_points=1000, errors=True)
for r in results:
    if r.is_ok():
        # r.result.errors is now populated
        print(f"{r.filename}: {r.result.errors[:5]}")

# Batch 2D integration
results = integrator.batch2d(files, n_radial=100, n_azimuthal=360)
for r in results:
    if r.is_ok():
        # r.result.intensity is flat array [n_radial * n_azimuthal]
        # r.result.n_radial and r.result.n_azimuthal for reshaping
        cake = np.array(r.result.intensity).reshape(r.result.n_azimuthal, r.result.n_radial)

# Batch 2D with azimuthal range restriction
results = integrator.batch2d(
    files,
    n_radial=100,
    n_azimuthal=90,
    azimuthal_range=(0, 1.57)  # radians
)
```

### Streaming Batch Processing

For large batches, use streaming iterators to process results as they complete:

```python
# Stream results as they complete
for r in integrator.batch1d_iter(files, n_points=1000):
    if r.is_ok():
        process(r.result)
    # Progress tracking available on the iterator
    # (note: progress is per-result, not real-time)

# 2D streaming with all options
for r in integrator.batch2d_iter(
    files,
    n_radial=100,
    n_azimuthal=360,
    errors=True,
    azimuthal_range=(0, 3.14),
    num_workers=4,
    num_threads=2
):
    if r.is_ok():
        process(r.result)
```

### Result Types

- **`IntegrationResult`**: Unified result with optional fields
  - `radial`: Radial bin centers (1D only)
  - `intensity`: Integrated intensities
  - `errors`: Error estimates (if `errors=True`)
  - `n_radial`, `n_azimuthal`: Dimensions for reshaping 2D results

- **`BatchResult`**: Wrapper for batch results
  - `filename`: Path of processed file
  - `result`: `IntegrationResult` or `None` if failed
  - `error`: Error message or `None` if successful
  - `is_ok()`, `is_err()`: Convenience methods

- **`BatchIterator`**: Streaming iterator with progress tracking
  - `total`: Total number of files
  - `completed`: Files processed so far
  - `progress`: Fraction complete (0.0 to 1.0)

### Multi-Detector Integration

Merge results from multiple detectors into a single diffraction pattern using `MultiIntegrator`:

```python
import dioptrin

# Create one integrator per detector with a shared radial range and unit
int1 = dioptrin.Integrator(
    "detector1.poni",
    method="pixel_split",
    unit="q_A^-1",
    radial_range=(0.5, 25.0),
)
int2 = dioptrin.Integrator(
    "detector2.poni",
    method="pixel_split",
    unit="q_A^-1",
    radial_range=(0.5, 25.0),
)

# Create multi-detector integrator
multi = dioptrin.MultiIntegrator(integrators=[int1, int2])

# Integrate — one image per detector
result = multi.integrate1d(images=[img1, img2], n_points=2000)
print(result.radial, result.intensity)

# With error estimates
result = multi.integrate1d(images=[img1, img2], n_points=2000, errors=True)

# Batch processing multiple frame sets
image_sets = [[img1_a, img2_a], [img1_b, img2_b]]
results = multi.batch1d(images=image_sets, n_points=2000)
```

All integrators must share the same `unit` and `radial_range`. The `"histogram"` method is not supported.
See `docs/MULTI_DETECTOR.md` for full details including Rust API and pyFAI comparison.

### Thread Configuration

```python
# Get/set global Rayon thread count
threads = dioptrin.rayon_num_threads()
dioptrin.set_rayon_threads(8)

# Batch methods accept worker/thread configuration
results = integrator.batch1d(
    files,
    n_points=1000,
    num_workers=4,    # Parallel workers
    num_threads=2     # Threads per worker (for Rayon)
)
# Total threads used: num_workers * num_threads
```

## Licensing

Dioptrin uses offline ed25519-signed license keys for its distributed Python wheels. The Rust library itself is always unrestricted. Dioptrin is **not published to PyPI** — licensed wheels are distributed privately to authorized users.

### When is a license required?

| Install method | License required? |
|---|---|
| `cargo build` / `cargo test` (Rust) | No — never |
| `pip install .` / `maturin develop` (from source) | No |
| Pre-built wheel (from CI / private distribution) | **Yes** |

The `licensed` Cargo feature controls this. Building from source uses `features = ["python"]` (see `pyproject.toml`), so no license check is compiled in. The CI release workflow builds wheels with `features = ["python", "licensed"]`, which embeds the license check.

### Installing a pre-built wheel

Licensed wheels are provided as `.whl` files. Install directly:

```bash
pip install dioptrin-0.3.0-cp312-cp312-manylinux_2_17_x86_64.whl
```

### Activating a license

A license file looks like this:

```
email=user@university.edu
expires=2027-01-01
---
<base64-encoded ed25519 signature>
```

There are two ways to provide it:

**Option 1: Environment variable** (recommended for HPC/beamline)

```bash
export DIOPTRIN_LICENSE="$(cat /path/to/license.key)"
```

**Option 2: File** (recommended for workstations)

Place the license file at `~/.dioptrin/license.key`.

If no valid license is found, creating an `Integrator` raises `dioptrin.LicenseError`. Downstream applications like Dioptas can catch this to fall back gracefully:

```python
try:
    integrator = dioptrin.Integrator("calibration.poni")
except (dioptrin.LicenseError, ImportError):
    # Fall back to pyFAI
    import pyFAI
    integrator = pyFAI.load("calibration.poni")
```

### Generating licenses (developers only)

The `keygen` binary signs license files. It requires the `licensed` feature:

```bash
# Generate a new ed25519 keypair (only needed once, writes to keys/)
cargo run --features licensed --bin keygen -- --generate-keypair

# Sign a license for a user
cargo run --features licensed --bin keygen -- \
    --email user@university.edu \
    --expires 2027-01-01 \
    --output license.key

# Use a custom private key location
cargo run --features licensed --bin keygen -- \
    --email user@university.edu \
    --expires 2027-01-01 \
    --private-key /path/to/private.key
```

The `keys/` directory contains:
- `public.key` — 32-byte ed25519 public key (committed to repo, compiled into the binary)
- `private.key` — 32-byte ed25519 private key (**gitignored**, keep this secret)

## Benchmarks

Run integration benchmarks:
```
cargo bench --bench integrate
```

Run LUT build benchmarks:
```
cargo bench --bench build
```
