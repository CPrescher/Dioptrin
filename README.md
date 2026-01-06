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
use dioptrin::lut::{build_lut2d_ps_csr, integrate2d};
use dioptrin::poni::Poni;

fn main() {
    let poni = Poni::load("test_data/v1.poni").expect("load poni");
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).expect("load frame");
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();
    let lut2d = build_lut2d_ps_csr(&geometry, &tth, &azimuthal, 1250, 360);

    let intensity = frame.data.clone();
    let cake = integrate2d(&intensity, &lut2d, None);

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

## Benchmarks

Run integration benchmarks:
```
cargo bench --bench integrate
```

Run LUT build benchmarks:
```
cargo bench --bench build
```
