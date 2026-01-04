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

All examples default to `test_data/CeO2_Pilatus1M.tif` and `test_data/v1.poni` if no arguments are provided.

## Benchmarks

Run integration benchmarks:
```
cargo bench --bench integrate
```

Run LUT build benchmarks:
```
cargo bench --bench build
```
