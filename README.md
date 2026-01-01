# Dioptrin — Fast 1D/2D diffraction integration with CSR LUTs

Dioptrin provides fast 1D/2D diffraction integration using compressed sparse row (CSR) lookup tables.
It supports counting, pixel-splitting, and supersampled integration methods, and integrates in terms
of two-theta (tth) and chi (azimuthal) coordinates. The `Integrator` handles geometry updates,
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

## 2D integration example (chi range)

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
    let n_chi = 360;
    let chi_step = 2.0 * PI / n_chi as f64;
    let chi_start = 0.0;

    let cake = integrator.integrate2d_with_chi_range(&frame, n_tth, n_chi, chi_start, chi_step);

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
    let chi = geometry.calc_chi_all();
    let lut2d = build_lut2d_ps_csr(&geometry, &tth, &chi, 1250, 360);

    let intensity = frame.data.clone();
    let cake = integrate2d(&intensity, &lut2d, None);

    println!("2D bins: {}", cake.len());
}
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
