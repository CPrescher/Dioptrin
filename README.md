# Dioptrin — Fast 1D/2D diffraction integration


[Physical detector geometry](docs/detectors.md) includes a frozen detector
catalogue, orientations, nonuniform/curved quadrilateral coordinates, Fit2D
splines and conversion. Attach a detector descriptor to use that physical
mapping consistently for integration and refinement.

Dioptrin provides fast 1D/2D diffraction integration using sparse lookup tables.
It supports counting, pixel-splitting, and supersampled integration methods, and integrates in terms
of two-theta (tth) and azimuthal coordinates. The `Integrator` handles geometry updates,
polarization/solid-angle corrections, and optional masking for each frame.

## User manual

The [user manual](https://dioptrin.readthedocs.io/en/latest/) collects the guides
and API examples. Its [Markdown source](docs/index.md) is kept in this repository.
See [integration performance](https://dioptrin.readthedocs.io/en/latest/PERFORMANCE.html)
for masks, float64 arithmetic and shared plans. File-reading comparisons and SIMD
decoder measurements belong in the [dioptrin-io manual](https://dioptrin-io.readthedocs.io/en/latest/PERFORMANCE.html).

<!-- manual-rust-start -->

## Supersampling setup

Supersampling counts a pixel's repeated contributions to each output bin while
generating its subpixels, then assembles the unique contributions into an exactly
sized CSR table. It avoids a detector-wide list of all `pixels × n_ss²` events
and the associated global sort. The same float64 geometry arithmetic, bin-edge
rules, sorted pixel order and normalized weights are preserved in 1D and 2D.
Geometry and the completed table remain shared by native batch workers.

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

<!-- manual-rust-end -->

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

### Fast accumulation (default)

```python
integrator = dioptrin.Integrator(
    "calibration.poni", method="supersampled", n_ss=5  # fast_math=True by default
)
result = integrator.integrate1d(data, 2000, errors=True)
integrator.set_fast_math(False)  # select strict accumulation
```

`fast_math` is enabled by default starting in 0.6.0. Set `fast_math=False`
when constructing an integrator to retain strict accumulation. Fast accumulation
uses four partial float64 sums and fused multiply-add where hardware support
is available. This changes summation order
and rounding, so results need not be bit-identical. Geometry, subpixel sampling,
CSR weights, correction formulas and Poisson propagation are unchanged. No
finite-only assumptions or global compiler fast-math flags are enabled.

The option applies to LUT-based 1D/2D integration, batches and intermediate sums used
by `MultiIntegrator`. Histogram and `azimuthal_std` retain their existing
algorithms. `fast_math` is readable as a property and can be changed with
`set_fast_math`; changing it is rejected while an asynchronous batch holds the
integrator. Calibration caches are reused when switching modes.

### Fixed and dynamic masks

Both modes retain float64 pixel-splitting weights, corrections and accumulators:

```python
integrator = dioptrin.Integrator("calibration.poni", mask_mode="fixed")
integrator.set_mask(detector_mask)

# For masks that depend on each image:
integrator.set_mask_mode("dynamic")
for image in images:
    integrator.set_mask(make_mask(image))
    result = integrator.integrate1d(image, 2000, errors=True)
```

`fixed` is the default. Fast 1D CSC execution caches active pixels and bin
normalization. The CSR path removes masked entries from a cached execution
table. Editing or clearing the mask refreshes the dependent caches; detector
geometry and original pixel weights remain available, so previously masked
pixels can be restored correctly.

`dynamic` retains a geometry-only CSC table and accumulates signal,
normalization and Poisson variance together for each image. Four private
histograms reduce dependent updates to nearby bins. Batches parallelize across
frames; direct single-image calls can parallelize over pixel ranges. With
`fast_math=False`, dynamic mode uses fused CSR to preserve strict summation
order. Sparse tables too large for compact CSC also use fused CSR.

Both policies apply to 1D/2D LUT integration and typed TIFF inputs. Histogram
and azimuthal-std algorithms retain their existing behavior. Fast arithmetic
can differ slightly with mask policy and thread count because of summation
order. Switching policy is rejected while an asynchronous batch owns the
integrator. A batch still uses one mask; dynamic mode lets callers replace it
between frame integrations and does not generate masks itself.

Compacted CSR or transposed CSC requires additional storage alongside the
original geometry table. Policy and layout switches retain valid tables and
normalization caches; mask, correction or geometry changes invalidate dependent
state. The `mask_mode` property reports the current choice. Applications
can select a policy from their enabled masking tools; Dioptrin does not infer
whether a supplied mask will change on the next image.

### Batch Processing

For processing multiple files, use the batch methods which efficiently share geometry and LUT cache:

For synchronous `batch1d` calls with C-contiguous float64 NumPy arrays, image
pixels are borrowed directly until all native workers finish. This avoids an
additional full copy of the input batch. Inputs must remain unchanged during the
call and must not be mutated concurrently. Array batches use the global Rayon
pool; configure it before the first integration with `set_rayon_threads` or
`RAYON_NUM_THREADS`. File and asynchronous streaming paths retain ownership of
their frame buffers. Standard CSR integrations compute Poisson variance directly
from each pixel, without a full variance-image temporary. Output arrays are
allocated for each result. Histogram and azimuthal-spread paths keep their
existing temporary-buffer behavior.

Internal CSR execution tables use checked 32-bit pixel indices when possible,
with a full-width fallback for larger indices. Float64 weights and public Rust
LUT types are unchanged. Per-bin normalization is cached separately for strict
and fast accumulation, preserving each mode's reduction order. Mask and
polarization changes invalidate these denominators; geometry and grid changes
rebuild the execution table.

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

Merge results from multiple detectors into a single diffraction pattern or 2D cake using `MultiIntegrator`:

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

# Shared-grid 2D cake: fractional polygon splitting is the default.
result2d = multi.integrate2d(
    images=[img1, img2], n_radial=2000, n_azimuthal=360,
    errors=True, normalization="absolute",
)
# Flattened output uses azimuthal_index * n_radial + radial_index.
# Absolute scaling includes each detector's pixel area / distance squared.

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

<!-- manual-python-end -->

## License

Dioptrin is open-source software released under the [MIT license](LICENSE).
Both the Rust library and Python wheels can be used without activation,
license keys, or expiration checks.

Install a pre-built wheel with `python -m pip install dioptrin`, or build the
Python extension from this repository with `maturin develop --release`.
Source archives are also included in new releases.

The Python names `LicenseError`, `LicenseNotFoundError`, `LicenseExpiredError`,
and `validate_license()` remain available for compatibility with existing
applications. `validate_license()` is a no-op; integration never raises these
legacy exceptions. Versions before 0.7.0 retain their original activation behavior.

## Benchmarks

Run integration benchmarks:
```
cargo bench --bench integrate
```

Run LUT build benchmarks:
```
cargo bench --bench build
```

## pyFAI-compatible full pixel splitting

Select `method="pyfai_full"` in Python or `IntegrationMethod::PyfaiFull`
in Rust to use approximate pyFAI full-histogram splitting for **both 1D and
2D**. The default `pixel_split` convention is unchanged. `histogram` remains
the separate unsplit legacy method.

```python
ai = dioptrin.Integrator("calibration.poni", method="pyfai_full", unit="2th_deg")
pattern = ai.integrate1d(image, 2000, errors=True)
cake = ai.integrate2d(image, 2000, 360, errors=True)
```

The 1D path follows signed radial line-area integration and renormalizes
range-clipped pixels. Optional sector-aware Rust tables are available through
`lut::build_lut1d_pyfai_csr`; azimuthal bounds are radians. The 2D path reuses
the shared-grid compatibility rasterizer. Singular beam pixels use nonnegative
detector-plane fractions; negative numerical weights are discarded. This is
approximate compatibility, not bitwise reproduction of every pyFAI boundary
case. Corrections, variance propagation, CSR/CSC plans and typed-frame input
reuse the existing native execution paths.
