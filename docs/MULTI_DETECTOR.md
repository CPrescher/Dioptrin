# Multi-Detector Integration

Dioptrin's `MultiIntegrator` merges integration results from multiple detectors into a single diffraction pattern or 2D cake, similar to pyFAI's `MultiGeometry`.

## How It Works

Each detector contributes raw signal, normalization, and variance to shared radial bins. These are summed across detectors, then divided once at the end:

```
For each detector i:
    sum_signal_i, sum_norm_i, sum_var_i = raw_integrate(data_i)

Merge:
    total_signal[bin] = Sum(sum_signal_i[bin])
    total_norm[bin]   = Sum(sum_norm_i[bin])
    total_var[bin]    = Sum(sum_var_i[bin])

Final:
    intensity = total_signal / total_norm
    sigma     = sqrt(total_var) / total_norm
```

The existing **1D** path uses relative per-pixel solid angle × polarization.
It does **not** include the detector constant `pixel1 * pixel2 / distance**2`
used by pyFAI MultiGeometry with `correctSolidAngle=True`. Unequal detector
geometries must not be assumed equivalent on that scale. The new **2D** path
provides an explicit `normalization="absolute"` option for this convention;
1D behavior is unchanged.

## Requirements

All integrators in a `MultiIntegrator` must:

1. Have the **same `radial_unit`** (e.g., all `"q_A^-1"`)
2. Have an **explicit `radial_range`** set (so all detectors share the same bin grid)
3. Have the **same `radial_range`** values
4. **Not** use the `"histogram"` method (use `"counting"`, `"pixel_split"`, or `"supersampled"`)

## Shared-grid 2D cakes

Available in **Dioptrin 0.11.0 and later**. Install or upgrade in the same
Python environment as your application:

```sh
python -m pip install --upgrade "dioptrin>=0.11.0"
```

Create one integrator per detector, with a common radial unit and range.
For example, after loading `img1` and `img2` in the orientation expected by
their calibrations:

```python
import numpy as np
import dioptrin

integrators = [
    dioptrin.Integrator(
        path, method="pixel_split", unit="q_A^-1",
        radial_range=(0.0, 20.0), polarization_factor=0.99,
    )
    for path in ["detector1.poni", "detector2.poni"]
]
# Apply any detector masks to integrators before constructing MultiIntegrator.
multi = dioptrin.MultiIntegrator(integrators)  # fractional polygons by default
result = multi.integrate2d(
    images=[np.ascontiguousarray(img1, dtype=np.float64),
            np.ascontiguousarray(img2, dtype=np.float64)],
    n_radial=2000,
    n_azimuthal=360,
    errors=True,
    azimuthal_range=(-np.pi, np.pi),  # EDGES in radians
    normalization="absolute",        # pyFAI MultiGeometry solid-angle scale
    normalization_factors=[1.0, 1.0], # per-detector monitor denominators
)
cake = np.asarray(result.intensity).reshape(360, 2000)
radial_centers = np.asarray(result.radial)      # configured radial unit
azimuth_centers_deg = np.degrees(result.azimuthal)
sigma = np.asarray(result.errors).reshape(360, 2000)
```

The exact Python signature is:

```python
MultiIntegrator.integrate2d(
    images, n_radial, n_azimuthal, errors=False, azimuthal_range=None,
    *, normalization="relative", normalization_factors=None,
) -> IntegrationResult
```

Each detector retains its own shape, pixel sizes, distance, PONI offsets,
rotations, mask and polarization. Images are C-contiguous 2D float64 arrays or
TIFF paths, in detector order; mixed lists are supported by this 2D method.
TIFF loading retains Dioptrin's existing vertical flip; NumPy arrays retain
caller orientation. Supply the same pixel order as the calibration. Configure
masks before constructing the multi-integrator: its shared Integrator references
prevent later Python setters from acquiring unique ownership. Reconstruct the
multi-integrator when calibration or mask settings change.

### Grids, methods and units

- Both bin counts must be positive. All detectors require identical finite,
  increasing, nonnegative `radial_range` edges in the selected radial unit.
- Units: `2th_rad`, `2th_deg`, `q_A^-1`, `q_nm^-1`. q requires a positive finite
  wavelength in meters for each detector. Geometry distances and pixel sizes
  use meters; rotations and azimuthal coordinates use radians.
- Azimuthal edges default to `(-pi, pi)`. The start is in `[-2*pi, 2*pi]` and
  the increasing span is at most `2*pi`. A sector crossing the discontinuity
  can be expressed as `(3*pi/4, 5*pi/4)`. Angles wrap over `2*pi`, then the
  requested sector is **clipped**. They never fold over the sector width.
- Bins are half-open. Out-of-range centers/samples are excluded; split pixels
  contribute their in-range fractions in the default fractional mode and in
  supersampling. The optional pyFAI histogram convention renormalizes clipped
  pixels, as described below. Empty bins contain zero intensity/std.
- Methods: `counting` (pixel centers), `pixel_split` (by default quadrilateral clipping in
  radial/azimuthal space, with a beam-pixel sampling fallback), `supersampled` (`n_ss > 0`, n_ss² samples/pixel).
  `histogram` is unsupported. Detectors can use different supported methods.
- Results flatten as `azimuthal_index * n_radial + radial_index`. Both
  coordinate arrays contain bin centers; neither contains edges.

### Splitting conventions and azint-style subpixel splitting

`MultiIntegrator(integrators, *, pixel_split_mode="fractional")` keeps exact
polygon-intersection weights as the default. The optional value
`"pyfai_histogram"` uses approximate line-area rasterization, corner clipping
and clipped-pixel renormalization to reproduce pyFAI's full histogram engine
more closely. It is a compatibility choice, independent of `normalization`.
Rust uses `multi.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram)`; the
raw single-detector API selects it through `grid.with_pixel_split_mode(...)`.
It affects only the new shared-grid 2D `PixelSplit` path.

Pixels enclosing or touching the polar origin cannot be described reliably by
a single polar quadrilateral. Both modes fall back to detector-plane sampling
there: 128 × 128 strata with four symmetric samples per stratum (65,536 samples
per singular pixel). The symmetric pattern avoids bias from grid points lying
exactly on azimuthal-bin edges. Samples from a pixel are combined before
squaring its weight for variance. Compatibility mode also discards negative
numerical raster areas and renormalizes the positive areas; it deliberately
does not reproduce pathological negative pyFAI weights. This is approximate,
not bitwise compatibility, and the singular fallback still clips by sample
fractions rather than renormalizing a partly included pixel.

For splitting throughout the detector using the same midpoint-subdivision
approach as [azint](https://maxiv-science.github.io/azint/), use the existing
`supersampled` method on **each detector**:

```python
integrators = [
    dioptrin.Integrator(path, method="supersampled", n_ss=21,
                       unit="q_A^-1", radial_range=(0.0, 8.0))
    for path in ["detector1.poni", "detector2.poni"]
]
multi = dioptrin.MultiIntegrator(integrators)
result = multi.integrate2d([img1, img2], 2000, 360,
                           normalization="absolute", errors=True)
```

`n_ss=21` means 441 sample centers per original pixel. A bin's weight is the
number of its samples divided by 441. This avoids polar-polygon ambiguity and
handles the seam sample by sample. Increasing `n_ss` improves geometric
resolution but increases table-preparation work; cached frames reuse the table.
Check convergence, for example between 21 and 41, for the chosen cake grid.
The sampling parameter is not a guaranteed accuracy threshold.

This is the same **class of geometric splitting** used by azint for rectangular
pixels; no azint runtime dependency is required. It does not duplicate azint's
complete integration estimator: azint averages individually corrected pixel
signals (`sum(w*I/correction)/sum(w)`), whereas this multi-detector path retains
its documented raw-sum convention (`sum(w*I)/sum(w*correction)`, with optional
absolute detector constants and monitors). Azint also uses float32 geometry
and opposite bin-edge inclusivity at exact edges. See the
[azint source](https://github.com/maxiv-science/azint/blob/4f78cc3b1d8cab67128c9679f55de6340c68f191/azint.cpp)
and [investigation](validation/multi2d-splitting.md) for the comparison limits.

### Normalization and uncertainties

For pixel p of detector d and output bin b, let `w[d,p,b]` be 1 for counting,
polygon intersection area / original pixel polygon area for pixel splitting,
or the fraction of subpixel samples in the bin for supersampling. Let `S` be
relative solid angle, `P` polarization and `M` the positive monitor value:

```text
signal[b]   = sum(d,p) w[d,p,b] * image[d,p]
norm[b]     = sum(d,p) w[d,p,b] * S[d,p] * P[d,p] * M[d] * A[d]
variance[b] = sum(d,p) w[d,p,b]^2 * max(image[d,p], 0)
intensity[b] = signal[b] / norm[b]
std[b]       = sqrt(variance[b]) / norm[b]
```

`normalization="relative"` (default) sets `A=1`. `normalization="absolute"`
sets `A=pixel1*pixel2/distance**2` **independently for every detector**. Both
modes always apply relative solid angle and configured polarization.
`normalization_factors` defaults to ones; it scales the denominator, matching
pyFAI monitor normalization, without scaling signal or variance. Arbitrary
signal multipliers can be applied to images before integration, but then the
Poisson assumption must be reassessed. The API does not accept explicit
variance arrays, dark/flat arrays or an azimuthal error model.

`errors=True` assumes independent Poisson counts, including independence between
detectors. Repeated subpixel contributions from one pixel to one bin are
combined before squaring their weight. Correlations between output bins caused
by pixel splitting are not returned.

### Qt/React/Tauri Dioptas adapter

Use `Multi2dOptions { normalization: MultiNormalization::Absolute, .. }` in
Rust, or `normalization="absolute"` in Python, to retain Qt's
`MultiGeometry(...).integrate2d(correctSolidAngle=True)` **intensity scale**.
Pass one image/calibration/mask per detector, common UI-selected radial and
azimuthal edges, and any per-detector monitors. Merge the returned raw sums
before division; do not average detector cakes or create an image mosaic.
Convert returned azimuthal coordinates to degrees only for display.

For closer full-histogram compatibility, also construct
`MultiIntegrator(integrators, pixel_split_mode="pyfai_histogram")`. The tested
beam/seam case then agrees within 5.7e-7 relative intensity and 1.4e-6 relative
std; the default fractional convention remains different by design.

For a matched no-splitting reference, use Dioptrin `counting` and pyFAI
`("no", "histogram", "cython")`, with the same effective edges, masks,
polarization and monitors. This does not assert identical output for different
splitting algorithms. In particular, Dioptrin clips polygon intersections in
float64; pyFAI's full histogram method distributes integrated line areas and
clips/recenters vertices. Their seam, beam-center and partial-range behavior
can differ. Supersampling offers an alternative when polar pixel polygons
are poorly resolved near the beam center.

The old ordinary `Integrator.integrate2d` chooses a detector-local radial grid;
it does not honor its `radial_range` as shared 2D edges. The new shared-grid
raw Rust API below is available for one detector too. Legacy polygon 2D LUTs
use row-normalized **areas**, rather than fractions of original pixel areas,
so a nonuniform single-detector polygon cake can also differ from the new
fractional path. Existing ordinary 2D and all 1D outputs are unchanged.

### Fallible Rust API

```rust
use dioptrin::multi::{MultiIntegrator, Multi2dOptions, MultiNormalization};

// Each Integrator already has a matching radial_unit and radial_range, and its
// own geometry/mask/polarization. try_new returns validation errors, not panics.
let multi = MultiIntegrator::try_new(vec![int1, int2])?;
let options = Multi2dOptions {
    azimuthal_range: (-std::f64::consts::PI, std::f64::consts::PI),
    errors: true,
    normalization: MultiNormalization::Absolute,
    normalization_factors: Some(vec![1.0, 1.0]),
};
let result = multi.integrate2d(&[&frame1, &frame2], 2000, 360, &options)?;
// Or borrow one (&[f64], rows, columns) per detector:
let result = multi.integrate2d_from_slices(
    &[(&data1, rows1, cols1), (&data2, rows2, cols2)], 2000, 360, &options,
)?;
// result.radial, result.azimuthal, result.intensity, result.errors
// result.raw: sum_signal, sum_norm, sum_variance, count, n_radial, n_azimuthal
```

`Integrator::integrate2d_raw_on_grid_from_slice(data, rows, cols, &grid, errors)`
returns an `IntermediateResult2d` on an explicit
`IntegrationGrid2d::new(n_radial, n_azimuthal, radial_edges, azimuthal_edges)?`.
This uses the same prepared kernels and fractional weights as the multi path.
Its raw normalization is relative; detector constants/monitors are applied
by MultiIntegrator. A grid is cached per Integrator and reused for later frames;
switching grids or calling a legacy ordinary 2D method rebuilds that cache.

See [the initial validation report](validation/multi2d.md),
[full-splitting investigation](validation/multi2d-splitting.md),
[first-frame and Eiger benchmarks](validation/multi2d-startup.md) and run
`python python_examples/test_multi2d.py` to reproduce the pyFAI comparisons.

```{toctree}
:hidden:

validation/multi2d
validation/multi2d-splitting
validation/multi2d-startup
```

## Python API (existing 1D)

### Setup

```python
import dioptrin

# Create one integrator per detector, all sharing the same radial range and unit
int1 = dioptrin.Integrator(
    "detector1.poni",
    method="pixel_split",
    unit="q_A^-1",
    radial_range=(0.5, 25.0),
    polarization_factor=0.99,
)
int2 = dioptrin.Integrator(
    "detector2.poni",
    method="pixel_split",
    unit="q_A^-1",
    radial_range=(0.5, 25.0),
    polarization_factor=0.99,
)

multi = dioptrin.MultiIntegrator(integrators=[int1, int2])
```

### Single Integration

```python
# One image per detector (file paths or NumPy arrays)
result = multi.integrate1d(images=[img1, img2], n_points=2000)
print(result.radial, result.intensity)

# With error estimates
result = multi.integrate1d(images=[img1, img2], n_points=2000, errors=True)
print(result.radial, result.intensity, result.errors)
```

### Batch Processing

```python
# List of image sets: [[det1_frame1, det2_frame1], [det1_frame2, det2_frame2], ...]
image_sets = [[img1_a, img2_a], [img1_b, img2_b], [img1_c, img2_c]]
results = multi.batch1d(images=image_sets, n_points=2000)

for result in results:
    print(result.radial, result.intensity)
```

### With HDF5 Data

```python
import h5py
import numpy as np
import dioptrin

int1 = dioptrin.Integrator("det1.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
int2 = dioptrin.Integrator("det2.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
multi = dioptrin.MultiIntegrator(integrators=[int1, int2])

with h5py.File("det1_data.h5", "r") as f1, h5py.File("det2_data.h5", "r") as f2:
    ds1 = f1["/entry/data/data"]
    ds2 = f2["/entry/data/data"]
    n_frames = len(ds1)

    image_sets = []
    for i in range(n_frames):
        image_sets.append([
            ds1[i].astype(np.float64),
            ds2[i].astype(np.float64),
        ])

results = multi.batch1d(images=image_sets, n_points=2000, errors=True)
```

## Rust API

```rust
use dioptrin::geometry::Geometry;
use dioptrin::integration::{Integrator, IntegrationMethod, RadialUnit};
use dioptrin::multi::MultiIntegrator;
use dioptrin::poni::Poni;

let poni1 = Poni::load("det1.poni").unwrap();
let poni2 = Poni::load("det2.poni").unwrap();

let int1 = Integrator::new(Geometry::new(0, 0, poni1))
    .with_method(IntegrationMethod::PixelSplit)
    .with_radial_unit(RadialUnit::QAngstrom)
    .with_radial_range(0.5, 25.0);

let int2 = Integrator::new(Geometry::new(0, 0, poni2))
    .with_method(IntegrationMethod::PixelSplit)
    .with_radial_unit(RadialUnit::QAngstrom)
    .with_radial_range(0.5, 25.0);

let multi = MultiIntegrator::new(vec![int1, int2]);

// Integrate
let (bins, intensity) = multi.integrate1d(&[&frame1, &frame2], 2000);

// With errors
let (bins, intensity, sigma) = multi.integrate1d_with_errors(&[&frame1, &frame2], 2000);

// From slices (for NumPy array path)
let (bins, intensity) = multi.integrate1d_from_slices(
    &[(&data1, dim1_1, dim2_1), (&data2, dim1_2, dim2_2)],
    2000,
);
```

## Choosing the Radial Range

Choose `radial_range` to cover the desired combined angular range. For 2D, it may intentionally clip detector coverage. Include pixel corners/subsamples when full split-pixel coverage is required; bin centers are not exact detector limits. A quick 1D estimate is:

```python
import dioptrin
import numpy as np

# Create integrators without radial_range first to check each detector's range
int1 = dioptrin.Integrator("det1.poni", unit="q_A^-1")
int2 = dioptrin.Integrator("det2.poni", unit="q_A^-1")

# Integrate each detector separately to see their ranges
r1 = int1.integrate1d(img1, n_points=1000)
r2 = int2.integrate1d(img2, n_points=1000)

# Find combined range
q_min = min(r1.radial[0], r2.radial[0])
q_max = max(r1.radial[-1], r2.radial[-1])
print(f"Suggested radial_range: ({q_min:.2f}, {q_max:.2f})")

# Then recreate with the shared range
int1 = dioptrin.Integrator("det1.poni", unit="q_A^-1", radial_range=(q_min, q_max))
int2 = dioptrin.Integrator("det2.poni", unit="q_A^-1", radial_range=(q_min, q_max))
multi = dioptrin.MultiIntegrator(integrators=[int1, int2])
```

## Comparison with pyFAI MultiGeometry

| Feature | pyFAI MultiGeometry | Dioptrin MultiIntegrator |
|---------|-------------------|------------------------|
| Merge method | Sum raw sums, divide once | Same |
| Solid angle correction | Relative per-pixel × detector constant | 2D: explicit relative/absolute; 1D: legacy relative |
| Radial range | Auto or explicit | Explicit (required) |
| Methods | CSR, LUT, histogram | Counting, pixel-split, supersampled |
| Error propagation | Several models/explicit variances | Poisson; squared fractional weights in new 2D |
| Batch processing | Manual loop | Built-in `batch1d`; 2D via repeated `integrate2d` |
