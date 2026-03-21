# Multi-Detector Integration

Dioptrin's `MultiIntegrator` merges integration results from multiple detectors into a single diffraction pattern, similar to pyFAI's `MultiGeometry`.

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

This ensures consistent scaling regardless of detector distance, pixel size, or orientation. The per-pixel normalization (`solid_angle * polarization`) already incorporates detector geometry, so no extra scaling factor is needed at merge time.

## Requirements

All integrators in a `MultiIntegrator` must:

1. Have the **same `radial_unit`** (e.g., all `"q_A^-1"`)
2. Have an **explicit `radial_range`** set (so all detectors share the same bin grid)
3. Have the **same `radial_range`** values
4. **Not** use the `"histogram"` method (use `"counting"`, `"pixel_split"`, or `"supersampled"`)

## Python API

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

The `radial_range` must cover the combined angular range of all detectors. To find appropriate bounds:

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
| Solid angle correction | Extra factor at merge | Built into per-pixel normalization |
| Radial range | Auto or explicit | Explicit (required) |
| Methods | CSR, LUT, histogram | Counting, pixel-split, supersampled |
| Error propagation | Supported | Supported |
| Batch processing | Manual loop | Built-in `batch1d` |
