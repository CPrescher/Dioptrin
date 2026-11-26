# Diffraction image simulation

`dioptrin::simulation` (and `dioptrin.simulate_diffraction` in Python) generates
synthetic 2D powder diffraction images for a given detector `Geometry` and
reference `Calibrant` (or pyFAI-style `.D` calibrant file). Simulated frames
use Dioptrin's exact pixel geometry, so they can be fed directly into
`Integrator`, peak pickers, and calibration routines for testing, algorithm
development, and geometry verification.

## Forward model

Given a `Geometry` with positive `poni.wavelength` (λ, in metres) and a
`Calibrant` with d-spacings `d_k` (in Ångström), each physical Bragg ring
(`λ ≤ 2 d_k`) sits at:

```text
2θ_k = 2 * arcsin(λ / (2 * d_k))
```

For each pixel `(r, c)`:

1. **Angular evaluation**: `2θ` is evaluated at the pixel centre
   `(r + 0.5, c + 0.5)` — or averaged over an `N × N` uniform subpixel grid
   when `oversampling = N` — using `Geometry::subpixel_transform()`. Flat uniform
   detectors take an affine-plane fast path; attached physical `Detector` models
   evaluate their full position mapping (splines, module gaps, orientation,
   binning, and non-planar geometry).
2. **Peak profile**: All active rings are summed using one of three unit-peak
   profiles parameterized by `fwhm_rad` (Full Width at Half Maximum in `2θ`
   radians):
   - **Gaussian** (`PeakProfile::Gaussian` / `"gaussian"`, default):
     `exp(-4 * ln(2) * (Δ2θ / fwhm)^2)`
   - **Lorentzian** (`PeakProfile::Lorentzian` / `"lorentzian"`):
     `1 / (1 + (2 * Δ2θ / fwhm)^2)`
   - **Pseudo-Voigt** (`PeakProfile::PseudoVoigt { eta }` / `"pseudo_voigt"`):
     `(1 - eta) * G(Δ2θ) + eta * L(Δ2θ)` with `eta` in `[0, 1]`
3. **Ring amplitudes**: Controlled by `ring_intensity` and scaled by `i_max`:
   - `Uniform` (`"uniform"`, default): every ring has relative amplitude `1.0`.
   - `FromCalibrant { fallback }` (`"calibrant"`): uses relative intensities
     parsed from `.D` inline comments (e.g. `# (1 1 1) 8 100.0`), normalized so
     the strongest listed ring is `1.0`. Rings without a listed intensity use
     `fallback` (default `0.1`, or `1.0` if no ring lists an intensity).
   - `Custom(weights)`: explicit non-negative per-ring weights aligned with
     `calibrant.d_spacings` (sorted descending in `d`, i.e. ascending in `2θ`).
4. **Physical corrections**:
   - `apply_solid_angle` (default `true`): multiplies the ring signal by
     `Geometry::calc_solid_angle()` (matching `Integrator`, which always divides
     by solid angle, and pyFAI's `fake_calibration_image`).
   - `polarization_factor` (default `None`): when `Some(p)` with
     `p` in `[-1, 1]`, multiplies the ring signal by the same linear polarization
     factor `Geometry::calc_polarization` computes.
5. **Background and noise**:
   - Adds `background` (default `0.0`).
   - Optionally applies Poisson shot noise (`poisson_noise = true`) and/or
     additive zero-mean Gaussian readout noise (`readout_noise > 0.0`) using a
     deterministic per-row PRNG seeded by `rng_seed` (identical output regardless
     of Rayon thread count). Note that additive Gaussian readout noise can
     produce negative, non-integer pixel values.
6. **Built-in detector mask**:
   - When a physical `Detector` is attached and `mask_value` is `Some(v)`
     (default `Some(0.0)`), pixels flagged by `Detector::builtin_mask` (such as
     module gaps) are set to `v` and skip noise.

## Rust API

```rust
use std::sync::Arc;
use dioptrin::calibration::Calibrant;
use dioptrin::detector;
use dioptrin::geometry::Geometry;
use dioptrin::integration::Integrator;
use dioptrin::poni::Poni;
use dioptrin::simulation::{
    PeakProfile, RingIntensity, SimulationOptions,
    simulate_diffraction, simulate_diffraction_from_file,
};

let det = Arc::new(detector::named("Pilatus1M").unwrap());
let [dim1, dim2] = det.max_shape.unwrap();
let poni = Poni::load("test_data/v2.poni").unwrap();
let geometry = Geometry::new(dim1, dim2, poni).with_detector(det).unwrap();

// Simulate directly from a .D calibrant file with default options:
let frame = simulate_diffraction_from_file(
    &geometry,
    "test_data/calibrants/CeO2.D",
    &SimulationOptions::default(),
).unwrap();

// Or customize peak profile, calibrant intensities, oversampling, and noise:
let calibrant = Calibrant::from_d_file("test_data/calibrants/CeO2.D").unwrap();
let options = SimulationOptions {
    profile: PeakProfile::PseudoVoigt { eta: 0.3 },
    fwhm_rad: 2.0e-3,
    i_max: 5000.0,
    background: 25.0,
    ring_intensity: RingIntensity::FromCalibrant { fallback: 0.1 },
    oversampling: 2,
    apply_solid_angle: true,
    polarization_factor: Some(0.95),
    mask_value: Some(0.0),
    poisson_noise: true,
    readout_noise: 2.0,
    rng_seed: 42,
};
let noisy_frame = simulate_diffraction(&geometry, &calibrant, &options).unwrap();

// Integrate the simulated frame directly:
let mut integrator = Integrator::new(geometry);
integrator.set_polarization_factor(Some(0.95));
let (radial, intensity) = integrator.integrate1d(&noisy_frame, 1000);
```

## Python API

`dioptrin.simulate_diffraction` accepts an `Integrator`, a PONI dict, or a
`.poni` file path for `geometry`, and a `Calibrant` or `.D` file path for
`calibrant`, returning a 2D `float64` NumPy array of shape `(dim1, dim2)`.

```python
import dioptrin

# From a .poni file + built-in detector name + .D calibrant file:
img = dioptrin.simulate_diffraction(
    "test_data/v2.poni",
    "test_data/calibrants/CeO2.D",
    detector="Pilatus1M",
    ring_intensity="calibrant",
    fwhm_rad=2.0e-3,
    i_max=5000.0,
    background=20.0,
    poisson_noise=True,
    rng_seed=42,
)

# Or from an existing Integrator and Calibrant object:
integrator = dioptrin.Integrator("test_data/v2.poni")
calibrant = dioptrin.Calibrant.from_file("test_data/calibrants/LaB6_SRM660b.D")
img = dioptrin.simulate_diffraction(
    integrator,
    calibrant,
    shape=(1043, 981),
    profile="pseudo_voigt",
    eta=0.4,
    polarization_factor=0.95,
    oversampling=2,
)

# Round-trip 1D integration:
integrator.set_polarization_factor(0.95)
result = integrator.integrate1d(img, n_points=1000)
```

### Important notes

- **`.poni` files and dicts** carry scalar PONI fields only (the `Detector`,
  `Detector_config`, and `SplineFile` fields are not loaded as a physical
  `Detector` by `Poni::load`). Pass `detector="Pilatus1M"` (or an `Integrator`
  with a physical detector attached) to include built-in detector geometry,
  splines, and module-gap masks.
- **Passing an `Integrator`** copies its `Geometry` only; it does not
  automatically inherit `integrator.polarization_factor` unless you pass
  `polarization_factor=...` explicitly.
- **Shape resolution**: If `shape` is omitted, `simulate_diffraction` uses the
  explicit `detector`'s native `max_shape` first, then the `Integrator`'s
  existing shape (or its attached detector's `max_shape`). When passing a
  `.poni` file or dict without a fixed-shape `detector`, `shape=(dim1, dim2)`
  is required.
- **Calibrant intensities**: `Calibrant.from_file` parses optional NIST
  relative intensities from `.D` inline comments into `calibrant.intensities`.
  Programmatic calibrants with intensities can be built with
  `Calibrant.from_d_spacings_and_intensities(name, d_spacings, intensities)`.
