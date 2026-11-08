# Native multi-detector 2D validation

This records the initial implementation at `7e6b73b`. The subsequent
[full-splitting investigation](multi2d-splitting.md) adds an opt-in pyFAI
histogram convention, a singular beam-pixel fallback and azint-style sampling
validation. Its report contains the current test totals and compatibility results.

Date: 2026-11-08. Implementation branch: `codex/multi-detector-2d`, isolated
worktree `/Users/clemens/Programming/Dioptrin-multi2d`, based on released 0.10.0
commit `d667a7f`. No release/version bump or desktop checkout changes.

## API and adapter contract

The [multi-detector guide](../MULTI_DETECTOR.md) specifies the complete API,
units, ranges, correction formula and Rust/Python examples. Entry points:

```text
Rust:
MultiIntegrator::try_new(Vec<Integrator>) -> Result<Self, String>
MultiIntegrator::integrate2d(&[&Frame], n_radial, n_azimuthal, &Multi2dOptions)
    -> Result<Multi2dResult, String>
MultiIntegrator::integrate2d_from_slices(&[(&[f64], rows, cols)], n_radial,
    n_azimuthal, &Multi2dOptions) -> Result<Multi2dResult, String>
Integrator::integrate2d_raw_on_grid_from_slice(data, rows, cols, &grid, errors)
    -> Result<IntermediateResult2d, String>
IntegrationGrid2d::new(n_radial, n_azimuthal, radial_edges, azimuthal_edges)
    -> Result<Self, String>

Python:
MultiIntegrator.integrate2d(images, n_radial, n_azimuthal, errors=False,
    azimuthal_range=None, *, normalization="relative", normalization_factors=None)
    -> IntegrationResult
```

`Multi2dOptions` contains `azimuthal_range`, `errors`, `normalization`
(`Relative` or `Absolute`), and optional per-detector `normalization_factors`.
The common radial edges/unit come from the detector integrators. Rust output
includes coordinates, intensity, optional errors and the existing raw
`IntermediateResult2d`. Python reuses `IntegrationResult`, populating both
coordinate arrays, errors and dimensions. Output layout is azimuthal × radial.

For Qt `MultiGeometry(correctSolidAngle=True)` scaling, select **Absolute**.
Each detector's raw normalization is multiplied by its own
`pixel1*pixel2/distance²` and monitor before merging. Signal and Poisson variance
are not multiplied by these factors. Relative solid angle and polarization
remain per pixel. This fixes the missing detector-constant issue for the new
2D path without changing legacy 1D results. `correctSolidAngle=False` is not
equivalent to the new Relative mode: Relative still applies relative solid angle.

A GUI adapter can consume the Rust crate directly through this local checkout
or its commit. Construct masks/calibrations before sharing Integrator Arcs,
retain them for subsequent frames to reuse prepared tables, and reconstruct
when detector settings change. Keep each detector's shape stable during
concurrent use of a shared Integrator. Frontend code must convert azimuthal
centers from radians to degrees if the display uses degrees.

## Numerical checks

Ten new Rust tests cover:

- One detector versus ordinary 2D on matched edges (counting and supersampling,
  including errors); corrected uniform polygon integration versus ordinary 2D.
- Identical independent detectors: equal intensity, doubled raw sums and std
  reduced by sqrt(2), under strict/fast arithmetic and fixed/dynamic masks.
- An independent counting oracle with unequal shapes, pixel sizes, distances,
  rotations, polarization, masks and monitors, in both normalization modes.
- Overlap weighted by fractional pixel coverage rather than by detector count.
- Disjoint coverage, complementary partial masks under both mask modes, fully
  masked detectors, empty bins and excluded radial and
  azimuthal sectors.
- Radial/azimuthal clipping of split pixels and variance from the combined
  fractional weight squared, not independent subpixel noise.
- All four radial units, periodic seams and starts at 0 and 2π, fractional
  contribution conservation, validation of invalid shapes/grids/masks/scales,
  bad q wavelength, nonfinite pixels and zero/overflowing supersample factors.

Four Python tests cover mixed TIFF/array inputs with unequal shapes and TIFF
orientation, result coordinates/layout, optional errors, monitor/absolute
scaling, invalid inputs and twelve matched pyFAI comparisons. No external image
download is needed for the new tests.

Full commands:

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets --features python -- -D warnings
maturin build --release -i /path/to/python
# Test the wheel in an isolated environment containing numpy, fabio and pyFAI:
python python_examples/test_multi2d.py --report docs/validation/multi2d-pyfai.json
```

The development verification extracted the wheel under `target/python-wheel`
and used `PYTHONPATH=target/python-wheel` with the existing Python 3.13 test
environment, leaving the installed package unchanged.

Validation completed: **189 Rust tests + 16 doctests passed**, all four Python
tests passed (including twelve pyFAI comparisons), strict Clippy and rustfmt
checks passed, and a release-mode CPython 3.13 ARM64 wheel built successfully.

## pyFAI comparison and limitations

Reference version: **pyFAI 2025.12.1**. Two different shapes, pixel sizes,
distances and rotations; masks, polarization 0.6, monitors [2, 3];
`correctSolidAngle=True`; shared 12 × 17 grids. pyFAI expands its radial upper
edge by float32 epsilon, so actual radial edges are recovered from the returned
centers and supplied to Dioptrin. Azimuthal edges are [-π, π] (pyFAI accepts
these as [-180, 180] degrees). Arrays use identical row orientation.

Raw metrics are in [multi2d-pyfai.json](multi2d-pyfai.json). Maximum relative
differences over bins populated by pyFAI:

| Scenario | Matched method | Intensity | std | Coverage difference |
| --- | --- | ---: | ---: | --- |
| Off axis, 2θ rad/deg | counting / no split histogram | 7.77e-9 | 7.77e-9 | none |
| Off axis, q Å⁻¹/nm⁻¹ | counting / no split histogram | 2.60e-8 | 2.60e-8 | none |
| Beam and seam, 2θ rad | counting / no split histogram | 5.12e-8 | 5.12e-8 | none |
| Radial crop, 2θ rad | counting / no split histogram | 1.56e-8 | 1.56e-8 | none |
| Off axis, 2θ rad/deg | polygon / full histogram | 0.106% | 0.725% | none |
| Off axis, q Å⁻¹/nm⁻¹ | polygon / full histogram | 0.060% | 2.23% | none |
| Beam and seam, 2θ rad | polygon / full histogram | 2.24% | 8.02% | one additional nonzero Dioptrin bin |
| Radial crop, 2θ rad | polygon / full histogram | 0.191% | 2.30% | none |

The tests assert near-rounding agreement for no splitting, and separately bound
and record polygon method differences and coverage differences. Polygon
comparison is **not** a claim of Qt-identical cakes. In particular, the beam/seam
case compares 72 pyFAI-populated bins; the additional Dioptrin bin is recorded
separately instead of hidden in relative-error statistics.

Source inspection establishes concrete convention differences:

- [pyFAI MultiGeometry 2025.12.1](https://github.com/silx-kit/pyFAI/blob/v2025.12.1/src/pyFAI/multi_geometry.py)
  scales each detector normalization by `pixel1*pixel2/dist**2` when solid-angle
  correction is enabled. This is the Absolute contract implemented here.
- [pyFAI fullSplit2D_engine](https://github.com/silx-kit/pyFAI/blob/v2025.12.1/src/pyFAI/ext/splitPixel.pyx)
  recenters corners, clips vertices to the requested limits and renormalizes
  their accumulated area. Its
  [line-area kernel](https://github.com/silx-kit/pyFAI/blob/v2025.12.1/src/pyFAI/ext/regrid_common.pxi)
  distributes integrated line areas through buffer rows, using float temporaries.
- Dioptrin reuses its polygon clipping helpers to intersect pixel quads with
  rectangular bins in float64, divides by the original pixel polygon area, and
  handles periodic copies across the azimuthal seam. Range clipping retains
  only the in-range fraction. These algorithmic differences explain why
  matching grids and normalization alone do not guarantee full-split identity.
  The measurements above do not attribute every bin's difference to one cause.

The ordinary Dioptrin polygon builder uses row-normalized areas rather than
per-pixel fractions. For nonuniform data the new single-detector polygon path
therefore need not equal legacy ordinary `integrate2d`. Counting and
supersampling match on fully covered matched grids; ordinary auto-range
supersampling can clamp samples beyond its estimated radial limits, while the
new explicit-grid path excludes them. These legacy APIs were deliberately left
unchanged. Supersampling has no direct equivalent in pyFAI MultiGeometry; it is
validated by conservation, variance identities and ordinary Dioptrin comparisons.

Only independent Poisson uncertainties are supported by this API. Explicit
variance arrays, dark/flat corrections, azimuthal error models, output-bin
covariances and a 2D batch convenience method are not added. No claim is made
about real detector-file decoding parity or end-to-end desktop UI behavior.

## Latency characterization

Synthetic unequal shapes 256 × 320 and 192 × 240; 256 radial × 180 azimuthal
bins; errors enabled, absolute normalization, polarization and monitors; eight
Rayon threads, macOS ARM64, Python 3.13.5. Twenty cached calls per method:

| Method | First call, including table preparation | Cached median |
| --- | ---: | ---: |
| Counting | 7.77 ms | 0.919 ms |
| Polygon splitting | 12.11 ms | 0.962 ms |
| Supersampled, n_ss=3 | 10.21 ms | 0.820 ms |

These measurements include Python input copies, raw merging and result-object
construction (output list getters are excluded), but no image decoding or UI work. This is a small synthetic
characterization, not a before/after speedup or a real-detector throughput
claim. Repeated cached results were identical. Reproduce with
`RAYON_NUM_THREADS=8 python python_examples/bench_multi2d.py --report result.json`.
The recorded environment and timings are in [multi2d-timing.json](multi2d-timing.json).
