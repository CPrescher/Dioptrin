# Changelog

All notable changes to Dioptrin are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Tilt-aware geometry inversion (`src/calibration/inversion.rs`).** Given
  a fitted Debye-Scherrer ellipse on the detector plane plus the
  calibrant's known `2θ_k`, closed-form algebra recovers `(D, PONI,
  rot1, rot2)` exactly for a flat tilted detector — no L-M required.
  Math: `axis_ratio = 1/√(cos²τ - sin²τ·tan²(2θ))` solves for the tilt
  magnitude τ, `semi_minor·px / (a·tan(2θ))` gives D, and the
  PONI-ring-center offset along the ellipse major axis closes the
  system. Aggregating across multiple rings (median per parameter)
  is robust to one bad ellipse fit. Verified on synthetic perpendicular
  and 31.6°-tilt cases to 10⁻⁹ precision (4 unit tests). Used as the
  bootstrap step in `auto_calibrate` for visibly tilted geometries (see
  next entry).
- **Hybrid bootstrap in `auto_calibrate`.** Replaces the rotation-zero
  bootstrap with a regime switch:
    - median ellipse axis_ratio < 1.05 → perpendicular bootstrap
      (PONI = ring-center, rot = 0, D = mean(b·px / tan(2θ))). More
      robust than tilt inversion when the geometry is near-perpendicular,
      because the `(1-C)/(1+tan²(2θ))` term that sets τ amplifies tiny
      ellipse-fit noise into spurious tilts.
    - median axis_ratio ≥ 1.05 → tilt-aware inversion (above).
  This restores the near-perpendicular L-M convergence on real LaB6 (poni
  recovered to ~0.1 %) while giving a physically meaningful starting
  point for tilted cases.
  Known limitation: on strip detectors like the Pilatus300kw in
  `test_data/ceo2_side.tif` (195 × 1475, 31° tilt), each ring contributes
  only a short arc and the 5-point ellipse RANSAC produces unreliable
  ellipses; the inversion then correctly inverts unreliable data into
  unreliable geometry. Closing this case requires constrained
  arc-to-ellipse fitting that enforces the physical axis-ratio prior — a
  separate follow-up slice. The shipped `test_auto_calibrate_ceo2_side.py`
  diagnostic documents the current behavior.
- **Radial-binning ring extraction (`dioptrin.extract_rings_radial`).**
  Given an approximate **ring center**, compute peak radii, build a
  smoothed radial histogram, pick local maxima as ring radii, and fit each
  ring's peaks with a single least-squares ellipse. Avoids the
  5-of-5-same-ring Monte Carlo coincidence that crippled RANSAC on dense
  multi-ring patterns — every peak from each ring goes directly into that
  ring's clean fit. Only valid when rings are approximately circular
  (low detector tilt); for heavily tilted detectors a tilt-aware
  ellipse-center estimator is needed instead.
- **Ring-center estimation (`dioptrin.estimate_ring_center`).** 3-point
  circle RANSAC + radial-band consistency scoring: sample triples of
  peaks, fit the circle they define, score the candidate center by how
  concentrated peaks are in radial bins from it. The function name is
  deliberately **not** "beam_center" or "PONI" — those three geometric
  points coincide only for a perpendicular detector and are distinct
  in general. This module's docstring covers the distinction in detail;
  the short version: for near-perpendicular detectors, ring-center ≈ PONI
  ≈ beam center to within a few pixels, and the estimator is a useful
  starting guess for the bootstrap PONI. For heavily tilted detectors
  the three differ by `~D · tan(tilt)` and a separate ellipse-based
  inversion is required.
  Works robustly on full-coverage multi-ring patterns, partial arcs
  (centroid estimators fail here), and off-detector beams — but only
  when the near-circular assumption holds. 5 Rust unit tests cover
  each case.
- **`AutoMethod::Radial`** is now the default for `auto_calibrate`,
  replacing the old 5-point RANSAC-ellipse approach. End-to-end real-data
  recovery on LaB6_40keV_MarCCD with no clicks and no geometry hint:
  refined distance 0.196656 m (truth 0.196687 m, Δ = 31 µm =
  0.016 %), rot1 4.27 mrad (truth 4.3 mrad, essentially exact),
  RMS residual 0.4 mrad in 8 L-M iterations. The old method is still
  selectable via `method="ransac"`.
  Known limitations:
  - The bootstrap initialises PONI to the estimated ring center. This
    is correct only for near-perpendicular detectors (where ring center
    ≈ PONI ≈ beam center). For tilted detectors PONI is at the foot
    of the perpendicular from sample to detector and is displaced from
    the ring center by `~D · tan(tilt)`; the radial method has no way
    to recover that displacement on its own.
  - The radial-binning assumption (rings are concentric circles around
    one center) breaks for tilted detectors because each ring becomes
    an ellipse with its own geometric center.
  Both issues are addressed by a future tilt-aware "geometry inversion"
  bootstrap that algebraically solves for `(D, PONI, rot1, rot2)` from
  one or more fitted ellipses + the calibrant's known `2θ_k`. The
  shipped `ceo2_side.tif` test fixture (Pilatus300kw strip detector at
  31° tilt) is a known failure case for the current bootstrap.
- **End-to-end auto calibration (`dioptrin.auto_calibrate`).** Wires the
  RANSAC ellipse clustering, ring-identity matching, and L-M refinement
  into a single call: given a list of picked peaks plus a calibrant,
  wavelength, and pixel size, returns a refined PONI with no manual
  seed clicks and no geometry hint. Internal pipeline: RANSAC →
  shared-center filter (drops spurious cross-ring ghost ellipses) →
  identify rings by tan(2θ) ratio matching → bootstrap PONI from
  matched-ellipse centers + s_i / tan(2θ_k) → L-M refine. A Rust unit
  test (`auto_calibrate_recovers_synthetic_geometry`) recovers a known
  synthetic LaB6 geometry to numerical precision.
  Known limitation: on dense calibrants like LaB6 (91 d-spacings),
  when the innermost ring is weakly detected, the ratio-only anchor
  disambiguation can lock onto a self-consistent chain at the wrong
  scaled distance. A follow-up will add an optional distance hint to
  resolve the ambiguity. The slice-2 and slice-3a/3b pipelines are
  unaffected.
- **Image-wide peak extraction (`dioptrin.find_peaks_image`).** A no-seed
  variant of Massif: subtract a Gaussian background, threshold by
  `mean + k·std` of the residual, return every 3×3 local maximum with
  sub-pixel parabolic refinement. Intended as the peak source for the
  auto-calibration pipeline.
- **RANSAC ellipse deduplication.** `ransac_ellipses` now merges fits
  with similar centers and similar semi-major axes (configurable
  thresholds), unioning their inlier sets and refitting. Defaults
  `5 px` center distance and `2 %` size, which removes the manual
  Python-side dedup that the slice-3b test used to do.
- **Ring identity matching (`dioptrin.identify_rings`).** Given a list of
  detected ellipses (sorted ascending by semi-major axis) and a known
  calibrant, identify which calibrant ring each ellipse corresponds to —
  with no input geometry hint (no distance, no beam-center estimate).
  Algorithm: the ratio `r_k / r_0 = tan(2θ_k) / tan(2θ_0)` is independent
  of detector distance, so a hypothesis-test loop over every
  `(anchor_ellipse, anchor_ring)` pair finds the assignment whose
  implied-distance scatter is tightest. Handles missing rings, spurious
  ellipses, smallest-ellipse-not-being-ring-0, and dense calibrants
  (LaB6 has 91 d-spacings) where the ratio chain alone is ambiguous —
  the distance-consistency check breaks the tie. Validated by 5 Rust
  unit tests on synthetic LaB6 (full sequence, missing rings, anchor
  offset, spurious extra ellipse, random sizes) and a real-data smoke
  test on LaB6_40keV_MarCCD.tif that recovers 10/11 ring identities
  matching Dioptas's iterative-fit labels (the one mismatch is a
  Dioptrin-rejected spurious ellipse).
- **RANSAC ellipse clustering (`dioptrin.ransac_ellipses`).** Given a list of
  2-D peak coordinates, fit a family of ellipses with outlier rejection.
  Implementation: direct linear conic fit (F=-1 normalization) with the
  ellipse discriminant check `4AC-B² > 0`, Sampson distance for inlier
  testing, greedy RANSAC that detects one ellipse, removes its inliers, and
  repeats. Optional `max_axis_ratio` filter rejects degenerate elongated
  fits when the user knows the detector is near-perpendicular (e.g.
  setting `1.2` for typical lab/synchrotron geometries). Returns sorted
  ellipses (semi-major ascending → ring order under a powder pattern) plus
  per-input-point assignment (`None` = outlier). Public Rust API in
  `dioptrin::calibration::{ellipse, ransac_ellipses}`. Synthetic Rust
  tests verify single-ellipse recovery to 10⁻⁶, multi-ellipse clustering
  with planted outliers, and pure-noise rejection. Real-data smoke test
  on Dioptas's LaB6 peaks confirms at least the brightest rings are
  cleanly captured at the true beam center.
- **Massif peak picking (`dioptrin.find_peaks_massif`).** A port of pyFAI's
  `Massif.find_peaks` algorithm: separable Gaussian background filter,
  binarisation against the local background, 4-connected component
  labeling, uphill walk to integer-pixel local max, and sub-pixel
  refinement via parabolic fit on the 3×3 neighbourhood. The Gaussian σ
  defaults to pyFAI's adaptive `max(5, max(dim1, dim2) / 50)`. Validated
  against Dioptas's pyFAI-backed Massif on the real LaB6_40keV_MarCCD
  frame: every Dioptas-picked control point has a Dioptrin-picked
  counterpart within sub-pixel distance (100% within 1 px on seed 1,
  96% within 1 px on seed 2). Combined with the refine-parity test this
  validates Dioptrin as a self-contained pyFAI replacement for
  calibration.
- **Geometry calibration (`dioptrin.calibration`).** A Levenberg–Marquardt
  refinement that takes picked control points and a known calibrant and
  refines the detector PONI. Highlights:
  - Hand-rolled L-M with finite-difference Jacobian (no `nalgebra` /
    `argmin` dependency).
  - Optional Huber IRLS loss for robustness against spotty / single-crystal
    contamination on rings.
  - Auto ring-assignment by predicted-2θ-nearest-d-spacing each iteration.
  - Free / fixed parameter mask (`distance`, `poni1`, `poni2`, `rot1`,
    `rot2`, `rot3`, `wavelength`).
  - Off-detector primary beam works correctly (PONI may lie outside the
    detector window).
  - Rotations wrap to `(-π, π]` after each accepted step to prevent drift
    along under-constrained periodic directions.
  - pyFAI/Dioptas pixel-center convention (`(d + 0.5) * pixel`) so picked
    points from those tools feed in directly.
- pyFAI-compatible `.D` calibrant parser; `LaB6_SRM660b.D` and `CeO2.D`
  shipped in `test_data/calibrants/`.
- Python API: `dioptrin.Calibrant`, `dioptrin.CalibrationPoints`,
  `dioptrin.CalibrationResult`, `dioptrin.refine(...)`.
- `LaB6_40keV_MarCCD.tif` + `.poni` test fixture from Dioptas's test corpus.
- `python_examples/test_calibration_lab6.py` — pyFAI-forward-model round-trip
  integration test (recovers a known LaB6 PONI to numerical precision).
- `python_examples/test_calibration_dioptas_parity.py` — cross-validates
  against Dioptas + pyFAI on the same 271 real Massif-picked points across
  10 LaB6 rings; refined PONIs agree at the picking-noise floor (distance
  to 0.4 ppm, PONI to 18 µm, rotations to 93 µrad).

## [0.5.2] - 2026-05-12

### Fixed

- Avoid an 80 MB `normalization` clone on every `integrate1d` / `integrate2d`
  call. `GeometryState.normalization` is now wrapped in `Arc<Vec<f64>>`, so
  the per-call cost is an atomic refcount bump. On a 12-worker batch1d run
  over Eiger 9M frames at 4400 bins, throughput goes from ~163 to ~303
  files/s, and serial→12-worker scaling improves from ~3× to ~4.8×. The
  single-worker numpy-array path more than doubles (53 → 127 files/s).

### Changed

- Batch worker loops (`process_files`, `process_files_streaming`,
  `process_queue_streaming`) reuse a single `Frame` per worker and decode
  TIFFs into its existing `data` buffer via the new
  `Frame::reload_from_tiff` instead of allocating a fresh `Frame` per file.
  Peak RSS during a 12-worker batch1d run drops from ~4.7 GB to ~3.85 GB.
  Throughput is unchanged.

## [0.5.1] - 2026-03-22

### Fixed

- `python_examples/batch_benchmark_comparison.py` no longer hard-requires
  `psutil`; memory tracking is skipped with a notice when the package is
  not installed.

## [0.5.0] - 2026-03-21

### Added

- `MultiIntegrator` for merging results from multiple detectors (similar in
  spirit to pyFAI's `MultiGeometry`). Sums raw signal / normalization /
  variance across detectors on a shared radial bin grid and divides once
  at the end. Available from Python as `dioptrin.MultiIntegrator`.
- `python_examples/batch_benchmark_comparison.py` for side-by-side
  throughput and peak-memory comparisons against pyFAI.

## [0.4.4] - 2026-03-03

### Added

- `dioptrin.validate_license()` Python function for offline license-key
  validation outside of `Integrator` construction.

### Changed

- License format no longer carries a tier field; license keys signed in
  the new format are required.

### Fixed

- License exception classes are always registered on the Python module,
  even when the binary is built without the `licensed` feature, so user
  code can `except dioptrin.LicenseExpiredError:` unconditionally.

## [0.4.3] - 2026-03-03

### Added

- `LicenseNotFoundError` and `LicenseExpiredError` Python exception
  subclasses for finer-grained handling of license errors.

## [0.4.2] - 2026-03-02

### Fixed

- maturin wheel build for `linux/x86_64` (`-i` flag passed correctly).

## [0.4.1] - 2026-03-02

Maintenance release: cargo fmt over license / keygen modules. No
user-visible changes.

## [0.4.0] - 2026-03-02

### Added

- Offline license verification using ed25519 signatures for Python
  wheels. License keys are checked at `Integrator()` construction; the
  public key is embedded in the binary.
- `benchmark` optional dependency group in `pyproject.toml`.

### Changed

- **Breaking:** `normalization` is now a required parameter in the LUT
  integration functions (`lut::integrate1d`, `lut::integrate2d`, etc.).
- LUT integration now follows pyFAI's weighted-average convention:
  `I = Σ(w · signal) / Σ(w · norm)`.
- Minimum Python version bumped to 3.11.

### Fixed

- `batch1d_iter` / `batch2d_iter` now preserve the input order of frames.
- README examples updated to match the current API.

### Security

- numpy upgraded to 2.4.2 (2.4.0 was yanked upstream).
- pillow upgraded to 12.1.1 (CVE-2026-25990).

## [0.3.0] - 2026-02-28

Earliest tagged release tracked in this changelog. See `git log v0.3.0`
for the contents at this point.

[Unreleased]: https://github.com/CPrescher/Dioptrin/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/CPrescher/Dioptrin/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/CPrescher/Dioptrin/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/CPrescher/Dioptrin/compare/v0.4.4...v0.5.0
[0.4.4]: https://github.com/CPrescher/Dioptrin/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/CPrescher/Dioptrin/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/CPrescher/Dioptrin/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/CPrescher/Dioptrin/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/CPrescher/Dioptrin/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/CPrescher/Dioptrin/releases/tag/v0.3.0
