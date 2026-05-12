# Changelog

All notable changes to Dioptrin are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
