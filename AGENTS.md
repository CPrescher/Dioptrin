# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Rust library code, organized by domain modules like `integration/`, `lut/`, `math/`, `calibration/`, `detector/`, `batch/`, and core files such as `lib.rs`, `geometry.rs`, `frame.rs`, `poni.rs`, and `multi.rs`; `src/python/` holds the PyO3 bindings (`--features python`).
- `benches/` holds Criterion benchmarks (`integrate`, `build`, `integrator`) and their harness configs.
- `tests/` holds Rust integration tests (`detector_reference.rs`).
- `test_data/` stores PONI inputs, test images, calibrants and reference outputs; tests also create `test_data/test_output/`.
- `python_examples/` contains Python examples and validation/comparison scripts (run all with `python test_examples.py`); `scripts/` holds maintenance and release checks.
- `target/` is Cargo build output and should not be edited manually.

## Build, Test, and Development Commands
- `cargo build` compiles the library.
- `cargo test` runs unit tests in `src/` and integration tests in `tests/`, including data-driven checks.
- `cargo bench --bench integrate` runs integration benchmarks.
- `cargo bench --bench build` runs LUT build benchmarks.
- `maturin develop` builds the Python extension into the active environment.

## Coding Style & Naming Conventions
- Rust 2024 edition; follow standard Rust formatting (`cargo fmt`) with default rustfmt settings (no repo-specific config).
- Use idiomatic naming: `snake_case` for functions/modules, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Keep module boundaries aligned with the existing domain structure (e.g., integration logic under `src/integration/`).

## Testing Guidelines
- Tests live alongside code (e.g., `src/integration/tests.rs`) and use `#[test]`.
- `test_data/CeO2_Pilatus1M.tif` is committed; tests download it from GitHub only if it is missing.
- Generated comparison output is written to `test_data/test_output/`; keep this directory out of commits unless explicitly requested.

## Commit & Pull Request Guidelines
- Commit messages follow a simple type prefix: `feat:`, `fix:`, `refactor:`, `doc:`, `bench:` (lowercase), then a short imperative summary.
- PRs should describe the change, list affected modules (e.g., `src/lut/`), and include benchmark or test results when performance is touched.
- If behavior or outputs change, note any updated reference data under `test_data/test_reference/`.

## Configuration & Data Tips
- PONI geometry files live in `test_data/`; prefer adding new fixtures there with clear naming (e.g., `v3.poni`).
- For large artifacts or external datasets, document download sources rather than committing binaries.

- Keep generated benchmark timings, fitted models, run logs and machine-specific
  input manifests out of Git. Keep benchmark code and reusable test fixtures in
  the repository; summarize measured results and conditions in the PR.
