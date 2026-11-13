# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Rust library code, organized by domain modules like `integration/`, `lut/`, `math/`, and core files such as `lib.rs`, `geometry.rs`, `frame.rs`, and `poni.rs`.
- `benches/` holds Criterion benchmarks (`integrate`, `build`) and their harness configs.
- `test_data/` stores PONI inputs and reference outputs; tests also create `test_data/test_output/`.
- `notebooks/` contains Python notebooks/scripts for validation checks.
- `target/` is Cargo build output and should not be edited manually.

## Build, Test, and Development Commands
- `cargo build` compiles the library.
- `cargo test` runs unit/integration tests in `src/`, including data-driven checks.
- `cargo bench --bench integrate` runs integration benchmarks.
- `cargo bench --bench build` runs LUT build benchmarks.

## Coding Style & Naming Conventions
- Rust 2021 edition; follow standard Rust formatting (`cargo fmt`) with default rustfmt settings (no repo-specific config).
- Use idiomatic naming: `snake_case` for functions/modules, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Keep module boundaries aligned with the existing domain structure (e.g., integration logic under `src/integration/`).

## Testing Guidelines
- Tests live alongside code (e.g., `src/integration/tests.rs`) and use `#[test]`.
- Some tests download `test_data/CeO2_Pilatus1M.tif` from GitHub on first run; ensure network access is available.
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
