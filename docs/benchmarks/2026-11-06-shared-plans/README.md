# Shared dynamic-plan validation and setup measurements

`Integrator.prepare_dynamic1d(shape, bins)` prepares one immutable geometry/CSR/CSC/correction snapshot. `plan.worker(num_threads=1)` creates a lightweight, sendable handle. Every frame passes its own complete mask; masks and normalization sums are independent between calls. Source-integrator edits do not mutate an existing plan.

Five fresh-process trials per case on the original 3262×3108 Eiger geometry, 2000 two-theta bins, pixel splitting, polarization 0.965, solid-angle correction, float64 arithmetic:

| Workers | Separate plans, seconds | Shared plan, seconds | Speedup | Separate/shared peak RSS, GiB |
| --- | --- | --- | --- | --- |
| 4 | 1.495 | 0.683 | 2.19x | 7.12 / 2.42 |
| 8 | 2.494 | 0.621 | 4.02x | 12.31 / 2.66 |

Separate plans were built concurrently, with the same N-thread global Rayon build budget as the shared case. Each final worker has one compute thread. Setup timing includes unprepared integrator creation, geometry/corrections/CSR/CSC preparation and worker handles. Process/import startup, reading, masking, numerical integration and validation are excluded. Each case has its own process and memory high-water mark. Background activity was recorded; eight-plan setup ranged 1.915–3.474 seconds, so these are observed medians rather than isolated hardware limits.

All handles were validated after timing against an independent fixed-mask CSR integration of the first original Eiger frame and static+threshold (>230) mask. Maximum relative L2 error was 3.71e-16. The companion Qt Dioptas benchmarks found approximately unchanged steady-state throughput when only plan sharing differed.

Raw measurements and native binary/calibration/mask/driver hashes: [setup.json](setup.json). Aggregates: [setup-summary.json](setup-summary.json). The reproduction driver is `scripts/benchmark_lut_setup.py` in the companion Qt Dioptas checkout; run its documented command there with the external Eiger data/calibration/mask recorded in the JSON. Large detector inputs and generated pattern arrays are intentionally not included here.

Regression coverage: `src/integration/dynamic.rs` and `python_examples/test_dynamic_plan.py` exercise shared storage, concurrent changing masks, strict/fast arithmetic, statistics, methods, source-snapshot isolation and shape/contiguity validation. Mode-switch tests verify that an ordinary Integrator preserves its original CSR geometry/weights when switching fixed/dynamic, while refreshing mode-specific execution state.

The Qt application currently releases its dynamic shared plan when a fixed-mode batch/map run begins; returning to dynamic mode builds one new shared plan. This is an application cache-lifetime policy, not a requirement of the immutable-plan API. Mask and threshold edits while staying dynamic can retain the same geometry plan.

These measurements used the local unreleased shared-plan build (version metadata 0.8.0); they do not describe the published 0.8.0 wheel.
