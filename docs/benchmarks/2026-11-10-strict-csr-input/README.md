# Strict CSR integration from decoder pixels

`FixedCsr1dInputPlan` shares an existing `Arc<FixedCsr1dPlan>` and maps decoder
addresses once for `InputOrder::Detector` or `InputOrder::FlipVertical`.
It accepts uint16/uint32 frames directly and runs on the caller's thread,
allowing a frame pipeline to own scheduling and decoder-buffer reuse.

The original coefficients, row-entry order, mask compaction, cached
normalization, floating-point multiply/add order, variance weights and final
division/square-root operations are retained. The typed path reuses the existing
strict row kernel. It does not use FMA, reassociate sums or regroup coefficients.
The mapped columns are shared across frames; no full-frame float64 conversion,
vertical-flip copy or nested Rayon dispatch is required.

This API targets uncorrected integer exposures with a fixed mask. Corrected
pixels, externally supplied variances and dynamic masks retain their existing
native execution types. The float64 resident CSR API is unchanged.

A macOS sample of the preceding eight-worker desktop Eiger pipeline found CSR
reduction as the dominant active stack (6,038 collapsed leaf samples), followed
by float64 expansion (2,066), then Bitshuffle decoding (1,502 across its main
stacks). Sampling is for hotspot attribution, not throughput measurement.
The desktop uses all 625 original 3262 × 3108 Eiger frames at 2,000 bins for
separate before/after timing and full float64 output comparison.

All 215 Rust library tests pass in release mode, including bitwise comparisons
of both integer types and input orders against the original float64 CSR path,
ragged/empty rows, unsorted/duplicate entries, fractional and subnormal weights,
partial/all masks, large counts, errors on/off, invalid-input recovery and
concurrent use. Strict library/test Clippy also passes. One pre-existing test
needed a missing borrow on an `Arc<Vec<f64>>` argument to compile the pinned
dynamic-plan test suite; production dynamic-plan behavior is unchanged.
