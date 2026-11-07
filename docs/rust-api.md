# Rust integration

The examples below come from the repository README. Generate the full Rust API
reference locally with `cargo doc --no-deps --open`.

```{include} ../README.md
:start-after: "<!-- manual-rust-start -->"
:end-before: "<!-- manual-rust-end -->"
```

## Fixed-mask integration from decoded blocks

`Integrator::prepare_fixed1d((height, width), bins)` creates an immutable snapshot
of the current geometry, corrections and fixed mask. It requires fixed mask mode,
CSC and fast arithmetic (float64 weights and sums, with fused multiply-add).
Counting, pixel splitting and positive supersampling are supported. Dynamic masks,
histogram integration, CSR selection and strict arithmetic are rejected.

Share one `Fixed1dPlan` across frame workers; each worker borrows it and owns four
reusable float64 partial histograms. After `begin_frame()`, push nonempty contiguous
blocks with `push_u16_block`, `push_u32_block` or `push_f64_block`. Offsets refer to
the original row-major input. `InputOrder::FlipVertical` maps stored rows into
calibrated detector rows without copying a full image. The default detector order
performs no orientation change.

Call `finish_frame()` only after the entire reader operation succeeds; call
`abort_frame()` on every reader error, including an error after the last callback.
An incomplete or invalid sequence cannot produce a result. `finish_frame()` applies
cached normalization once and optionally returns Poisson errors. Workers process
one frame serially; concurrency across frames belongs to the caller.

Later changes to the source integrator do not alter a snapshot. Recreate the plan
when geometry, corrections or the fixed mask change. Small float64 rounding
differences from full-frame reductions are possible. This is currently a Rust
API; it does not change Python bindings or Dioptas behavior.

The accompanying dioptrin-io `benchmarks/fused-eiger` source illustrates connecting
raw HDF5 block callbacks to this accumulator. The reader retains original integer
counts; Dioptrin owns masking and normalization.

To integrate one frame on several cores, give each worker a nonempty, contiguous
source pixel range using `begin_range(range)`. Feed only that range's ordered
blocks. After the reader succeeds, `plan.finish_ranges(&mut workers)` verifies
complete, ordered coverage, the same plan, input order and error mode. It merges
raw float64 histograms and applies normalization once. Gaps, overlaps, incomplete
ranges and reused results are rejected. A partial worker cannot finish a whole
frame through `finish_frame()`.

`dioptrin-io::hdf5::raw::RawReader::fold_frame_blocks` supplies these ranges while
reading the compressed frame once and decoding blocks concurrently. A worker
still owns its histograms privately; the large geometry lookup remains shared.

Each fused call targets one frame. The same fixed plan can be reused for successive
frames in an HDF5 stack or for compatible frames from different files; the API is
not restricted to a one-file session. This addition does not change application
batch/map routing. Dynamic-mask processing keeps its existing complete-frame path.
