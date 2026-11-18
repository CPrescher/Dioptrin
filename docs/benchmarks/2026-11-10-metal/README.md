# Metal integration prototype — November 10, 2026

The follow-up [complete startup, break-even, and 30-second reuse measurements](SETUP.md)
measure full process-to-first-pattern time. They supersede using the partial
setup timers below as an estimate of complete startup.

This experiment implements actual weighted diffraction integration on the Mac
GPU and compares it with the current Dioptrin CPU implementation. It is an
isolated prototype, with no changes to the production library or its dependencies.

The useful GPU case is repeated integration of larger images with a reusable
lookup table and input already available as FP32. Converting the current FP64
frame to FP32 each time consumes much of the benefit. The existing CPU CSR path
is a strong competitor, particularly for smaller images. Compensated GPU
accumulation substantially improves accuracy with little additional cost in
these measurements, but input and weight rounding still prevent general FP64
equivalence.

## Implementation

- `examples/benchmark_metal_fixture.rs` uses the public Dioptrin geometry/LUT
  builders, applies a fixed mask, and exports the exact CSR pixel assignments.
  It also benchmarks the real `Integrator` API, including its prepared caches.
- `integrate.metal` assigns a threadgroup to each pattern bin. Threads gather
  pixels, apply fractional splitting weights, and reduce the signal and optional
  Poisson variance. Each row has one writer; floating-point atomics are unnecessary.
- `MetalBench.swift` allocates reusable shared Metal buffers, compiles and
  specializes the kernel, submits each image, waits for completion, and reads
  the result back. It converts raw GPU sums to Double and applies the cached
  FP64 normalization on the CPU. The radial axis is unchanged by construction.
- `run.py` builds both programs, runs the experiment, verifies accuracy, records
  provenance, and removes the temporary binary fixtures. It needs only Python's
  standard library.

Both GPU variants use FP32 inputs and weights. The ordinary variant uses fused
multiply-add and SIMD reductions. The compensated variant retains a high/low
FP32 pair for products, accumulators, and reductions, and returns both components
to the CPU. Explicit error-free transforms run with Metal's safe math mode;
the code does not enable fast math. This improves summation accuracy without
restoring information already lost when inputs or weights were rounded to FP32.

For signal `s`, variance `v = max(s, 0)`, splitting coefficient `w`, and
normalization `n = polarization * solid_angle`, the calculation is:

```text
intensity[b] = sum(w * s) / sum(w * n)
std[b]       = sqrt(sum(w * w * v)) / sum(w * n)
```

The fixed mask is baked into the exported row entries. The denominator is
precomputed in FP64, as in the prepared CPU paths. Empty or fully masked bins
return zero. Changing geometry, mask, corrections, splitting method, or binning
requires preparing/exporting a new plan in this prototype.

## Measurement method

Hardware: Apple M4 Pro, 14 CPU cores (10 performance + 4 efficiency), 20 GPU
cores, 48 GB unified memory. OS and compiler versions, commit, source hashes,
and input file hashes are recorded in `results.json`.

Each image has 3,000 output bins. The CPU comparison includes the default fast
CSC path, fast CSR, and strict CSR, each at 1, 4, 10, and 14 Rayon threads. All
paths have fixed masks and the same solid-angle/polarization correction, with
polarization factor 0.965. Accuracy is checked against the existing public FP64
CSR integration, and every prepared CPU configuration must agree with that
reference to relative `1e-10` (absolute floor `1e-10`).

The GPU sweep covers 64, 128, and 256 threads per bin; ordinary and compensated
accumulation; and intensity alone or intensity plus Poisson std. Three input
modes distinguish data preparation from compute:

| Mode | Work included in wall time |
| --- | --- |
| `resident_f32` | Existing shared FP32 input; command encoding/submission, waiting, readback, and FP64 normalization |
| `f32_copy` | Copy a fresh FP32 image into the shared input buffer, plus the above |
| `f64_convert` | Accelerate FP64-to-FP32 conversion directly into the shared input buffer, plus the above |

The last mode starts from the same FP64 image representation as the CPU API.
The resident mode is a lower-bound scenario for a future shared-buffer decoder
or GPU pipeline; the prototype does not implement that decoder. Shared memory
avoids a discrete PCIe upload, but input copying and conversion still cost time.
The CPU API allocates output vectors; the Swift harness reuses its output arrays.
The radial vector is cached in the prototype; the CPU API also returns it.

Before timing input-copy/conversion variants, the input buffer is deliberately
zeroed, then the uploaded result must match the resident-input result. This
checks that repeated images do not conceal a missing or stale input upload.

Each configuration receives at least 100 ms of warm-up. The recorded run uses
51 CPU timing samples and 101 GPU timing samples per variant. Reported numbers
are medians; full samples and p95 values are retained. Device execution time is
reported separately using `gpuEndTime - gpuStartTime`, after completion.
Input preparation and CPU readback/normalization also have separate timings.
Timing instrumentation itself is inside the measured wall interval.

These are synchronous, repeated-frame integration measurements on an active
desktop. They exclude TIFF/HDF5 reading, decompression, file orientation changes,
geometry/LUT building, buffer allocation, and shader compilation. Both CPU and
GPU repeatedly access the same image, so this is a cache-friendly workload.
There is no asynchronous batching or overlap with decoding. Clock variation and
display/GPU activity affect results: small speed differences should not be
treated as established application-wide improvements. Initial runs with shorter
warm-up differed materially, which motivated the time-based warm-up used here.

## Results

| Image / method | Poisson std | Default CPU (ms) | Best CPU (ms; configuration) | Metal, FP64 input (ms) | CPU / Metal |
| --- | --- | ---: | --- | ---: | ---: |
| Pilatus counting | No | 0.589 | 0.090; fast_csr, 10 threads | 0.245 | 0.37x |
| Pilatus counting | Yes | 0.887 | 0.091; fast_csr, 10 threads | 0.241 | 0.38x |
| Pilatus pixel splitting | No | 3.459 | 0.407; fast_csr, 10 threads | 0.435 | 0.94x |
| Pilatus pixel splitting | Yes | 4.712 | 0.432; fast_csr, 10 threads | 0.434 | 1.00x |
| Eiger counting | No | 5.820 | 1.247; strict_csr, 14 threads | 1.715 | 0.73x |
| Eiger counting | Yes | 8.693 | 1.376; strict_csr, 14 threads | 1.712 | 0.80x |
| Eiger pixel splitting | No | 17.203 | 2.630; fast_csr, 14 threads | 2.387 | 1.10x |
| Eiger pixel splitting | Yes | 22.653 | 2.934; fast_csr, 14 threads | 2.400 | 1.22x |

All GPU columns below use compensated accumulation and include result readback
and FP64 normalization. “Best CPU” is the lowest median among the tested CPU
layouts and thread counts; the JSON identifies the selected configuration.
“Default CPU” uses the default CSC layout with its best tested thread count.

| Image / method, with std | FP64 input (ms) | Copy FP32 input (ms) | Resident FP32 input (ms) |
| --- | ---: | ---: | ---: |
| Pilatus counting | 0.241 | 0.198 | 0.150 |
| Pilatus pixel splitting | 0.434 | 0.393 | 0.356 |
| Eiger counting | 1.712 | 1.132 | 0.506 |
| Eiger pixel splitting | 2.400 | 1.823 | 1.156 |

Resident-buffer gains do not include decoding directly into those buffers.
The `f32_copy` measurement includes a fresh image copy, making it the more useful
comparison for an existing FP32 image held elsewhere in host memory.

## Numerical accuracy

The accuracy check covers every output bin, all threadgroup sizes, all input
modes, both accumulation variants, and both intensity/uncertainty modes. Every
real-image variant must satisfy
`abs_error <= 1e-5 + 1e-5 * abs(reference)` for intensity and std. Additionally,
every compensated real-image variant must have maximum intensity error smaller
than 0.01 of the reference Poisson std in bins with positive std. These are
explicit experimental acceptance criteria, not a universal scientific guarantee.

| Image / method | Ordinary FP32: max relative intensity error | Compensated: max relative intensity error | Compensated: max intensity error / std | Compensated: max relative std error |
| --- | ---: | ---: | ---: | ---: |
| Pilatus counting | 0 | 0 | 0 | 0 |
| Pilatus pixel splitting | 1.49e-07 | 3.11e-08 | 2.64e-05 | 3.26e-08 |
| Eiger counting | 1.56e-06 | 4.66e-10 | 0.00031 | 2.33e-10 |
| Eiger pixel splitting | 1.36e-06 | 5.37e-08 | 0.00458 | 5.34e-08 |

The Eiger TIFF has 63,770 values equal to signed-int32 maximum (2,147,483,647)
outside the supplied mask. These are retained identically for both engines.
This makes it a useful high-dynamic-range arithmetic stress test; the benchmark
does not establish whether those pixels are scientifically valid. Reported std
ratios use the same Poisson model as Dioptrin; no measured detector noise model
or model uncertainty is included.

The independent eight-bin boundary fixture checks empty bins, fully masked NaN
pixels, fractional weights, negative signals with zero Poisson variance, large
counts, cancellation, FP32 input rounding, and zero signal. Expected intensities
and one fractional-weight variance are hand-checked by the Rust fixture.

- Ordinary FP32 loses the unit contribution in `1e8 + 1 - 1e8`; compensated
  accumulation recovers the FP64 intensity of `1/3`.
- Both variants lose the unit difference between `16,777,217` and `16,777,216`
  when the former is converted to FP32. The CPU intensity is `0.5`, while the
  GPU intensity is zero. This is an expected, explicitly asserted limitation.
- Empty, masked/NaN, fractional, negative, and zero rows must pass. The report
  retains the expected boundary failures rather than labeling all tests as
  universally numerically equivalent.

An FP64 CPU fallback remains necessary for arbitrary FP64 inputs, strong signed
cancellation, values outside the usable FP32 range, or stricter accuracy
requirements. Neither GPU variant is a bitwise replacement for Dioptrin's FP64
arithmetic. The prototype supports Poisson uncertainty only, and does not cover
dynamic masks, azimuthal std, 2D cake integration, or all integration methods.

## Reproduce

From the repository root on an Apple silicon Mac with Xcode Metal support,
Rust, and Swift (the Swift harness uses macOS 15+ math-mode APIs):

```sh
python3 docs/benchmarks/2026-11-10-metal/run.py --samples 101 --cpu-samples 51
```

For the tracked Pilatus fixture and boundary cases alone:

```sh
python3 docs/benchmarks/2026-11-10-metal/run.py --small-only --output /tmp/metal-small.json
```

The full run also needs `test_data/eiger_0000.tiff` (39 MB) and
`test_data/eiger_0000.mask`. The mask is in the repository; the TIFF is not
(it is git-ignored), so the two Eiger cases cannot be rerun from a fresh
clone. Without the TIFF, the full `run.py` builds the example and then stops
with `FileNotFoundError` while hashing the inputs, before any measurement and
without writing its output; use `--small-only` (with `--output`, since the
default output is `results.json`). `run_setup.py` (without `--small-only`)
and `run_idle_cpu.py` in [SETUP.md](SETUP.md) also need the TIFF: they fail
at the first Eiger case, `run_setup.py` after rewriting `setup-results.json`
with the Pilatus cases. The derived
PONI under `docs/benchmarks/2026-11-05-bslz4/calibration.poni` supplies explicit
75-micrometer pixel sizes. TIFF signal and mask use the same existing Dioptrin
vertical-flip loaders. The real fixtures additionally mask negative/nonfinite
input values. Large generated binary exports live in a temporary directory and
are automatically removed; source data and reference data are unchanged.

Relevant Apple documentation:
[compute commands and shared buffers](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
and [GPU command-buffer timing](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime).

## Validation completed

- Full four-image/method benchmark: all 144 real-image GPU variants passed
  intensity/std checks; all 72 compensated variants also satisfied the
  intensity-error limit of 0.01 std. Input copy/conversion validation passed.
- The eight-bin boundary fixture passed its semantic checks and demonstrated
  the expected FP32 conversion failure. Compensated cancellation recovery passed.
- Metal API validation and GPU shader validation were enabled for all 36
  boundary-fixture configurations, with no validation errors.
- `cargo test --lib`: **202 passed, 0 failed**.
- Rust formatting, Python syntax, source-hash consistency, and
  `git diff --check` passed. Existing unrelated workspace changes were preserved.

No production reference data, integration defaults, or release artifacts were
changed. Large temporary benchmark fixtures were removed after measurement.
