# First-frame and cached-frame costs, including real Eiger data

2026-11-08. Apple M4 Pro (14 logical CPUs), **8 Rayon threads**, macOS ARM64,
release build, CPython 3.13.5. Same **2000 radial × 360 azimuthal** grid for all
methods, q range 0–20 Å⁻¹, polarization 0.6, fixed masks, absolute solid-angle
normalization, Poisson errors enabled. Masks and corrections are identical
between methods. Native Rust and Python results agreed to rtol 2e-12,
atol 1e-8 for every method/case.

The comparison uses the native shared-grid 2D MultiIntegrator for both cases:

- An unequal **two-detector pair**, 1043 × 981 and 768 × 640, totaling
  **1,514,703 pixels**, with synthetic nonuniform images, distinct pixel sizes,
  distances and rotations, per-detector masks and monitor denominators [1, 2].
- **One actual Eiger2CdTe_9M detector**, **3262 × 3108 = 10,138,296 pixels**,
  through the same multi-detector API. Three distinct frames, indices 0, 1, 2,
  from the local `eiger_0000.h5`, dataset
  `/entry_0000/instrument/eiger/data`; matching calibration and supplied mask
  with **887,588 excluded pixels**. The original image has orientation 2 and
  is vertically flipped for native geometry; the supplied TIFF mask is already
  bottom-up and is not flipped again. All unmasked invalid detector sentinel
  values were checked before timing. Explicit 75 µm pixels were obtained from
  the calibrated Eiger detector model.

This is a single Eiger benchmark, not two duplicated Eiger images pretending
to be an independent detector pair. The image-array hashes, geometry and mask
hash are in [the machine-readable receipt](multi2d-startup.json). Large detector
data are not committed; they are existing local test inputs.

## Native Rust latency

First frame includes fresh Integrator/mask construction, geometry/correction
calculation, table preparation, normalization preparation, integration and the
returned result buffers. Thirty cached calls cycle the three preloaded frames.

| Method | Detector pair: first | Pair: cached median / p95 | Eiger: first | Eiger: cached median / p95 |
| --- | ---: | ---: | ---: | ---: |
| Fractional polygons + beam fallback | 0.174 s | 8.66 / 8.98 ms | 0.729 s | 24.01 / 24.65 ms |
| pyFAI histogram compatibility | 0.151 s | 9.09 / 10.21 ms | 0.732 s | 23.97 / 24.31 ms |
| Subpixels, 21 × 21 | 1.943 s | 8.99 / 9.43 ms | 12.425 s | 24.95 / 26.57 ms |
| Subpixels, 41 × 41 | 6.171 s | 9.12 / 9.46 ms | 42.058 s | 24.50 / 24.94 ms |

On this Eiger/grid, 21 × 21 sampling costs about **17 times** as much startup
as fractional polygons, and 41 × 41 about **58 times**. Their subsequent
integration medians differ by less than 1 ms: sample contributions are combined
into one weight per original pixel/output bin when the table is built. Runtime
uses that compact sparse table, not the full subpixel grid.

These are one first-frame observation per fresh Integrator/method and 30 warm
observations. First-frame variability was not characterized; do not interpret
small differences between cold timings as a method ranking. This chat started no competing
benchmark/compiler workload during Eiger timing; the machine was not otherwise
reserved or isolated. The full per-call
sample vectors are retained in the receipt.

## Python API latency

These timings include copying NumPy inputs into owned frame buffers and
returning an IntegrationResult object. They **exclude** accessing the
`intensity`/`errors` getters, which materialize Python lists, and exclude any
subsequent NumPy conversion.

| Method | Pair: first | Pair: cached median / p95 | Eiger: first | Eiger: cached median / p95 |
| --- | ---: | ---: | ---: | ---: |
| Fractional polygons + beam fallback | 0.162 s | 10.32 / 10.89 ms | 0.828 s | 28.58 / 30.63 ms |
| pyFAI histogram compatibility | 0.153 s | 10.63 / 12.94 ms | 0.683 s | 26.59 / 27.29 ms |
| Subpixels, 21 × 21 | 1.784 s | 10.64 / 10.90 ms | 12.454 s | 26.84 / 27.62 ms |
| Subpixels, 41 × 41 | 6.070 s | 10.76 / 11.04 ms | 50.878 s | 27.23 / 28.86 ms |

Native/Python cold observations were separate sequential runs; their difference
is not a clean estimate of binding overhead. The warm figures are more useful
for that comparison. React/Tauri will consume the Rust path; frontend transfer,
serialization and plotting were not benchmarked.

## How closely do the methods agree?

The numerical reference here is **41 × 41 subpixel sampling on the same grid**,
not pyFAI and not an exact physical ground truth. All methods retain the same
raw-sum normalization/correction estimator. Compare frame 0 only.

Normalized RMS is `RMS(method - reference) / RMS(reference)` over bins covered
by both methods. The relative p95 uses positive-reference bins and denominator
`max(1, abs(reference))`. Coverage uses summed pixel weight > 1e-6.

| Case / method | Intensity normalized RMS | Intensity relative p95 | std normalized RMS | Extra / missing covered bins versus 41 × 41 |
| --- | ---: | ---: | ---: | ---: |
| Pair / fractional polygons | 0.2557% | 0.0384% | 0.8389% | 8 / 0 |
| Pair / compatibility | 0.6174% | 0.0790% | 3.5548% | 2 / 86 |
| Pair / 21 × 21 | 0.4553% | 0.0952% | 1.5258% | 1 / 9 |
| **Eiger / fractional polygons** | **0.01317%** | **0.02127%** | **0.3610%** | 38 / 0 |
| Eiger / compatibility | 0.12765% | 0.35996% | 2.0905% | 4 / 142 |
| Eiger / 21 × 21 | 0.02450% | 0.05700% | 0.3787% | 0 / 38 |

The Eiger reference has 349,503 covered bins out of 720,000, of which 348,794
have positive intensity. The pair reference has 272,512 covered/positive bins.
Small summary errors do not imply a universal binwise bound: individual bins
show much larger discrepancies. For example, maximum Eiger relative intensity
differences on common positive bins are 411% for polygons and 625% for 21 × 21
sampling under the stated metric. These maxima, p99 values, coverage differences
and all std comparisons are in the receipt. Bins present in only one method are
reported separately rather than hidden in RMS statistics.

For this real Eiger frame, fractional polygons are both much faster to prepare
and closer to the 41 × 41 reference than 21 × 21 sampling by these summary
metrics. Keep the distinction from the small isolated-pixel beam/seam tests:
there, subpixel sampling more closely follows detector-plane coverage. Finer
sampling can still be chosen when convergence of the intended cake region
justifies its startup cost. Matching Qt/pyFAI is a separate objective; use the
compatibility mode for that objective, with the limitations in the
[full-splitting investigation](multi2d-splitting.md).

## Exclusions and reproduction

Image decoding, input conversion/preloading, calibration-file loading for the
native harness, process/threadpool startup, result-buffer destruction, Python
output-list getters, frontend serialization and GUI display are outside the
native timers. Python's synchronous wrapper drops its raw merge intermediates
before returning, whereas the native API returns them for coverage inspection.
This is an API-level latency measurement, not an end-to-end application launch
or detector-file-open benchmark.

HDF5 read/decode of the three Eiger frames was measured separately at
37.5, 38.9 and 35.7 ms in this run. This does not establish uncached disk I/O
performance. TIFF mask loading and conversion were also excluded.

```sh
# Existing local test environment needs numpy, pyFAI, h5py, hdf5plugin and fabio.
# Install/build the development wheel in an isolated environment first.
RAYON_NUM_THREADS=8 python python_examples/bench_multi2d_startup.py \
  --eiger-dir /path/to/Dioptrin/test_data \
  --report docs/validation/multi2d-startup.json
```

The script builds `examples/bench_multi2d_case.rs`, predecodes/validates images,
and runs each native method in a fresh subprocess. Temporary raw inputs and
impulse/result buffers are removed after comparison. `--case eiger_9M` or
`--case unequal_pair_1p5M` runs an individual case. No installed package is
modified by the benchmark script.
