# Integration performance and float64 precision

[Manual index](index.md)

Dioptrin performs diffraction integration. The separate
[dioptrin-io performance guide](https://dioptrin-io.readthedocs.io/en/latest/PERFORMANCE.html)
owns file-reading comparisons, sparse/dense image definitions, scalar/SSE2/AVX2/NEON
backends and the Windows read/decode measurements. Those reader rates exclude
masking, normalization and integration.

## Fixed and dynamic masks

Fast 1D integration defaults to CSC with float64 geometry weights and
accumulators. The two mask policies reuse the same geometry-only coefficients:

| Policy | Work reused | Work when processing a frame |
|---|---|---|
| `fixed` | Active pixels and bin normalization, while their inputs stay unchanged | Accumulate signal and requested uncertainty terms |
| `dynamic` | Geometry coefficients and prepared sparse tables | Apply the supplied effective mask and accumulate signal, normalization and requested uncertainty terms together |

A policy switch does not rebuild geometry or discard valid weights and caches.
Changing a mask invalidates its dependent fixed-mask caches; changing geometry,
binning or corrections invalidates the corresponding scientific state. A dynamic
policy does not create a mask: the application supplies the effective mask for
each frame. Mask plugins and image preparation have their own costs.

Select `mask_mode="fixed"` or `mask_mode="dynamic"` when constructing an
integrator, or call `set_mask_mode`. To compare layouts explicitly, use
`sparse_layout="csr"` or `set_sparse_layout("csr")`.

## Accumulation and precision

Fast 1D CSC execution uses four partial histograms to reduce dependent updates
into the same output bins. Float64 weights and accumulators are retained. Fast
arithmetic changes summation order and may use hardware fused multiply-add;
results can therefore differ in rounding from strict accumulation.

`fast_math=False` selects strict arithmetic. Strict arithmetic, 2D integration
and azimuthal spread retain their compatible existing paths; selecting CSC does
not replace every algorithm. See the [Python API](python-api.md) for mask,
layout and arithmetic settings.

## Workers and setup time

For frame parallelism with changing masks, an application can prepare one
immutable `prepare_dynamic1d` plan and create a worker per application worker.
Workers share geometry, float64 tables and corrections. Each call supplies its
own image and mask and produces its own normalization and result arrays.
Rebuild the plan when its scientific settings change. Reading, mask generation,
frame ordering and cancellation remain application responsibilities.

Rust applications can retain `Dynamic1dWorker` scratch across frames:

```rust,ignore
let plan = integrator.prepare_dynamic1d((height, width), bins)?;
let mut worker = plan.worker(true);
let result = worker.integrate(
    &corrected_counts, Some(&effective_mask), Some(&pixel_variances), false,
)?;
```

The CSC worker recomputes normalization for each mask and reuses private signal,
normalization and variance histograms. Supplied variances must be finite,
nonnegative and match the image; they replace `max(signal, 0)`, preserving
original counting noise after background subtraction and deterministic
corrections. Without supplied variances, counting errors retain the Poisson
estimate. `Dynamic1dPlan::new(shape, lut, normalization)` accepts an existing
grid/sector LUT and shared normalization without baking in a mask. Azimuthal
spread retains its row-statistics implementation within the same plan.

Use one compute thread per frame worker when the application already processes
frames in parallel. More workers do not guarantee higher throughput: decoding,
mask plugins, memory bandwidth and retained frame buffers can limit the batch.
Report lookup-table and worker setup separately from steady-state processing.

## Matched comparisons

Keep the same input frames, geometry, masks, corrections, binning, arithmetic
and uncertainty settings. State whether a timing includes file reads, decoding,
float64 conversion, image copies, mask creation, normalization, integration and
result handling. The current Python array interface accepts C-contiguous 2D
float64 arrays; conversion belongs in a complete pipeline measurement.

Record CPU allocation, worker/thread counts, cache state, host load and the
range of repeated measurements. Compare layouts on the same machine and report
setup time alongside throughput. Historical
[integration benchmark records](https://github.com/CPrescher/Dioptrin/tree/main/docs/benchmarks)
retain the settings and scope of their tested revisions. For native file-reader
benchmarks and their downloadable records, use the
[dioptrin-io manual](https://dioptrin-io.readthedocs.io/en/latest/PERFORMANCE.html).

## Fused block decoding and combined Qt loading

The optional dioptrin-io reader decodes and integrates independent blocks of one
HDF5 frame across eight cores, sharing a fixed-mask float64 lookup plan. The native
Eiger 9M benchmark reduced read-to-pattern latency from 16.73 to 5.38 ms for
intensity and from 17.31 to 6.09 ms with Poisson errors. Reading, thread startup,
merging and normalization were included; file cache and lookup were warm. Lookup
preparation was about 0.78 s separately.

Qt's first adapter reread a frame already decoded for display and therefore ran
slower. The combined loader now returns both original integer display pixels and
a float64 pattern from one decode. Its task consumes that prepared pattern;
settings edits use the resident image rather than reading it again. All 625
original count hashes match independent h5py references; float64 intensity and
Poisson differences remain at rounding scale.

The full Qt benchmark—with real image/pattern controllers, snapshot creation,
worker dispatch, pattern application and the normal 25 ms debounce—shows about
**20% lower warm open-to-pattern latency** on the tested M4 Pro/Eiger 9M setup.
This whole-operation result differs from the roughly threefold native speedup.
Qt was offscreen at 1200×900; physical monitor/GPU presentation was not qualified.
The [dioptrin-io combined benchmark report](https://dioptrin-io.readthedocs.io/en/latest/benchmarks/2026-11-07-qt-combined/README.html)
records the exact input/settings, median/p95 observations, raw timings, source
provenance, one-off setup and all-frame validation. A copy is maintained in
Dioptas's `docs/benchmarks/2026-11-07-combined-hdf5.md`.

This optional source-built adapter supports fixed-mask 1D integration of compatible
Eiger-style HDF5 frames. Original display counts/dtype are preserved; integration
weights, sums and uncertainty terms remain float64. Dynamic masks, corrections,
transforms, unsupported layouts, batch/map processing and 2D-cake algorithms keep
their existing routes. No released packages change merely from these source edits.
