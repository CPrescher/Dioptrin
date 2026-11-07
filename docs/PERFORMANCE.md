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
