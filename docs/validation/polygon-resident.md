# Resident fixed-mask 1D execution

`Fixed1dPlan::integrate_frame` can prepare a bin-oriented table for complete
float64 images already in detector order. The table reuses the existing polygon
coefficients and fixed-mask normalization. It groups contributions by bin,
original CSC chunk, and original pixel-modulo-four lane. Each lane retains its
pixel order and fused multiply-add operations; lanes and chunks are reduced in
the same order as the existing CSC implementation. This preserves intensity
and Poisson standard-deviation bits while parallelizing over output bins and
avoiding histogram scatter/merge work.

Preparation is lazy. The default resident gather path times both kernels on
first use in each error mode (two warmups and ten timed integrations) and uses
the gather table only with a consistent measured advantage. It applies outside outer Rayon workers, with at least 64
bins and two existing CSC chunks, and only when the additional resident table
fits within 128 MiB. Other cases retain CSC. Streamed/typed block workers keep
their existing execution and do not allocate this resident table. Subsequent
frames reuse coefficients and normalization, while reading the new frame's
samples.

`with_csc_layout()` explicitly forces the pixel-oriented kernel, bypassing this
optional gather table and timing probes. The desktop exposes this as a manual override.

`with_auto_layout()` opts into a broad estimate rather than runtime timing:
CSR generally; CSC for images above about 8 MP at 2000 bins or more
without Poisson errors. The desktop defaults to CSR with manual overrides and
an optional Automatic estimate. The
`predict_1d_layout` API returns the recommendation and approximate crossover
large-image threshold. `selected_resident_layout(errors)` exposes the selected resident
orientation. It is deliberately a ballpark, not a CPU-specific timing guarantee.
It adds no timing probes. Preparing a selected CSR path can retain an additional
masked CSR table. Ordinary CSR uses the same weights but a different summation
order; bitwise parity is claimed only for the CSC-order gather path above.

Automatic selection leaves outer Rayon workers and streamed/typed block
execution on CSC, preserving the large-detector throughput baseline. Sectors,
dynamic masks, external variances and spread retain their general paths. Cakes
and pyFAI-compatible integration are outside this selector's qualification.

The frozen benchmark fit and raw observations remain in
`docs/benchmarks/2026-11-09-layout-crossover/` as research evidence. The fitted
relationship is not the production policy: the user chose a broad estimate and
manual overrides instead. The sweep covered four real detector files and two
software-binned derivatives, 0.29–10.14 MP, 128–8000 bins and eight M4 Pro threads.

Regression tests compare the old CSC reduction with the new layout bitwise for
weighted/unweighted tables, duplicates, masks, changed positive/negative samples,
Poisson errors and 1/4/8-thread pools. A complete polygon-frame test also checks
the lazy cache, error bars, and the outer-worker fallback. The matched desktop
benchmark retains old/new arrays and timing samples in
`desktop/artifacts/polygon-cache-optimization` in the Dioptas checkout.

In Dioptas, current calls enter through an outer Rayon worker pool. Those calls
retain CSC under this guard, including the opt-in selector; resident selection
inside that application is not qualified by this library change. Layout builder
methods use the last explicit selection: `with_auto_layout()` after
`with_csc_layout()` selects the estimate.
