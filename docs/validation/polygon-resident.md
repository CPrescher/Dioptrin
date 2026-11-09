# Resident fixed-mask 1D execution

`Fixed1dPlan::integrate_frame` can prepare a bin-oriented table for complete
float64 images already in detector order. The table reuses the existing polygon
coefficients and fixed-mask normalization. It groups contributions by bin,
original CSC chunk, and original pixel-modulo-four lane. Each lane retains its
pixel order and fused multiply-add operations; lanes and chunks are reduced in
the same order as the existing CSC implementation. This preserves intensity
and Poisson standard-deviation bits while parallelizing over output bins and
avoiding histogram scatter/merge work.

Preparation is lazy. It applies outside outer Rayon workers, with at least 64
bins and two existing CSC chunks, and only when the additional resident table
fits within 128 MiB. Other cases retain CSC. Streamed/typed block workers keep
their existing execution and do not allocate this resident table. Subsequent
frames reuse coefficients and normalization, while reading the new frame's
samples.

`with_csc_layout()` explicitly forces the pixel-oriented kernel, bypassing this
optional gather table and timing probes. The desktop uses this as its default
to preserve large-detector throughput.

`with_auto_layout()` opts into a different choice: ordinary fast CSR versus
CSC for complete resident frames. A bounded `predict_1d_layout` model supplies
an initial recommendation. Five alternating paired probes then measure this
actual plan and data after warming both layouts. CSR must win at least four
pairs and reduce median time by more than 15%; otherwise CSC stays selected.
The decision is cached independently for intensity-only and Poisson calls.
`selected_resident_layout(errors)` exposes the measured choice. A new immutable
plan is needed for changed geometry, mask, corrections or bin count. This adds
first-call overhead and can retain a second masked CSR table. Ordinary CSR uses
the same weights but has a different floating-point reduction order; bitwise
parity is claimed only for the CSC-order gather path above.

Automatic selection does not probe inside outer Rayon workers and does not
alter streamed/typed block execution. Those paths retain CSC. An isolated
kernel probe inside a parallel batch would not measure full-batch contention.
Strict compatibility CSR, cakes, sectors, dynamic masks and externally supplied
variance are outside this selector's qualification.

The prior was fitted on an M4 Pro using eight threads, 0.29–10.14 MP and
128–8000 radial bins, four detector files plus two software-binned derivatives.
It models log(CSC/CSR time) using log pixel count, log bin count, squared log
bin count, errors and frame parallelism. Eight-thread/range guards return CSC
outside the measured domain. Its 96.7% in-sample decisive-case accuracy drops
to 86.9% when holding out detector families (all Eiger derivatives stay together).
It is a timing prior, not a universal pixel/bin threshold. The desktop's
`INTEGRATION_LAYOUT_CROSSOVER.md` retains the fitted relationship, validation
limits and raw measurement provenance.

Regression tests compare the old CSC reduction with the new layout bitwise for
weighted/unweighted tables, duplicates, masks, changed positive/negative samples,
Poisson errors and 1/4/8-thread pools. A complete polygon-frame test also checks
the lazy cache, error bars, and the outer-worker fallback. The matched desktop
benchmark retains old/new arrays and timing samples in
`desktop/artifacts/polygon-cache-optimization` in the Dioptas checkout.
