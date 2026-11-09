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

Regression tests compare the old CSC reduction with the new layout bitwise for
weighted/unweighted tables, duplicates, masks, changed positive/negative samples,
Poisson errors and 1/4/8-thread pools. A complete polygon-frame test also checks
the lazy cache, error bars, and the outer-worker fallback. The matched desktop
benchmark retains old/new arrays and timing samples in
`desktop/artifacts/polygon-cache-optimization` in the Dioptas checkout.
