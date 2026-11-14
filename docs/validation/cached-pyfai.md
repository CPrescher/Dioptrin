# Cached compatibility plans

`FixedCsr1dPlan::new(lut, normalization, mask)` freezes an existing public 1D
LUT with a fixed mask. It uses strict CSR accumulation, compacts the fixed mask,
and caches bin normalization. It preserves the arithmetic of the public
`lut::integrate1d` / Poisson-error functions without changing LUT weights.

`Integrator::prepare_fixed2d_on_grid(shape, grid)` freezes the shared-grid
geometry, weights, corrections, mask and arithmetic policy. Its
`integrate_frame(data, errors)` checks the frame length and unmasked finite
samples, then produces normalized intensity directly. It avoids repeated static
geometry scans, coverage-count accumulation and raw output arrays. Geometry,
unit, polarization and mask edits to the original integrator do not mutate
an existing snapshot; the caller must prepare a new plan for changed settings.
Dynamic-mask integrators are rejected by the fixed-grid constructor.

The Rust regressions compare the snapshots bitwise against the existing
strict 1D LUT and raw 2D sum/division paths. Cases include four radial units,
sectors and azimuth wrapping, beam-centered geometry, masks, changed frame
samples, Poisson errors, strict/fast cake arithmetic, malformed LUT rejection,
and snapshot isolation after source-integrator edits. Splitting conventions,
singular-beam handling and approximate pyFAI compatibility are unchanged.

`FixedCsr1dInputPlan` shares a strict 1D plan and maps decoder addresses once
for detector order or a vertical flip. It accepts uint16/uint32 pixels directly,
avoiding a full-frame float64 conversion. Serial entry points support separate
frame workers; explicit parallel entry points schedule independent bins within
one image while keeping the strict summation order inside each bin. This path
requires uncorrected integer exposures with a fixed mask; corrected pixels,
dynamic masks, and supplied variances use their existing execution paths.
