# Initial Claude Fable review

Model: `claude-fable-5-1`. Reviewed PR #18 at `71cb466` on 2026-11-13.
Findings below describe that revision; the accompanying validation report records the resolutions.

Review of PR #18 (71cb466 vs origin/main a7e8df5). Nothing was modified, committed, pushed or posted.

**Verdict:** no confirmed numerical regression or API break from the consolidation. One low-severity inconsistency in a library fallback path, two low-severity observations, and doc nits. I consider it ready to merge, with finding 1 as a follow-up.

## Findings

**1. Low, confirmed by code reading. `Dynamic1dWorker::integrate` changes non-finite handling on its fallback path.**
- Where: `src/integration/dynamic.rs:234-248`. When the plan has no compact CSC (strict math, CSR layout, or a table too large for u32 indices), a frame with supplied variances is routed through the azimuthal-spread kernel at `src/lut/azimuthal.rs:61-84`. That kernel silently drops non-finite signal, nonpositive normalization and nonpositive fractions, and returns 0 for bins without a valid contribution. The CSC worker path at `src/lut/csc.rs:280-348` propagates NaN and Inf as documented.
- Trigger: build the plan from an integrator with strict math or CSR layout, then call the worker with a NaN pixel and supplied variances.
- Impact: identical inputs produce different outputs depending on plan configuration. Dioptas is not affected because it always builds fast CSC plans.
- Suggestion: route variance-without-CSC through a strict CSR kernel with explicit variance, or document the fallback.

**2. Low, pre-existing, Dioptas-side observation. The resident gather table and automatic layout are unreachable from Dioptas.**
- `src/integration/fixed.rs:243-246` and `263-264` fall back to plain CSC whenever the call runs on a Rayon worker. Dioptas calls `integrate_frame` only inside its worker pool (`backend/src/integration.rs:1077`, `1126`, `1194`), so "auto" means CSC there and the selected layout stays unset.
- Numerically harmless. Not introduced by this PR.

**3. Low, doc versus behaviour. Default `Fixed1dPlan::integrate_frame` still runs wall-clock timing probes.**
- `src/integration/fixed.rs:269-270, 291-315`. Without an explicit layout call, the first resident call per error mode runs 12 extra integrations and picks a layout by timing. The CHANGELOG sentence "No runtime timing probes are used" describes the opt-in selector only. I verified by reading that the gather table reproduces the CSC chunk, lane and entry order, so results are bitwise identical either way.

**4. Nit.** Calling `with_auto_layout` after `with_csc_layout` silently overrides the forced CSC (`fixed.rs:126-137`).

**5. Nit.** `dioptrin.pyi:380` says "min at any finite angle". Starts whose magnitude makes the bin step unresolvable, such as 1e17, are rejected. The Rust docs say "finite and resolvable", which is accurate.

**6. Nit.** The layout test list at `src/integration/layout.rs:41` repeats 2000.

## Verified as correct

- **Range correction.** I ran the installed extension on the 64x96 fixture for counting, pixel_split, supersampled and pyfai_full, with and without errors. (-3π,-π) equals (-π,π), (-7,-6) equals (-7+2π,-6+2π), and (7,8) equals (7-2π,8-2π) bit-for-bit in intensity, errors and axis. (-4π,4π), (-2π,2π+1e-9) and non-finite edges are rejected. MultiIntegrator gives the same result for shifted and unshifted full turns. The cache key uses the corrected grid, so no spurious rebuilds. Ranges inside ±2π are untouched and the pyFAI seam rule in `pyfai_split.rs:43-49` is unchanged. Python batch validation shares the same constructor.
- **Strict CSR snapshots.** `FixedCsr1dPlan` reproduces the summation order of the public 1D routines and the Integrator strict path. Integer inputs convert exactly. The FlipVertical column remap and mask compaction are consistent between construction and execution.
- **Cached cakes.** The snapshot copies geometry, method, unit, fast math, polarization, radial range and mask. The Integrator has no other correction input. The direct intensity kernel uses the same signal lanes and cached normalization as the raw path, with the same zero-norm division. Histogram and dynamic masks are rejected.
- **Dynamic plans.** The prepared table is frozen before workers start. The CSC worker equals the single-chunk reduction of the existing path. Shapes and variances are validated.
- **Massif.** Per-pixel tap order and the reflection rule are preserved, and the prepared search is the old body verbatim.
- **API.** No public removals. Python signatures unchanged.

## Validation limits

I did not run the Rust or Dioptas test suites myself and relied on your reported runs plus source reading. The Python reproduction used only the synthetic fixture. Finding 1 was established by reading, not execution. I reproduced no performance numbers.
