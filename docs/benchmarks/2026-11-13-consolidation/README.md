# Performance consolidation validation (2026-11-13)

The nine commits through `0c0d4fc` are merged with main `a7e8df5`, retaining
its numerical fixes. The owning polygon/counting/supersampling builders,
histogram routines, Python range routing, and existing regression/reference
outputs are retained. The overlapping grid helpers share validation and cache
keys without replacing the corrected edge policies.

Distant finite azimuthal starts are corrected automatically: `(-3π, -π)`
uses and returns `(-π, π)`. Spans above one turn, including `(-4π, 4π)`, are
rejected. Previously accepted ranges, sector clipping, non-finite pixel
propagation, and documented pyFAI seam dependence remain unchanged.

## Numerical checks

- 252 Rust unit tests and 6 detector-reference tests pass in debug and release;
  release also passes 16 doctests. Debug all-target tests exercise the benchmark
  harnesses and examples.
- Strict Clippy passes for all targets, both with and without Python bindings.
- 36 Python regression tests pass on an installed release extension, including
  array/file batches, iterators, masks, errors, pyFAI comparisons, and ranges.
- Typed TIFF/persistent-worker checks and the standalone pyFAI comparison pass.
  The example runner passes 23 scripts, with 6 existing data-dependent skips.
- Strict Sphinx HTML and Rust formatting checks pass.
- No existing numerical reference arrays were replaced.

New checks compare cached 2D snapshots bitwise with corrected true-range
integration for four methods, degree/q units, fast/strict policies, partial/all
masks, full-circle/seam-crossing ranges, changed/negative samples, and error
bars. Typed integer 1D snapshots are compared with corrected pixel-fraction
and clipped-range integration, including vertical input order and bin-parallel
execution. Shifted-range tests cover direct Python calls, eager TIFF batches,
array/generator streams, corrected output axes, and rejection of multiple turns.

## Scoped local timings

Apple M4 Pro, eight Rayon threads, Rust 1.99.1, optimized/LTO build.
CeO2 Pilatus 1M (`v1.poni`), 1043 × 981 pixels. Three fresh processes, each
with three warmups and fifteen samples per operation. The table shows the
median of their medians in milliseconds. Negative TIFF values are masked;
unsigned counts and float64 images are prepared before timing.

| Poisson errors | Cached strict float64 CSR | Serial uint32 CSR | Parallel uint32 CSR | Ordinary shared-grid cake | Cached cake |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 0.253 | 1.686 | 0.249 | 1.885 | 0.424 |
| On | 0.287 | 1.881 | 0.262 | 1.967 | 0.509 |

1D uses 1000 pyFAI-compatible bins; 2D uses fractional polygons with 300 × 180
bins. All timed alternatives are checked bitwise against the corrected ordinary
paths before timing. Cached cakes are about 4× faster in this fixture. Direct
parallel integer CSR is similar to cached parallel float64 CSR; it additionally
avoids a caller's full-frame conversion, which is excluded here. Serial integer
CSR is intended for workers processing separate frames and is slower than the
multicore paths for one image. This does not measure decoding, display, cold
setup, or end-to-end application throughput. The workstation is shared; these
are informational measurements, not a performance gate or cross-platform claim.

Raw measurements are in `trial-1.json`, `trial-2.json`, and `trial-3.json`.
Reproduce from the repository root:

```sh
RAYON_NUM_THREADS=8 cargo run --release --example benchmark_consolidated
```

The existing Massif and layout-crossover reports retain their original workload
scope. This consolidation does not establish new speed claims for those paths.

## Independent review

Claude Fable (`claude-fable-5-1`) reviewed PR #18 at `71cb466` after it was
opened. The [initial review](claude-fable-initial-review.md) found no regression
in the retained main fixes, and identified a supplied-variance CSR fallback that
incorrectly reused the sample-filtering spread kernel. A new regression fails
on the original fallback (finite intensity 1.5 instead of 5 with a zero-normalization
contribution) and passes with the direct CSR fallback. It also covers NaN/Inf,
signed and zero weights, masked/empty bins, and strict/fast CSR/CSC policies.
The optional spread computation keeps its documented filtering behavior.

The review's layout observations are documented in the resident-plan manual;
layout builder calls keep last-selection-wins behavior. The Python stub now
states the resolvable-edge requirement, and the duplicate layout test value
was replaced by 4000 bins. No application scheduling policy was changed.
