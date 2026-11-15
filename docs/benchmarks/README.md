# Benchmarks in the repository

This directory holds dated benchmark write-ups and their records. This page
describes the criterion benches in `benches/`, what each one times, and how to
compare two revisions with them. It is not part of the Sphinx manual
(`benchmarks/**` is excluded in `docs/conf.py`).

For comparisons against azint and pyFAI, see
[`benchmarks/comparison/`](../../benchmarks/comparison/README.md); for the
Python batch and stream methods, `python_examples/bench_batch_streams.py`
(below).

## Bench targets

All targets use `harness = false` criterion benches, read their fixtures from
`test_data/` relative to the repository root, and are not part of the
published crate. "Cold" means a new `Integrator` (or plan) per iteration, so
the time includes geometry arrays, corrections and the lookup table; "warm"
means the table is cached by an untimed first call. Unless stated otherwise
the input is the committed Pilatus 1M CeO2 image (1043 x 981) with
`test_data/v1.poni`, 1D uses 1250 bins and 2D 1000 x 360.

| Target | What it times | Compiles against v0.12.0 |
|---|---|---|
| `build` | LUT builders called directly: supersampled and pixel-split, 1D (1250 bins) and 2D (1250 x 360); cold, no detector | yes |
| `integrate` | `lut::integrate1d`/`integrate2d` on prebuilt counting, supersampled and pixel-split tables; warm | yes |
| `integrator` | `Integrator` 1D (histogram, counting, pixel-split, supersampled, with errors, q unit, polarisation) and 2D (counting, pixel-split, supersampled, with errors, start/step azimuthal range); warm | yes |
| `methods` | counting, `pixel_split`, `supersampled4` (n_ss = 4) and `pyfai_full` side by side: cold 1D/2D (`methods/build1d/*`, `methods/build2d/*`), the 1D `pyfai_full` table builder alone (`methods/lut1d/pyfai_full`), warm 1D/2D (`methods/integrate1d/*`, `methods/integrate2d/*`) | yes |
| `azimuthal` | 2D on a 90 degree sector and the full circle: true-range API (`azimuthal/range_sector/*`, `azimuthal/range_full/*`), start/step API (`azimuthal/start_step/*`), bracket API (`azimuthal/brackets/*`), warm; cold sector build with the supersampled radial clamping (`azimuthal/build_sector/*`); for `pixel_split`, `supersampled4`, `pyfai_full` | no (0.13 APIs) |
| `detectors` | `Detector::builtin_mask` for the circular-mask MAR 345 and Rayonix133 (unbinned, binned, transformed) and the gapped Eiger2 4M; `Detector::position` at every pixel of uniform (Pilatus 1M, Eiger2 4M binned), separable (Jungfrau 500k, curved CirPAD), transformed, binned and binned transformed maps; with a detector attached via `Geometry::with_detector`: the 1D `pyfai_full` table builder (corner-sign vote included), cold 1D `pixel_split` and `pyfai_full`, warm 1D `pyfai_full`. A transform of a uniform catalogue map only swaps shape, pixel sizes and mask; the Jungfrau 500k cases are the ones that go through `PixelMap::Transformed` | yes |
| `plans` | `Fixed1dPlan::integrate_frame` (forced CSC, auto layout; the plan as prepared probes a gather kernel on first use and is left out), `Integrator` 1D with `SparseLayout::Csc` vs `Csr`, `FixedCsr1dPlan` (f64) and `FixedCsr1dInputPlan` (u16, u32, u32 with parallel bins), `FixedGrid2dPlan` (`pixel_split`, `pyfai_full`); 1M (CeO2) and 4M (synthetic 2048 x 2048); warm | no (0.13 APIs) |
| `multi` | `MultiIntegrator` 1D and 2D with 2 and 3 detectors (CeO2 plus two synthetic detectors with other pixel sizes, tilts and beam positions) on one shared range; warm, and cold with 3 detectors; `pixel_split`, `pyfai_full` | yes |
| `batch` | `process_streaming` on in-memory frames (default read-ahead), 1D and 2D, 2 and 8 workers, 200 synthetic 256 x 256 frames and 16 CeO2 1M frames (frames per second; copying the frames is not timed, freeing them in the workers is; before `process_streaming` replaced it on main after 0.13.0, the same names timed `process_queue_streaming`); `process_files` on 16 CeO2 paths | no (`process_streaming` is new after 0.13) |
| `massif` | Massif peak search on the LaB6 MarCCD image (2048 x 2048): `find_peaks` (prepares on each call), `PreparedMassif::new`, one seed on a reused `PreparedMassif`, four seeds without and with reuse | no (`PreparedMassif` is new in 0.13) |

Benchmark IDs of the new targets are `<target>/<case>/<variant>`, so the IDs
of one revision line up with those of another. Shared inputs are in
`benches/common/mod.rs`; synthetic frames come from a fixed-seed generator.
The older `build`, `integrate` and `integrator` targets keep their original
IDs; `integrate` registers its two pixel-split IDs twice each, and criterion
reports the second of each as `... #2`.

### Running

```bash
cargo bench --bench methods                      # one target
cargo bench --bench methods -- 'methods/build1d/'   # one family (a regex)
cargo bench --bench detectors -- 'pyfai'         # every ID containing "pyfai"
cargo bench --bench plans -- --test              # run each once, no timing
```

`cargo bench` without `--bench` runs every target, including the three older
ones. The new targets use 0.3 s warm-up and 1.5 s measurement (20 samples;
cold builds 10 flat samples of one iteration each), favouring coverage over
precision: criterion warns "Unable to complete 20 samples in 1.5s" for some
cases, which only means the case took a little longer. On an Apple M4 Pro
under load from other builds (one-minute load 14 to 45), the timed runs took
43 s (methods), 48 s (azimuthal), 105 s (detectors), 64 s (plans), 59 s
(multi), 36 s (batch) and 17 s (massif), about 6 minutes in all for 115
benchmarks, plus about a minute to build the seven binaries with the release
profile's LTO. Pass a longer measurement time (`-- --measurement-time 5`) for
tighter intervals on a quiet machine.

The CI Benchmarks job (`.github/workflows/test.yml`) only compiles the three
older targets (`--no-run`); the new targets are compiled, unoptimised, by the
Clippy and Test jobs (`--all-targets` in `.github/workflows/ci.yml`) and are
not run in CI.

## Comparing two revisions

`scripts/bench_compare.py` checks out both revisions into temporary git
worktrees, copies this checkout's `benches/` and bench declarations into both
(so a revision older than the benches is timed with the same code), builds
both, runs the binaries in alternating rounds and prints a table:

```bash
python3 scripts/bench_compare.py v0.12.0 HEAD --benches methods detectors multi
RAYON_NUM_THREADS=1 python3 scripts/bench_compare.py main HEAD \
    --benches methods --filter 'methods/(build|integrate)1d/' --rounds 3
```

The table (also written to `target/bench-compare/<timestamp>/comparison.md`
with the raw criterion data, logs and `meta.json`) gives for each ID the median
over rounds of criterion's median time per revision, the ratio B/A, and the
range over rounds (or criterion's 95% interval of the median with one round).
Rows whose ranges overlap are marked "overlaps"; read the others against the
range widths, not as verdicts. `--report-only --out DIR` rebuilds the table
from an earlier run, and `--keep` keeps the worktrees and the build directory.

Limits:

- One round per revision is easily misled by a burst of load; the default
  is three rounds (an odd count, so the median over rounds is one of the
  rounds). In a 2-round test run of v0.12.0 against 0.13.0 on a loaded
  machine, one round of the warm 1D cases was 3x slower than the other.
- Criterion cannot interleave two binaries sample by sample. Load that changes
  between one revision's run and the other's shows up as a difference. The
  script alternates the order (A B, B A, ...) to spread drift over both;
  more `--rounds` and a quiet machine help. Check `uptime` first; the script
  records the one-minute load at the start and end of every run.
- A target that does not compile on one revision (see the table above) fails
  the build; leave it out with `--benches`.
- Both revisions are built with their own `Cargo.toml` profile settings.
- A revision is a commit: uncommitted library changes are not included.
- Without `--benches` all seven new targets run, which only works for
  revisions from 0.13 on, and `batch` only for revisions with
  `process_streaming` (after 0.13).
- Fixtures missing from a revision's `test_data/`, or present only as Git
  LFS pointers, are copied from this checkout.
- The environment is passed on unchanged, including `RAYON_NUM_THREADS`.

For kernel costs, use `RAYON_NUM_THREADS=1`: single-threaded times are far
less sensitive to other processes and to scheduling, and a change in a kernel
shows up undiluted. Multi-threaded runs are what users see, but on a shared
machine they mostly measure the other processes. The `batch` target sets one
thread per worker itself and varies the worker count.

`critcmp` is not needed. Without the script, the same comparison works by
hand with criterion's baselines: run `cargo bench -- --save-baseline a` on
one revision and `cargo bench -- --save-baseline b` on the other with the
same `CRITERION_HOME` (otherwise `target/criterion` of each checkout), then
compare the `estimates.json` files or use an external tool such as `critcmp`.

## Python batch and stream throughput

`python_examples/bench_batch_streams.py` times `batch1d`, `batch2d` (TIFF
paths), `batch1d_iter` and `batch2d_iter` over frame sizes, worker counts and
input kinds (`list-arrays`, `gen-arrays`, `list-paths`, `gen-paths`), and
prints frames per second as the median over repeats with min, max and spread.
It needs the extension built from the revision under test
(`maturin develop --release --features python`). `test_examples.py` skips
`bench_*` scripts and CI does not run it.

```bash
python python_examples/bench_batch_streams.py --sizes 256 1024 --workers 2 8 --repeat 5
python python_examples/bench_batch_streams.py --apis batch2d batch2d_iter --inputs list-paths gen-paths
```

## Not covered

- The 2D `pyfai_full` table builder and the `lut::prepared` types alone: both
  are crate-private, so they are timed through the `Integrator` (cold 2D,
  `FixedGrid2dPlan`, `Fixed1dPlan`).
- `interval_mask` (private) is timed through `builtin_mask` of a gapped
  detector; there is no separate `outside_circle` function.
- Detectors with a spline distortion, 2D integration with a detector
  attached, and transformed maps of curved detectors (CirPAD is timed
  untransformed only).
- The dynamic mask mode (`Dynamic1dPlan`), and error propagation and q units
  outside the older `integrator` target.
- Python overheads other than the batch and stream methods.
