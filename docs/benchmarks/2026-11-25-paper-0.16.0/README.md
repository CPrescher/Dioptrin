# Paper benchmark rerun (2026-11-25, Dioptrin 0.16.0)

Protocol for rerunning the paper comparison of Dioptrin, pyFAI and azint on the
three computers of the [24 November runs](../2026-11-24-paper-final/README.md)
(Windows workstation, DESY Maxwell, MacBook) with Dioptrin 0.16.0. The run
scripts are in [`benchmarks/comparison/paper-0.16.0/`](../../../benchmarks/comparison/paper-0.16.0/):

| Computer | Script | Start |
| --- | --- | --- |
| MacBook (M4 Pro, Metal) | `run-macos.sh` | `bash benchmarks/comparison/paper-0.16.0/run-macos.sh` |
| Windows (9800X3D, RTX 5070 Ti) | `run-windows.ps1` | `powershell -ExecutionPolicy Bypass -File benchmarks\comparison\paper-0.16.0\run-windows.ps1` |
| Maxwell (EPYC 9475F, H200 NVL) | `submit-maxwell.sh` | `bash benchmarks/comparison/paper-0.16.0/submit-maxwell.sh all` |

All three accept a dry run (`--dry-run`, `-DryRun`) that prints the command
lines, and a quick installation check (`--quick`, `-Quick`, `quick`).

## Source state

The benchmarks run on commit `8483c77` (v0.16.0 plus one commit that only
touches benchmark tests). The Mac passes ran at `8483c77`, started on
25 November from an untracked copy of `run-macos.sh` with the same passes,
options and output names. Later commits on this branch add only the scripts in
`benchmarks/comparison/paper-0.16.0/` and documentation. They change nothing
the harness checks:

- `compare.py` verifies the installed Dioptrin against `Cargo.toml`,
  `pyproject.toml` and every file under `src/` (`provenance.source_hashes`),
  plus the hash of the installed extension module. Nothing under
  `benchmarks/` or `docs/` is hashed, and the untracked `Cargo.lock` is not
  either.
- The Maxwell environment cache key (`environment.py`) adds the hashes of
  `benchmarks/comparison/{environment.py,setup.py,requirements.txt,provenance.py}`,
  which these commits do not change.

So an environment built at `8483c77` passes verification at the branch tip, and
the reverse. `environment.json` records the commit that was checked out; to
confirm that a later commit runs the same code:

```bash
git diff --stat 8483c77 HEAD -- src Cargo.toml pyproject.toml build.rs benchmarks/comparison/*.py benchmarks/comparison/datasets.json benchmarks/comparison/maxwell.sbatch
```

prints nothing.

## What changed against the 24 November runs

| Setting | 24 Nov | Now | Why |
| --- | --- | --- | --- |
| Dioptrin | 0.15.2 (`118a299`) | 0.16.0 (`8483c77`) | released version with the NumPy accessors and the new 2D tables |
| Dioptrin CPU patterns | `batch1d(..., num_workers=B, num_threads=1)` | `batch1d(frames, n)` (harness default `--dioptrin-1d-path global`) | the old call is 2.2 to 2.8 times slower |
| Dioptrin CPU cakes | one `integrate2d` per frame | `batch2d(frames, ...)` (`--dioptrin-2d-path batch --dioptrin-2d-batch-split nested`) | the library's normal call; measured on the Mac at 8 threads, Pilatus, 2000 x 360 with uncertainties: 585 against about 815 frames/s |
| Result conversion | outside the timed call | inside (automatic) | same as pyFAI and azint |
| Primary float64 pass | 1 trial, 3 blocks of 0.3 s | 3 trials, 3 blocks of 1 s (`--trials 3 --seconds 1.0`) | the single short trial was an open point |
| Frames per call at 12 and 16 threads (Windows) | 32 | 64 (`--frames 64`) | the 4-frame kernels need at least 4 frames per thread; measured on the Mac at 8 threads, Pilatus with uncertainties: 4180 frames/s with 32 or 64 frames, 1920 with 16 |
| New pass | none | call-path ablation (Dioptrin CPU only) | records default against split call (1D) and per-frame against batch (2D) on every computer |

Unchanged: datasets, 2000 bins, cakes 2000 x 360 and 1000 x 72, subpixels 10,
workers 1, 2, 4, 8, GPU precision `f32_compensated`, pyFAI 2026.9.0, azint
build flags. The float32 passes and the CUDA float64 passes keep the suite
defaults (1 trial, 3 blocks of 0.3 s). Further differences on Windows: the
float32 passes stop at 8 workers, and the SMT passes use 64 frames per call and
run 1 trial.

Not part of this set: the all-cores server runs on the scan subsets (at 48 and
96 threads Dioptrin needs 192 and 384 frames per call to stay on the fast
kernels). Those are prepared separately.

## Before starting, on every computer

1. Get the commit with these scripts and check that the tree is clean:

   ```bash
   git fetch origin
   git switch <branch with these scripts>   # or main, once merged
   git pull
   git status --short                        # must print nothing
   ```

2. `Cargo.lock` is not tracked. If the build stops with "cannot update the
   lock file ... because --locked was passed", the local lock file still names
   0.15.2. Fix it with:

   ```bash
   cargo update --offline -p dioptrin
   ```

3. Build (or rebuild) the benchmark environment, see the machine sections. The
   harness refuses to run if the installed Dioptrin does not match the source
   tree or lacks the array accessors. Do not pass `--allow-unverified-build`
   or `--allow-list-conversion` for paper runs; the scripts never do.
4. Run the quick check and read its result (below).
5. Close browsers and other programs, keep the computer from sleeping, and
   leave it alone while it runs.

### Reading the quick check

The quick check runs every variant of the machine's passes, GPU variants
included, on the small synthetic dataset at 1 and 2 workers (a few minutes).
In `benchmark-results/<date>-<machine>-quick/`:

- `skipped.json` must not list a GPU variant. A missing CUDA, Metal or OpenCL
  device does not fail the run; it only appears here and in the log as
  `SKIPPED`. On the Mac, `pyfai-full-omp` is skipped as expected (the macOS
  pyFAI wheel has no OpenMP).
- `failures.json` should be empty. The stock pyFAI 2026.9.0 full-splitting
  2D rows (`pyfai-full`, `pyfai-full-omp`) at 2 workers can fail their
  self-consistency check because of the uninitialised scratch buffer described
  in the 24 November README; that alone is not an installation problem. Any
  other failure is. When there are failures, `compare.py` exits with status 1
  and keeps its generated inputs in the result directory.

### Restarting

`compare.py` refuses to write into an existing result directory. The scripts
therefore skip every pass or job whose directory already exists. After an
interruption, delete (or rename) the directory of the unfinished pass and start
the script again; finished passes are not repeated.

## MacBook (M4 Pro + Metal)

```bash
brew install gcc   # once; azint needs it for OpenMP
python3 benchmarks/comparison/setup.py --with-metal --with-opencl --azint-fast-math
bash benchmarks/comparison/paper-0.16.0/run-macos.sh --quick
caffeinate -i bash benchmarks/comparison/paper-0.16.0/run-macos.sh
```

Progress: `benchmark-results/run-macos-0.16.0.log` and one `.log` per pass.

| # | Result directory | Contents | Duration (estimate) |
| --- | --- | --- | --- |
| 1 | `2026-11-25-macos-ablation-paths` | Dioptrin CPU only, 1D `global` and `split`, 2D `sequential`; 1 trial | about 1 h |
| 2 | `2026-11-25-macos-f64` | primary float64 pass, 1D and 2D, CPU, OpenCL and Metal; 3 trials of 3 x 1 s | about 11 h |
| 3 | `2026-11-25-macos-f32` | float32 inputs, same variants; 1 trial | about 3 h |

The durations are estimates from the 24 November run (990 rows in 3 h).

## Windows workstation (RTX 5070 Ti)

Build the environment in a Visual Studio Developer PowerShell (MSVC `cl` and
the Rust MSVC toolchain; no CUDA toolkit needed). `--azint-fast-math` is not
supported with MSVC, as on 24 November.

```powershell
python benchmarks/comparison/setup.py --with-cuda --with-opencl
```

The runs need no Developer PowerShell. From the checkout root, in Windows
PowerShell 5.1 or later:

```powershell
powershell -ExecutionPolicy Bypass -File benchmarks\comparison\paper-0.16.0\run-windows.ps1 -Quick
powershell -ExecutionPolicy Bypass -File benchmarks\comparison\paper-0.16.0\run-windows.ps1
```

Set the power plan so the computer does not sleep while the script runs.
Progress: `benchmark-results\run-windows-0.16.0.log` (start, exit code and time
of every pass) and one `.log` per pass. Each pass also leaves a
`benchmark-results\<name>.cmd` file with the exact command line it ran. The
script exits with status 1 if any pass failed. It was written without access
to Windows and has not been run there; if a pass fails at argument parsing, run
the line from its `.cmd` file (or from `-DryRun`) by hand.

The passes are in order of importance, so whatever is finished first is the
part the paper needs most. Stock `pyfai-full` and `pyfai-full-omp` are left out
of the 2D passes, as on 24 November.

| # | Result directory | Contents |
| --- | --- | --- |
| 1 | `2026-11-25-windows-ablation-paths` | Dioptrin CPU only, 1D `global` and `split`, 2D `sequential`; 1 trial |
| 2 | `2026-11-25-windows-f64-1d` | primary float64 1D, CPU, OpenCL and CUDA, 1 to 8 workers; 3 trials of 3 x 1 s |
| 3 | `2026-11-25-windows-f64-2d` | primary float64 2D, same; 3 trials of 3 x 1 s |
| 4 | `2026-11-25-windows-f64-1d-smt` | CPU variants at 1, 12 and 16 workers, `--frames 64`; 1 trial |
| 5 | `2026-11-25-windows-f64-2d-smt` | the same in 2D |
| 6 | `2026-11-25-windows-f32-1d` | float32 inputs, 1D, 1 to 8 workers; 1 trial |
| 7 | `2026-11-25-windows-f32-2d` | float32 inputs, 2D; 1 trial |
| 8 | `2026-11-25-windows-cuda-f64` | CUDA variants with `--gpu-precision f64`, 1D and 2D; 1 trial |

Duration: not measured on this computer. With the Mac's time per row, the two
primary passes (2 and 3) take about 10 h and all passes about 15 h (estimate).

`--frames 64` is rounded up to a multiple of the worker count, so the 12-worker
rows use 72 frames per call and the 1- and 16-worker rows 64.

## Maxwell (EPYC 9475F + H200 NVL)

On a login node, from the root of the checkout, after the steps in "Before
starting":

```bash
bash benchmarks/comparison/paper-0.16.0/submit-maxwell.sh quick
# wait for the job; read benchmark-results/2026-11-25-maxwell-quick/ (see above)
export PYFAI_PATCHED_WHEEL=/path/to/pyfai-2026.9.0-cp311-cp311-linux_x86_64.whl
bash benchmarks/comparison/paper-0.16.0/submit-maxwell.sh all
```

`PYFAI_PATCHED_WHEEL` is the wheel built from pyFAI 2026.9.0 with the one-line
`numpy.empty(256)` -> `numpy.zeros(256)` patch described in the 24 November
README. Without it (or if the file is missing) the script skips that one job
and submits the rest. Other settings, all optional: `PARTITION` (default
`upex`), `CONSTRAINT` (default `H200&GPUx1`), `RESULTS` (default
`$PWD/benchmark-results`). The script exports `WITH_OPENCL=1 WITH_CUDA=1
AZINT_FAST_MATH=1` to every job, sets `PYFAI_WHEEL` only for the patched job,
prints each job id and appends it to `$RESULTS/maxwell/jobs-2026-11-25.txt`.
`all` also submits the quick check if its result directory does not exist yet.
Jobs that start together do not build the same environment twice
(`environment.py` serialises builds with a file lock); the patched job builds
its own environment.

| Job | Result directory | Contents | `--time` (estimate) |
| --- | --- | --- | --- |
| quick | `2026-11-25-maxwell-quick` | installation check; builds and caches the environment | 0:30 |
| primary | `2026-11-25-maxwell-f64-1d` | float64 1D, CPU, OpenCL and CUDA; 3 trials of 3 x 1 s | 12:00 |
| primary | `2026-11-25-maxwell-f64-2d-fine` | float64 2D, `2000x360` only; 3 trials of 3 x 1 s | 12:00 |
| primary | `2026-11-25-maxwell-f64-2d-coarse` | float64 2D, `1000x72` only; 3 trials of 3 x 1 s | 12:00 |
| primary | `2026-11-25-maxwell-f64-2d-pyfai-patched` | patched `pyfai-full`, `pyfai-full-omp`, `pyfai-full-opencl`, 2D, both cakes; 3 trials of 3 x 1 s | 12:00 |
| optional | `2026-11-25-maxwell-ablation-paths` | Dioptrin CPU only, 1D `global` and `split`, 2D `sequential`; 1 trial | 4:00 |
| optional | `2026-11-25-maxwell-f32-1d` | float32 inputs, 1D; 1 trial | 6:00 |
| optional | `2026-11-25-maxwell-f32-2d` | float32 inputs, 2D; 1 trial | 8:00 |
| optional | `2026-11-25-maxwell-cuda-f64` | CUDA variants with `--gpu-precision f64`, 1D and 2D; 1 trial | 4:00 |

The time limits are estimates from the 24 November job (1056 rows in about
5 h at 1 trial). If `upex` does not accept 12 h, lower `--time` in the script;
the 2D job is already split by cake size for that reason. The stock 2D jobs
leave out `pyfai-full` and `pyfai-full-omp`; on 24 November the stock run was
kept to document the 54 failures, and that record still stands.

`compare.py`'s own `--output` is given after the one `maxwell.sbatch` passes,
so it takes precedence: the results land in `$RESULTS/<result directory>`,
while `$RESULTS/maxwell/<job-id>-*/` holds the job log, build record and
`environment.json` of the job, and `$RESULTS/maxwell/dioptrin-<job-id>.out`
the Slurm output.

## What to collect

From each computer:

- every `2026-11-25-*` result directory (each has `results.csv`,
  `summary.md`, `environment.json`, `numerical.json`, `failures.json`,
  `skipped.json`);
- the per-pass `.log` files and the run log (`run-macos-0.16.0.log`,
  `run-windows-0.16.0.log`); on Windows also the `.cmd` files;
- from Maxwell also the `maxwell/<job-id>-*` folders,
  `maxwell/dioptrin-<job-id>.out` and `maxwell/jobs-2026-11-25.txt`.

Archive them outside Git next to the 24 November records, in a folder
`2026-11-25-paper-0.16.0/`. The tables, Figure 7 and the text of the paper are
rebuilt from them.

## Measured results

Pending.
