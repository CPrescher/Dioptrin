# Reproducible CPU integration benchmarks

Compare Dioptrin, azint, and pyFAI from a normal Git clone. The **standard suite
requires no private files, manual data copying, or dataset preparation**.
Maxwell uses the same setup and runner through a small Slurm wrapper.

## Clone, install, and run

On Linux or macOS, install Python 3.11+, the stable Rust toolchain, Git, and a
C++17 compiler with OpenMP. Linux distributions normally provide the latter in
GCC (`g++`). On macOS, use `brew install gcc`; setup selects Homebrew's versioned
compiler automatically. Set `CC` and `CXX` to override that selection. On Windows,
use WSL2 for this benchmark tooling.

```bash
git clone https://github.com/CPrescher/Dioptrin.git
cd Dioptrin
python3 benchmarks/comparison/setup.py
.venv-benchmark/bin/python benchmarks/comparison/compare.py
```

Setup creates `.venv-benchmark`, builds the current Dioptrin source in release
mode, builds the pinned upstream azint source, and installs pyFAI. Both source
builds target the machine's CPU. Installation needs internet access; do it once
per machine/checkout, and rerun setup after changing the Dioptrin source. No
Dioptrin runtime dependencies are changed. Setup records wheels, compiler flags,
Rust versions, Cargo.lock, and resolved Python packages in the environment's
`build-record/` directory. Don't reuse native wheels on a different CPU model.
CI uses `setup.py --portable` to omit native CPU flags, making shared Cargo
caches safe across runner CPU models. This mode checks functionality rather
than native CPU performance.

### azint build flags

By default azint is compiled with `-O3` plus the native CPU flag
(`-mcpu=native` on ARM, `-march=native` on x86), without `-ffast-math`.
Dioptrin, by contrast, runs with its default `fast_math=True`, which lets it
reassociate its CSR accumulation; azint's `_spmv` is a sequential double
reduction that the compiler only reorders with fast-math. For a like-for-like
build use `--azint-fast-math`. It appends `-ffast-math` to azint's `CFLAGS` and
`CXXFLAGS` and, only if the compiler accepts it (clang), `-flto=thin` to
`CFLAGS`/`CXXFLAGS`/`LDFLAGS`. This follows the "CPU/LTO/fast-math" azint
build of the 2026-10-24 comparison. GCC rejects `-flto=thin`; azint is one
translation unit, so no other LTO flag is substituted and the record says so.
The flags apply to azint only, not to Dioptrin or pyFAI.

`--azint-only` rebuilds just azint, at the same pinned commit, in an existing
environment. It does not upgrade pip, reinstall requirements, or rebuild
Dioptrin, and it keeps the existing Dioptrin build record:

```bash
python3 benchmarks/comparison/setup.py --azint-only --azint-fast-math  # fast-math azint
python3 benchmarks/comparison/setup.py --azint-only                    # back to the default build
```

`build.json` (and `<venv>/benchmark-build.json`) records an `azint_build` entry:
variant (`default` or `fast-math`), the exact `CC`/`CXX`/`CFLAGS`/`CXXFLAGS`/`LDFLAGS`,
the LTO decision, the wheel and installed extension SHA-256, and a check that
importing the extension did not turn on flush-to-zero. Each flavour's wheel is
kept under `build-record/wheels/azint-<variant>/`. `compare.py` copies this entry
into `environment.json` (`azint_build`), prints the azint flags in `summary.md`,
and treats a loaded `_azint` extension that differs from the recorded one like
a stale Dioptrin build (rejected unless `--allow-unverified-build`). Records
written before this entry existed show `variant: unrecorded` and their shared
flags under `legacy_flags`.

The runner rejects stale or missing Dioptrin build records by default. It checks
Rust source, Cargo/Python build configuration, and the installed module's bytes,
and records actual module paths. Rerun setup after changing source. For deliberate
comparisons using a separately installed binary, `--allow-unverified-build`
overrides the check and records a visible warning. Both parent and workers ignore
`PYTHONPATH` and user site-packages to keep imports on the selected interpreter.

Results go into a new, ignored `benchmark-results/<timestamp>/` directory:
`summary.md`, `results.csv`, `numerical.json`, raw timing blocks, per-case logs
and output arrays, hardware/thread settings, source revision, and input hashes.
The standard run time depends on the CPU and setup costs; use the quick suite
first to verify installation.

```bash
.venv-benchmark/bin/python benchmarks/comparison/compare.py --suite quick
.venv-benchmark/bin/python benchmarks/comparison/compare.py --workers 1,2,4
.venv-benchmark/bin/python benchmarks/comparison/compare.py --output /tmp/my-new-results
```

The runner works from any current working directory. Use its absolute path if
you aren't in the clone. Outputs must be new directories. Default thread budgets
are restricted to the process's CPU affinity; explicit `--workers` values outside
that allocation are rejected.

## Suites and data

Dataset definitions and timing defaults live in [datasets.json](datasets.json).

| Suite | Inputs | Default CPU budgets |
| --- | --- | --- |
| `quick` | Four distinct generated 96 x 128 images | 1, 2 |
| `standard` (default) | Bundled real Pilatus1M image; four distinct generated 2048 x 2048 and 3072 x 3072 images | 1, 2, 4, 8 |
| `extended` | Standard inputs plus four generated 4096 x 4096 images; longer timing and two trials | 1, 2, 4, 8 |

Synthetic data include narrow powder rings, an amorphous halo, background,
Poisson fluctuations, module gaps, a beamstop, a tilted detector, and known
polarization/solid-angle corrections. Their masks, geometry, seeds, generator
version, and frame hashes are recorded. NumPy PCG64 seeds are fixed; changing
NumPy's sampling implementation can change bytes, which the recorded hashes
make visible. These are benchmark fixtures, not a full detector/acquisition
simulation or an independent scientific oracle for the generator's geometry.

Two workloads are reported separately: `repeated` reuses the first image;
`frames` cycles through the four distinct generated images. Both are preloaded
in-memory integration measurements. The real Pilatus fixture supplies one
measured image, so it only has a repeated-image row. Synthetic frame generation,
file decoding, and writing outputs are outside all timing blocks. The distinct
series increases the working set and exercises correct per-frame ordering; it
is still not a disk/acquisition throughput benchmark.

The standard suite reuses `test_data/CeO2_Pilatus1M.tif` and `v1.poni`, already
part of this repository. If Git LFS leaves only a pointer, or the image is absent,
the runner automatically downloads the same image from a pinned public revision,
checks its SHA-256, and caches it under `~/.cache/dioptrin-benchmarks/` (or
`$XDG_CACHE_HOME/dioptrin-benchmarks/`). Corrupt downloads are rejected. It does
not modify the clone's test data. Once cached, later runs need no data download.

```bash
# Fully offline once dependencies and the bundled/cached Pilatus image exist:
.venv-benchmark/bin/python benchmarks/comparison/compare.py --offline
# Generated data only, requiring no measured image or network access:
.venv-benchmark/bin/python benchmarks/comparison/compare.py --datasets synthetic-4m synthetic-9m --offline
# List all definitions without loading benchmark libraries:
python3 benchmarks/comparison/compare.py --list-datasets
```

`--datasets`, `--workloads repeated frames`, `--bins`, `--frames`, `--trials`,
`--blocks`, and `--seconds` override suite defaults. `--cache-dir` changes the
verified-download cache. Larger frame stacks are temporary files within the
run directory; they are removed after a successful run by default, while their
hashes and specifications remain. `--keep-inputs` retains them for investigation.
A failed run preserves both inputs and completed results.

The optional `eiger` dataset supports the earlier local Eiger TIFF/PONI/mask
through `--datasets eiger --data-dir /path/to/test_data`; no suite selects it.
It is not included or downloaded because that private dataset has no published
benchmark distribution. Add future public datasets through a versioned manifest
entry with immutable source URL and checksum, after establishing its source
and redistribution terms. Large experimental data do not belong in Git history.

### Archivable Eiger subsets

Measured-data rows for the paper run from small subsets that can be archived
publicly, not from the full scans. `prepare_subset.py` copies N evenly spaced
frames (default 32, first and last included) of a Bitshuffle/LZ4 HDF5 scan as
their **original compressed chunks** (`read_direct_chunk`/`write_direct_chunk`,
same dataset path, dtype, one frame per chunk, same Bitshuffle block size and
LZ4; nothing is recompressed). It records the scan's SHA-256, the source frame
indices, the dataset attributes, the filter settings and the SHA-256 of every
decoded frame inside the file (`/benchmark_subset`), decodes every copied frame
again and compares it with the decoded scan frame. The HDF5 file holds no
timestamp: the same scan and hdf5plugin version give identical bytes. Beside it
go the PONI, the mask and a JSON manifest, so the directory is a self-contained
bundle:

```bash
OUT=/path/outside/the/clone/subsets   # do not put scan data into Git
.venv-benchmark/bin/python benchmarks/comparison/prepare_subset.py subset \
  --scan /data/eiger9m/eiger_0000.h5 --dataset entry_0000/instrument/eiger/data \
  --poni /data/eiger9m/eiger_0000.poni --mask /data/eiger9m/eiger_0000.mask \
  --name eiger9m --out-dir "$OUT"
.venv-benchmark/bin/python benchmarks/comparison/prepare_subset.py subset \
  --scan /data/diopside/scan0001/eiger_0000.h5 --dataset /entry_0000/ESRF-ID11/eiger/data \
  --poni /data/diopside/CeO2_tmp.poni --catalog-detector Eiger2_4M \
  --name eiger4m --out-dir "$OUT"
# Re-check later, with or without the original scan:
.venv-benchmark/bin/python benchmarks/comparison/prepare_subset.py verify \
  --subset "$OUT/eiger9m-subset32.h5" [--scan /data/eiger9m/eiger_0000.h5]
```

The mask file is written in the orientation of the HDF5 frames. A supplied mask
may be stored in another orientation (the Eiger 9M mask is flipped vertically
relative to the HDF5 frames); `--mask-transform auto` chooses the flip under
which it covers at least 99.9% of the subset's invalid pixels (value >= 2^31 - 1)
while every other flip covers less than 99%, and refuses otherwise. Without
`--mask`, a mask is generated: pixels that are 0 in every subset frame (module
gaps and dead pixels) or invalid in any subset frame. `--catalog-detector`
records how much of a pyFAI catalogue detector's gap mask it covers.

`tile` builds a scan-sized file from a subset only, again from its compressed
chunks (frame i is subset frame i mod N), for file-reading pipelines:

```bash
.venv-benchmark/bin/python benchmarks/comparison/prepare_subset.py tile \
  --subset "$OUT/eiger9m-subset32.h5" --frames 625 --output "$OUT/eiger9m-tiled625.h5"
```

The `eiger9m-subset` and `eiger4m-subset` datasets read such bundles from
`--subset-dir` (default `--data-dir`); no suite selects them, so the standard
suite still needs no private data. The loader checks that the subset was cut
from the scan named in `datasets.json` (`source_sha256`) and that every decoded
frame matches its recorded hash. The mask is the union of the bundle's mask, the
calibration detector's mask and pixels invalid in any subset frame. With the
suite's 32 frames each `frames` pass integrates every subset frame once:

```bash
.venv-benchmark/bin/python benchmarks/comparison/compare.py \
  --datasets eiger9m-subset eiger4m-subset --subset-dir "$OUT" --workloads frames
```

Each frame is decoded to float64 in RAM before timing (32 Eiger 9M frames:
2.6 GB); decoding stays outside every timing block.

## What is measured and checked

All methods use the same float64 input arrays, masks, physical geometry,
2theta bounds, bin count, solid-angle correction, and polarization 0.965. The
standard suite uses 2,000 bins and reports intensity-only and Poisson modes.

Methods are Dioptrin `pixel_split`, `supersampled`, and `pyfai_full`; azint
subpixel integration in two rows, `azint` and `azint-cached`; pyFAI full CSR and
bbox CSR using Cython on Python frame threads (`pyfai-full`, `pyfai-bbox`);
`pyfai-full-omp`, pyFAI's recommended CPU mode, one full CSR Cython engine
integrating frames in sequence with `OMP_NUM_THREADS` = CPU budget; and
`pyfai-full-opencl`, method `("full", "csr", "opencl")` on one OpenCL device. Supersampling is 15 x 15 for Pilatus, 5 x 5 for the larger
generated detectors, and 3 x 3 for quick.
The runner requires pyFAI to use the requested method and rejects fallbacks.

`pyfai-full-omp` runs only where pyFAI's CSR extension contains OpenMP. pyFAI
has no runtime flag for this, so a probe looks for the OpenMP runtime entry
points (`GOMP_*`, `__kmpc_*`) in `pyFAI/ext/splitPixelFullCSR` and, on Linux,
for the runtime in the process maps. The macOS wheels have none; there the
variant is skipped with a `SKIPPED` line in the log and `summary.md` and an entry
in `skipped.json`, instead of reporting a number that OpenMP settings cannot
affect. Linux wheels link GCC's OpenMP.

`pyfai-full-opencl` needs `pyopencl` (`setup.py --with-opencl` installs a binary
wheel; no source build) and a device; otherwise it is skipped the same way.
`--opencl-device` selects `gpu` (default), `cpu`, `acc`, `any` or explicit
`PLATFORM:DEVICE` ids. `environment.json` (`variant_probes.opencl`) and each
case record the platform, device, versions and whether the device has fp64.
pyFAI's OpenCL CSR kernels compute in single precision whatever the device:
float32 input, coefficients and arithmetic, with double-word (float2,
compensated) accumulators (`single_precision_accumulation: true`). The frames
are converted from the shared float64 inputs to float32 on the host inside the
timed call. It runs at a CPU budget of 1 in one process (one host thread, one
engine); rows for higher budgets or process counts are not scheduled.

The two azint rows use the same build and sparse matrix:

- `azint` is azint as documented: `AzimuthalIntegrator(..., mask=mask)` and
  `integrate(frame)`. Because `mask` is set, every call multiplies the frame by
  the inverted mask and recomputes the normalisation with an extra SpMV,
  although masked pixels are already excluded from the matrix.
- `azint-cached` sets the public attribute `ai.mask = None` after construction,
  so `integrate()` reuses the normalisation the constructor cached
  (`norm = A @ 1`). With masked pixels absent from `A` and zeroed in every
  input, this is the same arithmetic in the same order. After its warm-up the
  case integrates every distinct frame with a stock masked copy and fails
  unless intensity and Poisson sigma are bit-identical; the parent also
  compares each `azint-cached` output with the matching `azint` case and writes
  `azint_cached_identity.json`. Both rows are in the default variant list.

These algorithms, internal precision, and correction estimators differ.
Cross-library differences are reported without asserting identical answers.
Azint averages corrected pixels, while Dioptrin/pyFAI normalize weighted raw
counts. pyFAI can slightly expand the upper bin bound; axis offsets are reported.

The CPU-budget column is the total global Rayon pool size for Dioptrin's native
array batch API. For competitors it is the number of persistent Python frame
threads. Thread settings per case:

| Variant | Frame threads | `OMP_NUM_THREADS` | `RAYON_NUM_THREADS` |
| --- | ---: | ---: | ---: |
| `dioptrin-*` | (Rayon pool) | 1 | budget |
| `azint`, `azint-cached` | budget | budget | 1 |
| `pyfai-full`, `pyfai-bbox` | budget | 1 | 1 |
| `pyfai-full-omp` | 1 | budget | 1 |
| `pyfai-full-opencl` | 1 | 1 | 1 |

`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` are 1, and
`OMP_DYNAMIC`/`OMP_PROC_BIND` are `FALSE` everywhere. In the pinned azint source
the only OpenMP region is the sparse-matrix construction (`generate_matrix`);
`spmv`, `set_correction`, and the Python wrapper (NumPy only) have none. So
`OMP_NUM_THREADS = budget` only parallelises azint set-up, making its "ready"
time comparable with Dioptrin's budget-sized Rayon pool. The static OpenMP
schedule assigns contiguous detector rows in thread order, so the matrix,
and therefore the output, is identical for any OpenMP thread count. pyFAI
stays at 1 in its Python-thread rows: its macOS wheel has no OpenMP, and on
Linux more threads would change what those rows measure; `pyfai-full-omp` is
the OpenMP row. Each case's settings are in `cases/*.json`
(`threads`), its result JSON (`thread_environment`), `results.csv`
(`omp_num_threads`, `rayon_num_threads`), and `environment.json` (`case_threads`).

Azint shares its sparse matrix through shallow copies: geometry, mask, sparse
arrays, and (for `azint-cached`) the cached `norm` are shared and only read;
the per-call `norm_1d`/`norm_2d` attributes are assigned on each copy, and
scratch arrays are newly allocated per call. Its pinned native SpMV releases the GIL.
PyFAI owns an initialized CSR engine
per worker. Native batch scheduling and result allocation are timed. Setup,
first-result and all-workers-ready times, and process-lifetime peak RSS are
separate columns. Setup includes backend import and lazy initialization.
Frame workers receive contiguous slices of the logical input sequence, so the
distinct-frame workload does not pin a worker to one image at 4/8 CPUs. Batch
lengths are rounded up to a multiple of the CPU budget for balanced slices.
The fixed pyFAI mask is installed on its detector once, using its cached mask
checksum rather than recalculating the checksum from a mask argument each call.

### Several processes per case

`--processes 1,2,4` runs every library additionally as P cooperating processes
for each CPU budget divisible by P (budget 1 and P = 1 are always included).
Each process gets budget/P workers (Rayon threads, Python frame threads, or
OpenMP threads for `pyfai-full-omp`), a contiguous 1/P share of the logical
frame sequence, copies of only its own distinct frames, allocated in that
process, and its own tables. The parent releases all processes at a common
start time (100 ms after the command, spin-waited); each integrates the same
number of passes over its frames and reports its finish time. Throughput is
all frames divided by the wall time from the common start to the last process
finishing. Two untimed calibration blocks set the passes per block so that a
block lasts about `--seconds`. The result records passes per block, the start
lag, per-process peak RSS (and their sum as `peak_rss_mib`), affinity and the
command prefix of each rank; `results.csv` has the added columns `processes`,
`workers_per_process` and `binding`. Every process validates its outputs
against the one-process, one-worker reference before and after timing, and
`azint-cached` repeats its stock bit-identity check in each process.

`--bind numa` starts rank r under `numactl --cpunodebind=N --membind=N`,
round-robin over the NUMA nodes in the allocation (frames and tables are then
first touched on that node). It needs Linux and `numactl`; `--bind none`
(default) works everywhere, including macOS. This replaces the earlier
Dioptrin-only socket-local experiment: every library gets the same layouts.
P = 1 rows keep the single-process measurement described above.

### Masking the beam pixel

`--mask-beam-pixel` adds, for all libraries, the pixel(s) whose area contains or
touches the point of normal incidence (the PONI, default) to the mask and zeroes
them in every frame. `--mask-beam-pixel direct` uses the direct-beam position
(2theta = 0, pyFAI's Fit2D centre) instead; with detector tilts the two differ
by several pixels. `fixture.json` (`beam_pixel_mask`), the log and `summary.md`
record the point, the pixels, which were already masked, and the minimum-2theta
pixel; `results.csv` adds `mask_beam_pixel` and `beam_pixels_added`. Compare
`negative_sigma_bins` with and without the option. For the Eiger 9M subset,
pyFAI full CSR (Cython) returns negative sigma in the second bin of every frame;
masking the PONI pixel leaves it, masking the direct-beam pixel removes it.

Every case runs in a fresh process. Method order alternates between trials.
Every output frame, including its order, is checked against the same backend's
one-worker reference before and after timing. One-worker references are always
included even if only higher worker counts are requested. Numerical quality
and self-consistency are separate fields: finite negative Poisson uncertainties
are retained and flagged; nonfinite outputs, missing results, and disagreement
with self-references fail the case. Raw per-frame outputs remain available.

Core comparison versions are in [requirements.txt](requirements.txt); azint's
commit is pinned in `setup.py`. Resolved transitive dependencies and compiler
versions are recorded. Rust dependencies resolve on the first setup of a fresh
clone because the library does not track Cargo.lock; Python build/transitive
dependencies also float within their constraints. For an exact environment
replay, reuse the saved Cargo.lock and Python freeze alongside the same source
and compiler. The default recipe fixes core versions and inputs, not every
transitive dependency across future dates. This suite measures in-memory 1D integration
(CPU, plus pyFAI's OpenCL CSR where available); it does not assess 2D caking,
decompression, or file-reading performance.

## Maxwell Slurm wrapper

From a terminal on Maxwell, clone the repository and submit from its root:

```bash
git clone https://github.com/CPrescher/Dioptrin.git
cd Dioptrin
/software/tools/bin/my-partitions
sbatch benchmarks/comparison/maxwell.sbatch
```

The job loads `maxwell mamba` and looks for a compatible **persistent Python
environment**. On a cache miss it installs a minimal Rust toolchain in node-local
temporary storage and invokes the same `setup.py`. On a cache hit it skips Rust
installation, pip installation, and compilation, then runs `compare.py --suite
standard` using the cached interpreter. All compilation and timed work still run
on the allocated compute node.

By default, environments live under `.benchmark-environments/` in the clone,
on shared storage. The cache key includes CPU model/instruction flags, OS/libc,
Python version/ABI/path, compiler versions/flags, requested Rust toolchain,
source/build inputs, and setup/dependency recipes. The same CPU class can reuse
an environment across nodes; a different CPU class or changed build source gets
a separate build. Benchmark timing options, input datasets, and README changes
do not rebuild the environment. The cache checks imports, installed package
versions/dependency consistency, and the recorded Dioptrin source/binary before
reuse. A broken environment is replaced by a fresh generation.

Concurrent jobs validate warm environments independently. Only construction of
the same cache entry is serialized, using a POSIX file lock on shared storage;
waiting jobs log the lock location. Validation commands have 60-second timeouts
and report why an environment needs rebuilding. Completed generations
are never moved or modified by the cache manager. Forced/failed rebuilds leave
previous generations available to jobs already using them; failed builds are
never published as ready and are removed after a caught setup failure. Don't run `pip install` into a managed cached env;
change the dependency recipe or request a rebuild instead.

Builds inherit an allowlist of compiler flags, library paths, locale, and network
settings. Ambient Cargo, maturin, PyO3, Python, and pip overrides are excluded;
pip configuration files are disabled. Compiler flags/library paths and ancestor
Cargo configuration files are included in the cache key. Each job's interpreter
ignores Python environment overrides and user site-packages.

Each run writes `environment.json` alongside its job log, including the exact
interpreter, fingerprint, and whether it was reused. Build metadata is copied
into `benchmark-results/maxwell/<job-id>-.../build/`; native wheels remain in the
cached generation. Node-local Rust tools/targets are cleaned on success, but
the Python environment remains for later jobs. First builds need outbound HTTPS
to GitHub, PyPI, and Rust distribution servers; a compatible cached environment
needs no installation downloads. The real-input download cache is separate.

Use a persistent shared directory (for example, your assigned DUST space) to
place the environment cache elsewhere, or request a fresh generation:

```bash
export ENV_CACHE_ROOT=/your/shared/storage/dioptrin-environments
sbatch benchmarks/comparison/maxwell.sbatch

# Force a fresh environment without deleting one used by running jobs:
REBUILD_ENV=1 sbatch benchmarks/comparison/maxwell.sbatch
```

The default `stable` Rust selector and floating transitive dependencies do not
trigger daily rebuilds: existing validated environments keep their recorded
versions until build inputs change or `REBUILD_ENV=1` is requested. Multiple CPU
classes/source revisions/forced generations use additional storage. Simultaneous
forced requests share a rebuild completed while they waited; a later forced
request builds again. The generated Cargo.lock is deliberately excluded from
the key: after supplying a saved lock for an exact replay, use `REBUILD_ENV=1`.
README-only commits can reuse a build recorded under an earlier commit because
the build content hashes are unchanged.

Old cache generations can be removed manually after all jobs using them have
finished. `ready.json` identifies the current generation; `published.json` marks
generations that may have been used by jobs. A killed build can leave an
unpublished directory, which can be removed once no build is active. Do not
delete published generations while their jobs are running. Shared-filesystem
locking and cross-node reuse still need qualification on Maxwell itself.

Slurm requests one exclusive CPU node, its memory, two hours, and `allcpu` by
default. Override the partition with one your account can use, and pass runner
options after the script name:

```bash
sbatch --partition=upex benchmarks/comparison/maxwell.sbatch --suite quick
sbatch benchmarks/comparison/maxwell.sbatch --workers 1,2,4,8,16 --trials 2
squeue -u "$USER"
tail -f dioptrin-JOBID.out
```

The wrapper builds azint with `--azint-fast-math` by default (`AZINT_FAST_MATH=1`;
set `AZINT_FAST_MATH=0` for the default azint build). Copy the subset bundles to
Maxwell storage (they are small) and pass `--subset-dir`. A CPU-node run of
both subsets with Python threads, OpenMP pyFAI, and 1/2/4 processes, with and
without the beam pixel:

```bash
SUBSETS=/your/storage/subsets
sbatch benchmarks/comparison/maxwell.sbatch --datasets eiger9m-subset eiger4m-subset \
  --subset-dir "$SUBSETS" --workloads frames --workers 1,2,4,8,12,24,48,96 \
  --processes 1,2,4 --bind numa \
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full \
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp
sbatch benchmarks/comparison/maxwell.sbatch --datasets eiger9m-subset eiger4m-subset \
  --subset-dir "$SUBSETS" --workloads frames --workers 1,2,4,8,12,24,48,96 \
  --processes 1,2,4 --bind numa --mask-beam-pixel direct \
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full \
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp
```

A GPU node needs `WITH_OPENCL=1` (pyopencl in a separate cached environment),
your GPU partition, and any site modules the OpenCL driver needs in
`EXTRA_MODULES`; the wrapper saves `nvidia-smi` and `/etc/OpenCL/vendors`:

```bash
WITH_OPENCL=1 EXTRA_MODULES="<modules, if needed>" sbatch --partition=<gpu-partition> \
  benchmarks/comparison/maxwell.sbatch --datasets eiger9m-subset eiger4m-subset \
  --subset-dir "$SUBSETS" --workloads frames --workers 1 \
  --variants dioptrin-pyfai_full pyfai-full pyfai-full-opencl --opencl-device gpu
```

`allcpu` can be preempted/requeued; every restart writes a new results directory.
Higher worker counts can require substantial memory for pyFAI full CSR. Optional
wrapper environment settings are `BUILD_MODULES`, `EXTRA_MODULES`, `PYTHON_BIN`,
`BUILD_JOBS` (default 8), `RUST_TOOLCHAIN` (stable), `RESULT_ROOT`, `SUITE`
(standard), `CC`, `CXX`, `ENV_CACHE_ROOT`, `REBUILD_ENV=1`, `KEEP_BUILD=1`
(node-local build tools), `AZINT_FAST_MATH` (default 1) and `WITH_OPENCL` (default 0).
Both build switches are part of the environment cache key.

References: [Maxwell batch jobs](https://docs.desy.de/maxwell/documentation/jobs/running_batch_jobs/),
[Python modules](https://docs.desy.de/maxwell/development/python/),
[azint](https://maxiv-science.github.io/azint/).
