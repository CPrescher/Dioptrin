# Reproducible CPU integration benchmarks

Compare Dioptrin, azint, and pyFAI from a normal Git clone. The **standard suite
requires no private files, manual data copying, or dataset preparation**.
Maxwell uses the same setup and runner through a small Slurm wrapper.

## Clone, install, and run

On Linux or macOS, install Python 3.11+, the stable Rust toolchain, Git, and a
C++17 compiler with OpenMP. Linux distributions normally provide the latter in
GCC (`g++`). On macOS, use `brew install gcc`; setup selects Homebrew's versioned
compiler automatically. Set `CC` and `CXX` to override that selection. On Windows,
use WSL2 for `compare.py`. `setup.py` also builds a native Windows environment
(for the GPU runner below) from a Visual Studio Developer PowerShell with the
MSVC C++ build tools and the Rust MSVC toolchain.

```bash
git clone https://github.com/CPrescher/Dioptrin.git
cd Dioptrin
python3 benchmarks/comparison/setup.py
.venv-benchmark/bin/python benchmarks/comparison/compare.py
```

On native Windows, run `python benchmarks/comparison/setup.py`, then use
`.venv-benchmark\Scripts\python.exe -E -s -B` in place of the interpreter for
the GPU runner. On Windows x64, setup pairs Dioptrin's `x86-64-v3` with
AZINT's MSVC `/arch:AVX2`; `--azint-fast-math` is not available with MSVC.
The official pyFAI wheel keeps its upstream flags.

Setup creates `.venv-benchmark`, builds the current Dioptrin source in release
mode, builds the pinned upstream azint source, and installs pyFAI. Linux and
macOS source builds use native CPU flags. Installation needs internet access; do it once
per machine/checkout, and rerun setup after changing the Dioptrin source. No
Dioptrin runtime dependencies are changed. Setup records wheels, compiler flags,
Rust versions, Cargo.lock, and resolved Python packages in the environment's
`build-record/` directory. Don't reuse native wheels on a different CPU model.
`setup.py --with-cuda` builds Dioptrin with its `cuda` feature for the
CUDA rows (see "Dioptrin CUDA"), and `setup.py --with-metal` builds with the
`metal` feature on macOS for the `dioptrin-*-metal` rows; `build.json` records
`cuda`, `metal`, and `cargo_features`. Both CUDA and Metal rows support 1D and
2D cakes (`--dimensions 1d 2d`), `--gpu-precision {f32_compensated,f32,f64}`,
and `--input-dtype {float64,float32}`. For the unified Windows / Maxwell /
macOS paper replay commands, see
[docs/benchmarks/2026-11-24-paper-final/README.md](../../docs/benchmarks/2026-11-24-paper-final/README.md).
For the Dioptrin 0.16.0 rerun, one run script per machine is in
[paper-0.16.0/](paper-0.16.0/), with the protocol in
[docs/benchmarks/2026-11-25-paper-0.16.0/README.md](../../docs/benchmarks/2026-11-25-paper-0.16.0/README.md).
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
CPU runs save machine details in `environment.json` and GPU runs save them in
`gpu-results.json`. Capture covers the OS, CPU model, physical and logical cores,
RAM, and, where available, CPU affinity, Apple performance and efficiency cores,
GPU inventory and driver, and relevant Slurm allocation settings.
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
`pyfai-full-opencl`, method `("full", "csr", "opencl")` on one OpenCL device;
and, opt-in (`--variants`), Dioptrin's CUDA plans in four rows,
`dioptrin-pixel_split-cuda`, `dioptrin-pyfai_full-cuda`,
`dioptrin-pixel_split-cuda-batch` and `dioptrin-pyfai_full-cuda-batch`, in 1D
and 2D (see "Dioptrin CUDA" below). Supersampling is 15 x 15 for Pilatus, 5 x 5 for the larger
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

### Dioptrin CUDA

The four CUDA rows run Dioptrin's experimental CUDA plans
(`Integrator.prepare_cuda1d`, and `prepare_cuda2d` for 2D cases) under the same
protocol as `pyfai-full-opencl`: the same float64 fixture frames, passed
unchanged, the same Integrator settings as the CPU Dioptrin rows (method, mask,
2theta range, solid angle, polarization 0.965), a CPU budget of 1 (one host
thread, `RAYON_NUM_THREADS=1`) and one process. With `--dimensions 2d` they also
get 2D cases (at budget 1 and one process) and take part in the 2D preflight;
the 2D timed call is `plan.integrate2d(frame)` per frame or `plan.batch2d(frames)`,
and the untimed check compares with CPU `Integrator.integrate2d` (and, for the
batch rows, `batch2d` with per-frame `integrate2d`). The 1D description below
applies to 2D with these calls. They are not in the default variant list: name
them with `--variants`, so a default run never initialises a CUDA device or
NVRTC.

| Variant | Method | Timed call |
| --- | --- | --- |
| `dioptrin-pixel_split-cuda` | `pixel_split` | `plan.integrate1d(frame)` once per frame |
| `dioptrin-pyfai_full-cuda` | `pyfai_full` | `plan.integrate1d(frame)` once per frame |
| `dioptrin-pixel_split-cuda-batch` | `pixel_split` | `plan.batch1d(frames)` over the whole batch |
| `dioptrin-pyfai_full-cuda-batch` | `pyfai_full` | `plan.batch1d(frames)` over the whole batch |

The `-cuda` rows make one public call per frame, with one transfer each way and
one synchronisation per frame, as pyFAI's OpenCL `integrate1d` does; they are
the like-for-like rows for comparison with `pyfai-full-opencl`. The
`-cuda-batch` rows use Dioptrin's multi-frame call: frames are processed in
chunks of up to `max_batch_frames` (at most 256 per kernel launch). Both are
reported.

**Set-up (not timed).** The case process builds the Integrator, sets the mask
and calls `prepare_cuda1d(shape, bins, precision="f32_compensated",
device=--cuda-device)`. This initialises the device (context, NVRTC compile of
the kernels, module load), builds the sparse table on the CPU and uploads it
once. `setup_first_s` runs to the end of the first integrated frame, as for
pyFAI's first OpenCL call. The untimed warm-up pass sizes the plan's device
scratch, so no device memory is allocated inside timing blocks.

**Timed region.** Every timed call copies each float64 frame from host memory
to the device (8 bytes per pixel), converts it to float32 there, integrates,
copies the pattern back and synchronises before returning. Like every Dioptrin
row, it then ends with NumPy float64 intensity (and Poisson sigma) for every
frame, created inside the timed call (see "Result conversion" below); pyFAI's
timed call returns NumPy arrays. Transfers use ordinary (pageable) NumPy
memory, as pyFAI's do.

**Precision.** All CUDA rows use `f32_compensated`: float32 pixels and weights,
sums accumulated as double-word (hi, lo) float32 pairs, and denominators,
division and output in float64. pyFAI's OpenCL CSR engine also works on float32
input and coefficients with double-word (float2) compensated accumulators, so
this is the counterpart setting. Differences that remain:

- Dioptrin computes the normalisation denominators (solid angle,
  polarization, pixel weights) once, in float64 on the CPU, when the plan is
  prepared and divides in float64 on the device. pyFAI, as far as we know,
  accumulates the normalisation on the device for every frame and divides in
  float32.
- pyFAI converts the float64 frame to float32 on the host inside its call;
  Dioptrin uploads float64 and converts on the device. Both conversions are
  inside each library's timed call. There is no input-dtype option.
- The GPU reduction order is fixed and differs from the CPU's, so CUDA results
  agree with CPU Dioptrin within tolerance, not bit for bit. `fast_math` and
  `sparse_layout` do not affect the GPU path.

**Checks.** Besides the usual self-validation against the one-worker reference,
each CUDA case, after set-up and outside timing, compares every distinct frame
with CPU `Integrator.integrate1d` of the same Integrator: the radial axis to
1e-12 relative (whether it is exactly equal is recorded), intensity and sigma
with `rtol=1e-5, atol=1e-5 * max|cpu|`. Larger differences fail the case. The
`-cuda-batch` cases also require `batch1d` to be bit-identical to per-frame
`integrate1d`. This check runs CPU integrations while the device idles, so one
more untimed pass of the case's frames follows it before the first timed block
(`setup_ready_s` is unaffected). Each result JSON records, under `cuda`, the call, the precision,
device ordinal and name, compute capability, total memory, driver, NVRTC and
binding versions, `max_batch_frames`, `table_entries`, the input dtype, the
CPU agreement (`max_abs` and `max_rel_nonzero` for intensity and sigma) and the
batch identity. In `numerical.json` (entry kinds under "Numerical entries"
below), `dioptrin-pyfai_full-cuda` and `-cuda-batch` get counterpart entries
against `dioptrin-pyfai_full`; the counterpart of the `dioptrin-pixel_split-cuda`
rows is `dioptrin-pixel_split`, the reference itself, so their reference entry
is marked `"also_counterpart": true` instead of being repeated.

**Build and run-time requirements.** The rows need a Dioptrin built with the
`cuda` Cargo feature: `setup.py --with-cuda` (maturin `--features python,cuda`;
`environment.py --with-cuda`, `WITH_CUDA=1` for the Slurm wrapper). The build
needs only the Rust toolchain: no CUDA toolkit, nvcc or driver. At run time the
node needs an NVIDIA driver and NVRTC (`libnvrtc` with `libnvrtc-builtins`)
on the dynamic loader's path. NVRTC must be no newer than the CUDA version the
driver supports (`nvidia-smi`, "CUDA Version"), or the driver cannot load the
compiled kernels, and it must support the GPU's compute capability (CUDA 13
needs 7.5 or later). On a cluster, `module load` a CUDA toolkit no newer than
the driver before running. Do not set `CUDARC_CUDA_VERSION` when building.
`--cuda-device` (default 0, counted after `CUDA_VISIBLE_DEVICES`) selects the
device. When a CUDA row is requested, a probe in its own child process (separate
from the pyFAI probe, with a 600 s timeout that covers the NVRTC compilation)
records `dioptrin.cuda_info()` in `environment.json` (`variant_probes.cuda`),
and `summary.md` prints the device, driver, NVRTC version and precision. Without
the feature, with a Dioptrin that predates the CUDA API, without a usable
device, or when the probe child exits non-zero, is killed by a signal or times
out (`probe_failed`, with the exit status in the reason), the CUDA rows are
skipped with a `SKIPPED` line and an entry in `skipped.json`, like the OpenCL
row; every other variant still runs. With CUDA rows in a run, `summary.md` is
headed "CPU and GPU integration benchmark" and names the CUDA and OpenCL rows;
runs without CUDA rows keep the earlier wording.

A `WITH_CUDA=1` (`--with-cuda`) environment is a different build with its own
cache key. Its CPU Dioptrin rows run on that cuda-feature wheel; the CPU code
is the same.

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

**Numerical entries.** 1D entries in `numerical.json` come in two kinds, both
for one-worker, trial-0 rows and matched by dataset, workload and error mode.
Reference entries (no `comparison` key) compare every row with
`dioptrin-pixel_split`, as before. Counterpart entries
(`"comparison": "counterpart"`) follow them and compare a GPU row with its CPU
method: `pyfai-full-opencl` with `pyfai-full`, and `dioptrin-pyfai_full-cuda`
and `dioptrin-pyfai_full-cuda-batch` with `dioptrin-pyfai_full`. The
`dioptrin-pixel_split-cuda` rows' counterpart is the reference itself; their
reference entry carries `"also_counterpart": true` and no second entry is
written. A counterpart entry is left out when the counterpart row did not run.
2D entries are described under "2D cakes".

**Dioptrin CPU 1D call.** `--dioptrin-1d-path` chooses what the CPU Dioptrin
1D rows time (several values give one case each):

- `global` (default): `Integrator.batch1d(frames, n, errors=...)` over the
  list of arrays, with neither `num_workers` nor `num_threads`: the library's
  default call. All frames run on the global Rayon pool, whose size the case
  sets to the CPU budget (`RAYON_NUM_THREADS`) and checks
  (`dioptrin.rayon_num_threads()`, the case fails otherwise). This call uses
  Dioptrin's 4-frame tiled kernels when it has at least 4 frames per pool
  thread (frames per call >= 4 x budget; the standard suite's 32 frames
  reach that up to a budget of 8). Below that it integrates frame by frame,
  so a scaling curve with a fixed frame count can switch kernels between
  budgets (for example 32 frames: tiled up to 8 threads, per frame at 16);
  raise `--frames` to at least 4 x the largest budget to keep one kernel.
  With `--processes P` each process makes its own call with frames/P frames
  on budget/P threads, which is the same condition.
- `split`: `batch1d(frames, n, errors=..., num_workers=budget, num_threads=1)`.
  Since Dioptrin 0.15.0 this runs `budget` frames at a time, each on a
  one-thread pool that the call builds (and drops) itself, without the tiled
  kernels; results are bit-identical. In 0.14 the same arguments were ignored
  for arrays and the call ran as `global`, so earlier results with these
  arguments are `global` measurements.

The path is recorded in the `dioptrin_1d_path` column and in each
`*.result.json` (empty for other libraries, GPU rows and 2D rows), and
`summary.md` shows it with the method and names the call. Case files of the
non-default path carry it in their names (`...-dioptrin-pixel_split-split-w8-...`),
so the two paths never share files or one-worker references; `numerical.json`
compares each Dioptrin row with the reference of its own path (other rows with
`global`) and records `dioptrin_1d_path` and `reference_dioptrin_1d_path`.

**Result conversion.** pyFAI's timed `integrate1d`/`integrate2d` return
`Integrate1dResult`/`Integrate2dResult` objects holding NumPy arrays: the
radial axis (recomputed per call), intensity, sigma with Poisson errors and,
in 2D, the azimuthal axis. azint's timed `integrate` returns a tuple of NumPy
arrays, intensity and sigma (1D and 2D); its axes are constant attributes
(`radial_axis`, `azimuth_axis`) that it does not return per call. Dioptrin's
calls return `IntegrationResult` (inside `BatchResult` for batch calls), whose
data are read through attributes. So that every row ends with the same thing
in hand, every timed Dioptrin call (CPU 1D, every CPU 2D path, CUDA and Metal
in 1D and 2D) ends with NumPy float64 intensity, and sigma with Poisson errors,
for every frame: shape `(n,)` in 1D and `(n_azimuthal, n_radial)` in 2D. Batch
calls also unwrap every `BatchResult` (`is_ok()`, `.result`) inside timing.
The axes are per-integrator constants and, as for azint, are read outside
timing (pyFAI's per-call axes stay inside its timed call). Two conversions
exist, recorded per case in `dioptrin_result_conversion`:

- `array`: the zero-copy read-only `IntegrationResult` properties
  `intensity_array` and `errors_array` (float64 ndarrays), used whenever the
  installed build has them.
- `list`: `np.asarray(result.intensity)` and `np.asarray(result.errors)`
  (reshaped to the cake in 2D) for builds without them (0.15.2 and earlier),
  which build Python lists of floats first.

The `array` views are read-only (`flags.writeable` is False) and keep their
`IntegrationResult` alive; pyFAI's and azint's arrays are writable. Nothing in
the harness writes to the outputs, so this does not change what is timed, but
code that modifies results in place needs `.copy()` for Dioptrin.

The `list` conversion runs serially under the GIL and can cost more than the
integration: for a 2000x360 cake about 32 ms against 1.6 ms. A run with such a
build would mostly time list building, so the harness refuses it. Before any
case runs, the parent checks whether the installed `dioptrin.IntegrationResult`
has the accessors and, if not, stops with an error naming the cause (install
Dioptrin from a revision that has them; 0.15.2 wheels do not). Every case
process, and every rank with `--processes`, repeats the check on its first
result. `--allow-list-conversion` runs such a build anyway: the log starts
with a `WARNING LIST CONVERSION` banner, `summary.md` repeats the warning, the
case records keep `dioptrin_result_conversion` = `list`, and
`environment.json` records `dioptrin_result_arrays: false`. Use it only to
reproduce the older timing, never to compare with `array` rows or with other
libraries.

Validation and the saved outputs reuse the arrays the timed call produced (no
second conversion); it additionally requires float64 ndarrays of exactly the
expected shape and batch results in frame order (`array[i]`). Results before
this change timed only the native call and left the conversion outside timing.

For CPU Dioptrin rows the CPU-budget column is the number of threads that
compute, set up differently per call. On the `global` 1D path and the 2D
`sequential` path and `nested` batch split it is the size of the global Rayon
pool (`RAYON_NUM_THREADS`, which the case checks). On the `split` 1D path, the
2D `stream` path and the `frames` batch split it is `num_workers`: the call runs
that many frames at a time, each on a one-thread pool it builds itself, and the
global pool of the same size sits idle. For competitors it is the number of
persistent Python frame threads. Thread settings per case:

| Variant | Frame threads | `OMP_NUM_THREADS` | `RAYON_NUM_THREADS` |
| --- | ---: | ---: | ---: |
| `dioptrin-*` | (Rayon pool) | 1 | budget |
| `azint`, `azint-cached` | budget | budget | 1 |
| `pyfai-full`, `pyfai-bbox` | budget | 1 | 1 |
| `pyfai-full-omp` | 1 | budget | 1 |
| `pyfai-full-opencl` | 1 | 1 | 1 |
| `dioptrin-*-cuda`, `dioptrin-*-cuda-batch` | 1 | 1 | 1 |

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
per worker. Native batch scheduling, result allocation and, for Dioptrin, the
conversion of the results to NumPy arrays ("Result conversion" above) are
timed. Setup,
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
first touched on that node). Before any case runs, the runner fails with a
message naming the node if the node's CPUs in the runner's affinity are fewer
than the workers of the ranks bound to it (ranks sharing a node share its CPUs;
the largest budget/P scheduled is checked). It needs Linux and `numactl`; `--bind none`
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

### 2D cakes

`--dimensions 2d` (or `1d 2d`; the default is `1d`) measures azimuthally
resolved integration ("cakes") with the same fresh-process runner, workloads,
CPU budgets, `--processes`/`--bind`, `--mask-beam-pixel`, trials and
self-checks. Each case integrates onto `n_radial` x `n_azimuthal` bins;
`--cakes` selects the sizes (default `2000x360 1000x72`), always over the full
azimuth, chi in [-180, 180) degrees (pyFAI's convention, seam at +-180).

| Variant | 2D call | Uncertainties |
| --- | --- | --- |
| `dioptrin-pixel_split`, `-pyfai_full`, `-supersampled` | `Integrator.integrate2d(frame, n_radial, n_azimuthal, errors, azimuthal_range=(-pi, pi))` | native |
| `pyfai-full`, `pyfai-bbox`, `pyfai-full-omp`, `pyfai-full-opencl` | `integrate2d(..., radial_range, azimuth_range=(-180, 180), method=(split, "csr", impl))`, exact method, no fallback | native (`error_model="poisson"`) |
| `azint`, `azint-cached` | `AzimuthalIntegrator(..., azimuth_bins=linspace(0, 360, n+1))`, rows relabelled onto [-180, 180) | computed, but see below |

Threads per variant are those of the 1D table. Dioptrin's in-memory 2D path
is chosen with `--dioptrin-2d-path`:

- `sequential` (default): one frame at a time through `integrate2d`, whose
  sparse product runs row-parallel on the Rayon pool (`RAYON_NUM_THREADS` =
  CPU budget), like `pyfai-full-omp` but without copies. `integrate2d` holds
  the GIL, so Python frame threads would serialise; `batch2d` takes TIFF paths
  only.
- `stream`: `batch2d_iter` over the list of arrays with `num_workers` = budget,
  `num_threads=1` (frame-parallel). The stream copies every frame once, in one
  producer thread, before integrating it; on large detectors this copy limits
  throughput at higher budgets.
- `batch`: one `Integrator.batch2d(frames, n_radial, n_azimuthal, errors,
  azimuthal_range=(-pi, pi), ...)` call over the list of arrays per timed call.
  The arrays are borrowed, not copied. `--dioptrin-2d-batch-split` sets how the
  call splits the CPU budget B (several values give one case each):
  - `frames` (default): `num_workers=B, num_threads=1`. B frames at a time,
    each in a Rayon pool of one thread that the call builds (and drops) itself,
    with no parallelism inside a frame: the arrangement of the azint and pyFAI
    frame threads. The idle global pool (`RAYON_NUM_THREADS` = B, used for the
    first cake) is not used by the timed calls.
  - `nested`: neither argument. All frames run on the global Rayon pool of B
    threads (`RAYON_NUM_THREADS`), which also parallelises inside each frame.

  Either way the process computes on B threads; `cpu_per_wall` in each
  `*.result.json` (CPU time over wall time of the timed blocks) records it.
  With the `list` result conversion (`--allow-list-conversion`, builds without
  the accessors) the timed blocks also include the single-threaded conversion
  of the cakes to NumPy arrays after the call, which lowers `cpu_per_wall`
  below B; the `array` views are created without copying and do not.
  `nested` is the call the library makes by default for a list of arrays
  (Dioptrin 0.15.2: without `num_workers`/`num_threads` all frames run on the
  global pool, with the 4-frame tiled 2D kernels when the call has at least
  4 frames per pool thread and the method supports them).
  Needs a Dioptrin whose `batch2d` accepts NumPy arrays (after 0.14.0, which
  takes TIFF paths only): with an older build the cases are listed in
  `skipped.json` as unsupported, never run through another path. `frames` also
  needs a build whose array `batch2d` honours `num_workers`/`num_threads`
  (fix branch `a50915b` or later); earlier array builds ignored both and ran
  `nested`, which cannot be told apart from Python, so check the installed
  build record. Same bins, range, error mode, corrections, mask and
  self-checks as `sequential`.

The path is recorded per case in the `dioptrin_2d_path` column and the batch
split in `dioptrin_2d_batch_split` (empty for other paths and libraries). Case
files of non-default paths carry the path and split in their names
(`...-dioptrin-pixel_split-batch-frames-w8-...`), and `summary.md` lists the
split with the method. To time a
Dioptrin build from another branch, install it into a separate environment and
pass `--allow-unverified-build`: the source check compares the installed build
record with this checkout, so it reports the mismatch, and `environment.json`
keeps the installed record (`installed_build.source.commit`) and the problems
(`build_verification_problems`).

pyFAI builds one CSR engine per frame thread, sequentially during set-up, as in
1D; its "ready" time therefore grows with the budget. azint frame threads share
one sparse matrix.

**Same grid for every library.** Radial edges are `n_radial` equal bins between
the extremes of the pixel-corner 2theta (`preflight2d.json`, `radial_range_2d`).
Every library gets these edges: pyFAI and azint explicitly, Dioptrin as
`radial_range` of its `Integrator` (the 1D cases keep their own range, which
starts at 0). Since Dioptrin 0.15.0 every method honours `radial_range` in 2D:
`pixel_split` and `supersampled` then use the shared-grid tables of
`MultiIntegrator` (for `pixel_split` with exact fractional polygon areas, which
differ slightly from its default 2D table), and 2D results carry the radial
axis, which the harness reads. Builds before 0.15.0 honoured the range in 2D
only for `pyfai_full`: `pixel_split` used the corner extremes anyway (the same
edges), while `supersampled` used its own default edges (the extremes of the
subpixel hull and pixel centres; on Pilatus the lower edge differs by 0.013
degrees, most of a 2000-bin width), so its timed cakes were on their own
radial grid, reported in `numerical.json` as `max_radial_axis_difference_deg`.
Those builds also returned no 2D radial axis. The preflight records whether the
build returns one (`dioptrin_returns_2d_radial_axis` in `preflight2d.json`);
only when it does not, it reads fallback axes from each method's 1D default grid
(with the range for `pyfai_full` only, the old rule; `dioptrin_radial_axes`,
otherwise null), and a 2D result without a radial axis from a build that
returns one fails the case. For the numerical comparison only, the `supersampled` one-worker
reference case also integrates the first frame, after timing, with
`MultiIntegrator([integrator])`, which takes explicit edges, onto the common
grid (`*.cake0common.npz`, entries with `"grid": "common"`); with 0.15.0 or
later this is the same grid as its timed cakes. The preflight checks that
`MultiIntegrator` reproduces `Integrator.integrate2d` where both use the same
edges (`pixel_split`; `multiintegrator_check` in `preflight2d.json`; to within
4e-11 relative on the smoke datasets).

**Subpixels.** `--subpixels` sets the subpixels per side for
`dioptrin-supersampled`, `azint` and `azint-cached` in 2D (default 10; several
values give one case each, e.g. `--subpixels 5 10 20`). The value is in every
case record and in the `subpixels` column; 1D cases keep the per-dataset
`n_splitting`.

**Azimuth convention.** Before any 2D case, a fresh preflight process per
dataset integrates an asymmetric chi pattern (100 counts for 20 < chi < 80
degrees, 30 for -150 < chi < -120, 1 elsewhere; pyFAI's chi) with every
requested library onto a 100 x 36 cake, divides by the cake of a flat image and
requires the bright, medium and dark rows of every library to be where pyFAI
puts them, with matching azimuthal bin centres. A mirrored, rotated or flipped
cake fails, and then no 2D case of that dataset runs. azint measures azimuth on
[0, 360) with the same sign as pyFAI's chi; the harness gives it 0..360 edges
and rolls its rows by `n_azimuthal / 2`, which is exact for subpixel sampling
(no seam) but needs an even `n_azimuthal` (odd sizes are skipped for azint).
The result is in `<dataset>/preflight2d.json` (`convention`).

**Uncertainties.** All three libraries return 2D Poisson uncertainties. azint's
pinned `azint.py` divides the 2D sigma by the normalisation twice
(`errors = sqrt(var)/norm`, then `errors_2d = errors/norm`), so its 2D sigma is
timed as azint computes it but is not comparable; `uncertainty_support` records
this. A library that returned no 2D sigma would be recorded as `unsupported` in
`skipped.json`, not run without errors.

**Columns and outputs.** `results.csv` appends `dimension`, `n_radial`,
`n_azimuthal`, `subpixels`, `dioptrin_2d_path`, `uncertainty_support`,
`table_nnz`, `table_nnz_total`, `table_nnz_source`, `estimated_table_nnz`,
`estimated_memory_mib`, `rss_inputs_mib`, `peak_rss_ready_mib`,
`setup_rss_growth_mib` and `dioptrin_2d_batch_split` (empty for 1D rows; 1D columns and variant names are
unchanged), then `input_dtype`, `gpu_precision`, `dioptrin_1d_path` (CPU
Dioptrin 1D rows only) and `dioptrin_result_conversion` (every Dioptrin row;
see "Result conversion"). `setup_first_s` is the time to the first cake (import, set-up and
table construction) and `setup_ready_s` the time until every worker is built
and warmed. `peak_rss_ready_mib` is the peak RSS when all workers are ready,
before any validation reference is read; `peak_rss_mib` is the process-lifetime
peak; `setup_rss_growth_mib` is the peak growth from the loaded inputs to ready.
`table_nnz` is the number of stored weights per engine where the library exposes
it (pyFAI `engine.nnz`; `table_nnz_total` sums all engines); Dioptrin and azint
do not expose it, and `estimated_table_nnz` is then the estimate below.
`summary.md` has a "2D cakes" table.

**Numerical checks.** Every output cake, in order, is checked against the
same variant's one-worker reference before and after timing, as in 1D
(`azint-cached` is also checked bit for bit against stock azint). The first
cake of every one-worker case goes to `cases/*.cake0.npz`, and `numerical.json`
gets one entry (`"dimension": "2d"`) per case and reference: against
`dioptrin-pixel_split` (polygon clipping), and against a counterpart
(`pyfai-full` vs `dioptrin-pyfai_full`, `azint` vs `dioptrin-supersampled`,
OpenMP/OpenCL pyFAI vs `pyfai-full`). Each reports the axis differences, the
bins populated in both or only one, and for intensity and sigma the relative L2
difference, the number and fraction of bins differing by more than 1 %, the
maximum relative difference and the median ratio, separately for all bins, the
interior rows and the two rows next to the +-180 degree seam. Differences
between splitting rules are reported, not corrected: pyFAI full splitting
differs from polygon clipping at azimuthal bin edges and at the seam, and azint
averages corrected pixels (about 1-2 % different with polarization). The full
per-frame reference stacks are deleted after a successful run unless
`--keep-inputs`; the `cake0` files stay.

**Memory gate.** 2D tables are large, and pyFAI holds one per frame thread.
Before launching a 2D case the runner estimates its peak memory (per process:
interpreter and libraries, frames, per-pixel geometry arrays, one table per
engine including construction temporaries, two batches of results, the
validation reference) and skips it, with the estimate and limit in
`skipped.json` and `summary.md`, if it exceeds `--max-memory-fraction` (default
0.7) of physical RAM, or of a smaller cgroup limit. Table sizes are estimated
from the pixel-corner spans in 2theta and chi (`preflight2d.json`,
`table_estimates`): per pixel, (span/bin width + 1) radially times azimuthally,
capped at subpixels^2 for subpixel sampling; on Pilatus 1M and Eiger 4M this
matched pyFAI's `engine.nnz` to within 0.1 %. Model constants are in
`MEMORY_MODEL` (compare.py) and `environment.json`, calibrated on the Pilatus
and Eiger 4M smoke runs so that the estimate exceeds the measured peak RSS by
roughly 1.1-1.6x; each case records `estimated_memory_mib` next to its measured
peak. On the Eiger 9M subset the largest estimate at the laptop budgets is
pyFAI full CSR, 2000 x 360 with Poisson errors and the 32-frame workload: about
13 GiB at 8 frame threads (one 2D table per thread), against 7 GiB for Dioptrin
and azint at any budget, so nothing is skipped on a 48 GB machine at the
default 0.7. The per-dataset preflight itself is not gated; on the Eiger 9M
subset it peaks at about 7.5 GB. Cases whose one-worker reference did not run
are skipped with that reason.

Laptop run, 2D, pilatus and both subsets, five trials:

```bash
SUBSETS=/path/to/subsets
.venv-benchmark/bin/python benchmarks/comparison/compare.py \
  --datasets pilatus eiger9m-subset eiger4m-subset --subset-dir "$SUBSETS" \
  --dimensions 2d --cakes 2000x360 1000x72 --subpixels 10 --trials 5
```

(`--workloads` defaults to both; pilatus has only `repeated`. Add
`--allow-unverified-build` only for a deliberately stale build.)

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
transitive dependency across future dates. This suite measures in-memory 1D
integration and, with `--dimensions 2d`, 2D cakes (CPU, plus pyFAI's OpenCL CSR
where available); it does not assess decompression or file-reading performance.

## Detector-size and GPU comparison

See the [paper benchmark recipes](../../docs/benchmarking.md#repeat-the-detector-size-comparison)
for the fixed-work Windows and macOS sweeps.

The GPU runner uses the same geometry, detector masks, radial range and
bin count. It requires pyopencl in the comparison environment and native
CUDA with NVRTC available to the Rust test executable. Each prepared
uint32 frame is checked against the fixture for exact representation.
Dioptrin measures both ordinary and pinned uploads and calculates
products and sums in f64. pyFAI uses its full-split CSR OpenCL engine
with ordinary NumPy buffers. A per-frame CUDA series uses the same
one-frame-at-a-time workload. The CUDA batched series uses a single upload,
launch, and download for the whole batch.
The script checks the exact OpenCL implementation and device and rejects
fallbacks. The timed batch starts in host memory and ends with the
patterns back on host memory. Plan preparation, allocation and input
conversion into the already prepared buffers happen before warm timing.
The runner records a separate GPU kernel series. CUDA events enclose the
integration launch. OpenCL profiling sums pyFAI's buffer reset, pixel
preprocessing and CSR reduction kernels, plus any separate conversion kernel
in the profiled trace. Both device measurements exclude
the transfers and CPU dispatch in the host-to-pattern timings.

~~~text
python -B benchmarks/comparison/gpu_compare.py --gpu-name "RTX 5070 Ti" --frames 16 --bins 2000 --repeats 5 --trials 3 --output benchmark-results/paper-gpu
python -B benchmarks/comparison/plot_scaling.py --cpu benchmark-results/paper-cpu --gpu benchmark-results/paper-gpu/gpu-results.json --output benchmark-results/paper-figures
~~~

Pass --nvrtc with the directory containing the CUDA libraries if they
are outside the search path. The GPU results include intensity-only
and Poisson modes, five timed batches after two warmups, both CUDA
layouts, and per-layout comparisons against strict CPU output on two distinct
inputs. Cross-engine comparisons report absolute, relative L2 and positive-bin
errors. Axis differences above 0.0001 degrees or a relative L2 above 1% fail
fixture validation. First-pattern time starts with one decoded frame on the
host, before loading geometry or preparing a plan, in a fresh process for
each backend and layout. Driver and compiler caches keep their normal state.
The standard figure uses the Block layout for all CUDA series. The GPU series is
separate from CPU core scaling because it measures GPU work on a
single device. Plotting requires Matplotlib and produces separate GPU kernel
and host-to-pattern figures as PDF and PNG.

This comparison uses prepared inputs. The separate
[Eiger file benchmarks](../../docs/benchmarks/2026-11-18-cuda/README.md#native-integer-upload-and-hdf5)
report the effect of real HDF5 and TIFF loading, including decoding,
with their own calibration, file-cache settings and timing boundaries.

## Maxwell Slurm wrapper

For the paper runs, follow the [EuXFEL node selection and replay commands](../../docs/benchmarking.md#run-on-desy-maxwell).

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

The 2D equivalent on a CPU node passes the same options through. One job per
dataset keeps each job's wall time bounded. 2D set-up is long, and pyFAI
builds one 2D table per frame thread in sequence (a few seconds each for the
Eiger 9M subset, so minutes per case at 96 threads, for every error mode, cake
and trial), so raise the time limit, or run the `pyfai-full`/`pyfai-bbox`
Python-thread rows in a separate job with fewer budgets:

```bash
for DATASET in pilatus eiger9m-subset eiger4m-subset; do
  sbatch --time=24:00:00 benchmarks/comparison/maxwell.sbatch --datasets "$DATASET" \
    --subset-dir "$SUBSETS" --dimensions 2d --cakes 2000x360 1000x72 --subpixels 10 \
    --trials 5 --workers 1,2,4,8,12,24,48,96 --processes 1,2,4 --bind numa \
    --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full \
               azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp
done
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

For the Dioptrin CUDA rows add `WITH_CUDA=1` (Dioptrin built with the `cuda`
feature, in its own cached environment) and load a CUDA toolkit module through
`EXTRA_MODULES`; NVRTC comes from that module at run time only, and its
version must not be newer than the node driver's CUDA version (check
`nvidia-smi` on the node). `EXTRA_MODULES` is loaded before the environment
step, so the module's `LD_LIBRARY_PATH` is part of the cache key: keep the same
module for runs that should share an environment. With `WITH_CUDA=1` the
wrapper also saves `nvidia-smi` and `LD_LIBRARY_PATH`. One job for the GPU
comparison, frame-per-call and batch rows next to pyFAI OpenCL and the CPU
rows at one CPU:

```bash
WITH_OPENCL=1 WITH_CUDA=1 EXTRA_MODULES="<CUDA toolkit module>" \
  sbatch --partition=<gpu-partition> benchmarks/comparison/maxwell.sbatch \
  --datasets eiger9m-subset eiger4m-subset --subset-dir "$SUBSETS" \
  --workloads frames --workers 1 --bins 2000 \
  --variants dioptrin-pixel_split dioptrin-pyfai_full \
             dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda \
             dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch \
             pyfai-full pyfai-full-opencl --opencl-device gpu --cuda-device 0
```

With `--cpus-per-task` (as in the paper commands) the wrapper runs the
benchmark as `srun --ntasks=1 --cpus-per-task=N` with explicit binding:
`--hint=nomultithread` when the job used it (seen in `SLURM_HINT` or the job
record), otherwise `--cpu-bind=cores`, so the step gets whole cores. It writes
the step command to `launch.txt`. This is untested outside Slurm; check
`launch.txt` and the affinity in `environment.json`, and if a
`--hint=nomultithread` job's `launch.txt` lacks it, resubmit with
`SLURM_HINT=nomultithread` in the environment. Without `--cpus-per-task` the
launch is unchanged.

`allcpu` can be preempted/requeued; every restart writes a new results directory.
Higher worker counts can require substantial memory for pyFAI full CSR. Optional
wrapper environment settings are `BUILD_MODULES`, `EXTRA_MODULES`, `PYTHON_BIN`,
`BUILD_JOBS` (default 8), `RUST_TOOLCHAIN` (stable), `RESULT_ROOT`, `SUITE`
(standard), `CC`, `CXX`, `ENV_CACHE_ROOT`, `REBUILD_ENV=1`, `KEEP_BUILD=1`
(node-local build tools), `AZINT_FAST_MATH` (default 1), `WITH_OPENCL` (default 0)
and `WITH_CUDA` (default 0). All three build switches are part of the
environment cache key.

References: [Maxwell batch jobs](https://docs.desy.de/maxwell/documentation/jobs/running_batch_jobs/),
[Python modules](https://docs.desy.de/maxwell/development/python/),
[azint](https://maxiv-science.github.io/azint/).
