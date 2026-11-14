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

## What is measured and checked

All methods use the same float64 input arrays, masks, physical geometry,
2theta bounds, bin count, solid-angle correction, and polarization 0.965. The
standard suite uses 2,000 bins and reports intensity-only and Poisson modes.

Methods are Dioptrin `pixel_split`, `supersampled`, and `pyfai_full`; azint
subpixel integration; pyFAI full CSR and bbox CSR using Cython. Supersampling is
15 x 15 for Pilatus, 5 x 5 for the larger generated detectors, and 3 x 3 for quick.
The runner requires pyFAI to use the requested method and rejects fallbacks.

These algorithms, internal precision, and correction estimators differ.
Cross-library differences are reported without asserting identical answers.
Azint averages corrected pixels, while Dioptrin/pyFAI normalize weighted raw
counts. pyFAI can slightly expand the upper bin bound; axis offsets are reported.

The CPU-budget column is the total global Rayon pool size for Dioptrin's native
array batch API. For competitors it is the number of persistent Python frame
threads, each with one OpenMP/BLAS thread. Azint shares its sparse matrix through
shallow copies: geometry, mask, and sparse arrays are shared without modification;
the per-call `norm_1d`/`norm_2d` attributes are assigned on each copy, and scratch
arrays are newly allocated per call. Its pinned native SpMV releases the GIL.
PyFAI owns an initialized CSR engine
per worker. Native batch scheduling and result allocation are timed. Setup,
first-result and all-workers-ready times, and process-lifetime peak RSS are
separate columns. Setup includes backend import and lazy initialization.
Frame workers receive contiguous slices of the logical input sequence, so the
distinct-frame workload does not pin a worker to one image at 4/8 CPUs. Batch
lengths are rounded up to a multiple of the CPU budget for balanced slices.
The fixed pyFAI mask is installed on its detector once, using its cached mask
checksum rather than recalculating the checksum from a mask argument each call.

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
transitive dependency across future dates. This suite measures CPU 1D integration; it does not assess
2D caking, GPU integration, decompression, or file-reading performance.

## Maxwell Slurm wrapper

From a terminal on Maxwell, clone the repository and submit from its root:

```bash
git clone https://github.com/CPrescher/Dioptrin.git
cd Dioptrin
/software/tools/bin/my-partitions
sbatch benchmarks/comparison/maxwell.sbatch
```

The job loads `maxwell mamba`, installs a minimal Rust toolchain in node-local
temporary storage, invokes the same `setup.py`, then runs `compare.py --suite
standard`. All compilation and timed work run on the allocated compute node.
Wheels/build records and results remain on shared storage under
`benchmark-results/maxwell/<job-id>-.../`; temporary tools are cleaned on success.
It needs outbound HTTPS to GitHub, PyPI, and Rust distribution servers.

Slurm requests one exclusive CPU node, its memory, two hours, and `allcpu` by
default. Override the partition with one your account can use, and pass runner
options after the script name:

```bash
sbatch --partition=upex benchmarks/comparison/maxwell.sbatch --suite quick
sbatch benchmarks/comparison/maxwell.sbatch --workers 1,2,4,8,16 --trials 2
squeue -u "$USER"
tail -f dioptrin-JOBID.out
```

`allcpu` can be preempted/requeued; every restart writes a new results directory.
Higher worker counts can require substantial memory for pyFAI full CSR. Optional
wrapper environment settings are `BUILD_MODULES`, `PYTHON_BIN`, `BUILD_JOBS`
(default 8), `RUST_TOOLCHAIN` (stable), `RESULT_ROOT`, `SUITE` (standard), `CC`, `CXX`, and `KEEP_BUILD=1`.
CPU-specific builds are made afresh for the allocated node.

References: [Maxwell batch jobs](https://docs.desy.de/maxwell/documentation/jobs/running_batch_jobs/),
[Python modules](https://docs.desy.de/maxwell/development/python/),
[azint](https://maxiv-science.github.io/azint/).
