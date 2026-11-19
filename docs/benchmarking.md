# Comparison benchmarks

The [benchmark suite](https://github.com/CPrescher/Dioptrin/tree/main/benchmarks/comparison)
compares 1D integration across CPUs, GPUs and detector sizes. Check out the same
Dioptrin revision and use the same dependency versions and input hashes when
repeating measurements across machines.

## Timing boundaries

| Measurement | What is timed |
| --- | --- |
| CPU worker scaling | Integrate prepared host-memory frames and return patterns. Record setup, first result and peak process memory separately. |
| GPU integration | Time GPU kernels for decoded frames. Include on-device preprocessing, reduction and normalization, with transfer and CPU dispatch outside the device time. |
| GPU host to pattern | Upload prepared host-memory frames, integrate and return patterns to the host. Compare CUDA and pyFAI OpenCL one frame at a time and record CUDA batches with pinned and ordinary inputs. |
| GPU first pattern | Start with one decoded frame in host memory in a fresh process. Load the geometry and mask, initialize the backend and return one pattern. |
| Decode only | Read and decompress detector data into native host arrays. Record the file-cache conditions and exclude integration. |
| File to pattern | Start the processing workers, read and decode files or HDF5 datasets, integrate the frames and collect the patterns. Use a cold-cache first read for user-facing throughput and record the storage device. |

The worker sweep uses a measured Pilatus 1M frame and generated 4M and 9M
frames. The recipe below repeats one frame per detector, with the same mask,
geometry, corrections and 2,000 radial bins. Compare Dioptrin full splitting
with pyFAI full CSR, and Dioptrin supersampling with AZINT subpixel integration.
The suite records numerical differences alongside throughput. GPU trials use
native 32-bit pixels on the same fixtures and report intensity-only and Poisson
results. The separate [Eiger HDF5 and TIFF measurements](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-11-18-cuda/README.md#native-integer-upload-and-hdf5)
include file loading and decoding.
The [FabIO and dioptrin-io comparison](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-11-18-cuda/README.md#eiger-hdf5-with-fabio-and-dioptrin-io)
includes decoding into ordinary host memory and directly into GPU upload buffers.

## Benchmark scripts

Run these scripts from a repository checkout at the revision under test. The
checkout provides their supporting modules, Rust harnesses, and fixtures.

| Script | Measurement |
| --- | --- |
| {download}`compare.py <../benchmarks/comparison/compare.py>` | Prepared-input CPU worker scaling for Dioptrin, pyFAI, and AZINT. |
| {download}`gpu_compare.py <../benchmarks/comparison/gpu_compare.py>` | CUDA and OpenCL kernel and host-to-pattern timings. |
| {download}`plot_scaling.py <../benchmarks/comparison/plot_scaling.py>` | Generate the CPU and GPU figures from recorded timings. |
| {download}`eiger_decode.py <benchmarks/2026-11-18-cuda/eiger_decode.py>` | Decode a real Eiger HDF5 scan with FabIO or dioptrin-io. |
| {download}`hdf5_batch.py <benchmarks/2026-11-18-cuda/hdf5_batch.py>` and {download}`pyfai_hdf5.py <benchmarks/2026-11-18-cuda/pyfai_hdf5.py>` | Measure the Dioptrin and pyFAI Eiger pipelines under recorded cache conditions. |
| {download}`hdf5_persistent.py <benchmarks/2026-11-18-cuda/hdf5_persistent.py>` | Time new scans with a fresh or retained CUDA plan. |
| {download}`disk_batch.py <benchmarks/2026-11-18-cuda/disk_batch.py>` | Compare buffered and unbuffered TIFF reads with a fixed CUDA layout. |
| {download}`prepare_io_benchmarks.py <benchmarks/2026-11-18-cuda/prepare_io_benchmarks.py>` | Build the isolated dioptrin-io reader and record build provenance. |

The [Eiger reproduction commands](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-11-18-cuda/README.md#reproduction)
show how to build the isolated dioptrin-io variant and run the file benchmarks.

## Repeat the detector-size comparison

Follow the [environment setup and validation instructions](https://github.com/CPrescher/Dioptrin/tree/main/benchmarks/comparison#clone-install-and-run).
Activate the comparison environment and run the CPU sweep from the repository
root. On Windows, use the workstation's worker budgets.

```text
python -E -s -B benchmarks/comparison/compare.py --offline --datasets pilatus synthetic-4m synthetic-9m --workloads repeated --workers 1,2,4,6,8,10,12,14 --variants dioptrin-supersampled azint azint-prepared-mask dioptrin-pyfai_full pyfai-full --modes poisson --bins 2000 --frames 16 --trials 3 --blocks 3 --seconds 0.25 --output benchmark-results/paper-cpu
```

On the M4 Pro MacBook, run the same workload with 1, 2, 4, 6, 8 and 10 workers.
The chip has 10 performance cores and 4 efficiency cores. macOS schedules the
workers. Windows and Maxwell use the same worker budgets through 10, with
additional 12- and 14-worker points.

```bash
python3 benchmarks/comparison/setup.py
.venv-benchmark/bin/python -E -s -B benchmarks/comparison/compare.py \
  --offline --datasets pilatus synthetic-4m synthetic-9m --workloads repeated \
  --workers 1,2,4,6,8,10 \
  --variants dioptrin-supersampled azint azint-prepared-mask dioptrin-pyfai_full pyfai-full \
  --modes poisson --bins 2000 --frames 16 --trials 3 --blocks 3 --seconds 0.25 \
  --output benchmark-results/paper-macos-cpu
```

For GPU runs, install PyOpenCL and make CUDA NVRTC available to the benchmark
executable. The CUDA 13 build needs compute capability 7.5 or newer and a
driver compatible with the installed NVRTC release's PTX. Check the GPU and
driver on a Maxwell allocation before submitting the GPU job. Specify the
NVIDIA model present in the allocation. The runner
checks that both backends use the requested model and that pyFAI selected its
OpenCL engine. Use a single-GPU allocation where both runtimes can see the
same device.

[NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/latest/why-cuda-compatibility.html)

For the Windows x64 replay, set `$env:RUSTFLAGS = '-C target-cpu=x86-64-v3'`
in a Visual Studio Developer PowerShell before building the GPU benchmark.
Pass `--nvrtc` if the NVRTC libraries are outside the Windows search path.

```text
python -B benchmarks/comparison/gpu_compare.py --gpu-name "RTX 5070 Ti" --frames 16 --bins 2000 --repeats 5 --trials 3 --output benchmark-results/paper-gpu
```

## Run on DESY Maxwell

Log in to `max-exfl-display.desy.de` with your Maxwell account. Clone the
benchmark revision on shared storage. The example uses the PR #67 branch.
Submit jobs from the login shell and run the timed work on Slurm compute nodes. Check
your partitions before submitting. DESY lists `exfel` for eligible EuXFEL
staff and `upex` for eligible EuXFEL users. [Maxwell login](https://docs.desy.de/maxwell/documentation/interactive_login/), [partition access](https://docs.desy.de/maxwell/documentation/access/), [EuXFEL allocation guide](https://dataanalysis.pages.xfel.eu/user-documentation/jhub/)

```bash
ssh USER@max-exfl-display.desy.de
git clone --branch codex/cuda-1d-prototype https://github.com/CPrescher/Dioptrin.git
cd Dioptrin
git rev-parse HEAD
/software/tools/bin/my-partitions
partition=exfel                   # use upex if that is your allocation
sinfo -N -p "$partition" -o '%N %t %c %m %f %G'
```

Use the same checked-out revision as the Mac run. Use Python 3.14 if
available. The batch wrapper accepts `PYTHON_BIN` when the default interpreter
differs. To prepare that interpreter once, run the following from the login
shell. [Maxwell Mamba setup](https://docs.desy.de/maxwell/development/python/mamba-setup/)

```bash
module load maxwell mamba
. mamba-init
mamba create -y -n dioptrin-paper python=3.14
mamba activate dioptrin-paper
export PYTHON_BIN="$(command -v python)"
```

### CPU target and submission

Target one exclusive **AMD EPYC CPU node** in `exfel` or `upex`. Start with
the `EPYC` constraint. When `sinfo` shows a specific available CPU model,
use that model's feature together with `EPYC` for subsequent submissions.
If your allocation offers Intel Xeon nodes, use one Xeon model consistently
when an EPYC node is unavailable.
The example requests 14 physical cores with one hardware thread per core.
Check the recorded effective CPU affinity and model when comparing runs.
The worker budgets still describe the number of software workers on each
machine. [Maxwell node constraints](https://docs.desy.de/maxwell/documentation/jobs/running_batch_jobs/#using-constraints), [Slurm CPU placement](https://slurm.schedmd.com/mc_support.html)

```bash
cpu_feature=EPYC
sbatch --partition="$partition" --constraint="$cpu_feature" \
  --ntasks=1 --cpus-per-task=14 --hint=nomultithread --time=02:00:00 \
  benchmarks/comparison/maxwell.sbatch \
  --offline --datasets pilatus synthetic-4m synthetic-9m --workloads repeated \
  --workers 1,2,4,6,8,10,12,14 \
  --variants dioptrin-supersampled azint azint-prepared-mask dioptrin-pyfai_full pyfai-full \
  --modes poisson --bins 2000 --frames 16 --trials 3 --blocks 3 --seconds 0.25
squeue -u "$USER"
```

The wrapper builds and caches the Python environment on the first run for a
given node class and source revision. It writes the result to
`benchmark-results/maxwell/<job-id>-.../comparison/` and saves the hostname,
Slurm allocation and module list alongside it. Keep `comparison/results.csv`,
`comparison/numerical.json`, `comparison/environment.json` and the build
record for the PR. `sbatch --test-only` can check the resource request.
Use the partition list above to check account access.

### GPU target and allocation

Target **one NVIDIA A100** for both Dioptrin CUDA and pyFAI OpenCL. Its
compute capability is 8.0, and it provides a data-center FP64 comparison to
the Windows RTX 5070 Ti. Check for `A100` under your EuXFEL partition first.
The public Maxwell availability snapshot on 19 November 2026 showed P100s
under `exfel` and `upex`. A P100 has compute capability 6.0 and would need
a separate CUDA 12 build and validation for this comparison. If the live
EuXFEL list has only P100s, check `allgpu` for an A100 if `my-partitions`
lists it. `allgpu` jobs can be preempted. An H100 or H200 is another
compatible target if listed and accessible. [Maxwell GPU availability](https://max-jhub.desy.de/hub/static/nodes.html), [partition preemption](https://docs.desy.de/maxwell/documentation/jobs/preemption/), [NVIDIA GPU capabilities](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/), [CUDA 13 architecture support](https://docs.nvidia.com/cuda/cuda-features-archive/)

```bash
sinfo -N -p "$partition" -o '%N %t %f %G' | grep -E 'A100|H100|H200|P100'
sinfo -N -p allgpu -o '%N %t %f %G' | grep -E 'A100|H100|H200'
gpu_partition="$partition"        # use allgpu if that is where the A100 is available
gpu_feature=A100                  # use the exact Slurm feature shown by sinfo
/software/tools/bin/savail -p "$gpu_partition" -f "$gpu_feature"
srun --partition="$gpu_partition" --constraint="$gpu_feature" --gres=gpu:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=14 --hint=nomultithread \
  --time=02:00:00 --pty bash -l
```

Inside the allocation, check the model, GPU visibility and driver before
building. For the Windows-matched NVRTC 13.4.59, target an R615 or newer
driver, and run the CUDA smoke test to verify that the driver accepts the
generated PTX. Use a recent Rust toolchain on the compute node. Install it
in your user environment with `rustup` if `cargo` is unavailable. The NVRTC
wheel provides the user-space compiler libraries. The node supplies the
driver and NVIDIA OpenCL ICD. [CUDA 13.4 driver branches](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver), [PTX compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [Rust installation](https://www.rust-lang.org/tools/install), [NVIDIA NVRTC wheel](https://pypi.org/project/nvidia-cuda-nvrtc/13.4.59/)

```bash
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv
module load maxwell mamba
. mamba-init
mamba activate dioptrin-paper
command -v cargo && rustc -Vv
gpu_env="$PWD/.benchmark-environments/maxwell-gpu-$SLURM_JOB_ID"
python benchmarks/comparison/setup.py --venv "$gpu_env"
gpu_python="$gpu_env/bin/python"
"$gpu_python" -m pip install pyopencl==2026.1.5 nvidia-cuda-nvrtc==13.4.59
nvrtc_dir=$("$gpu_python" - <<'PY'
from importlib.metadata import distribution
nvrtc = distribution("nvidia-cuda-nvrtc")
library = next(p for p in nvrtc.files if p.name.startswith("libnvrtc.so"))
print(nvrtc.locate_file(library).parent)
PY
)
export LD_LIBRARY_PATH="$nvrtc_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RUSTFLAGS='-C target-cpu=native'
DIOPTRIN_REQUIRE_CUDA=1 cargo test --release --features cuda --lib \
  integration::cuda::tests -- --nocapture
"$gpu_python" -c 'import pyopencl as cl; print([(p.name, d.name) for p in cl.get_platforms() for d in p.get_devices() if d.type & cl.device_type.GPU])'
```

Continue if the allocated NVIDIA GPU is visible to both runtimes and the
smoke test passes. The comparison expects a unique OpenCL GPU matching the
requested model. If multiple matching devices are visible, request an
isolated GPU allocation where both runtimes can select the same device.
Set `gpu_model` in the allocated shell to the actual name substring.

```bash
gpu_model=A100
"$gpu_python" -E -s -B benchmarks/comparison/gpu_compare.py \
  --gpu-name "$gpu_model" --frames 16 --bins 2000 --repeats 5 --trials 3 \
  --output "benchmark-results/maxwell-gpu-$SLURM_JOB_ID"
```

Keep `gpu-results.json` and the per-case logs. The Maxwell commands above
measure prepared-input integration and startup. The Windows cold-file
recipe uses Windows unbuffered copies and drive counters.

## Machine records and results

The CPU runner writes `environment.json` and the GPU runner writes
`gpu-results.json`. Both capture the OS, CPU model, physical and logical core
counts, RAM and, where available, CPU affinity and GPU inventory. The GPU run
records the device and driver used by OpenCL. On Maxwell, the suite records the
Slurm allocation, and the wrapper saves the scheduler's job details and the
loaded software modules. Keep the CPU worker budget with the results because a
worker count can include simultaneous multithreading on one physical core.
Workers follow the OS scheduler within the recorded allocation. Report these
as worker budgets, since affinity does not identify which cores handled each batch.

Add any relevant run conditions that require local knowledge, such as laptop
power mode, whether it was plugged in, competing jobs, or which storage held
the files for a file-to-pattern run. Keep the benchmark code and fixtures in the
Dioptrin repository and archive the paper's raw timings, figures, environment
records and input hashes with its supplementary material.
