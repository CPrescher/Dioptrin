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
python -E -s -B benchmarks/comparison/compare.py --offline --datasets pilatus synthetic-4m synthetic-9m --workloads repeated --workers 1,2,4,8,12,14 --variants dioptrin-supersampled azint azint-prepared-mask dioptrin-pyfai_full pyfai-full --modes poisson --bins 2000 --frames 16 --trials 3 --blocks 3 --seconds 0.25 --output benchmark-results/paper-cpu
```

On the M4 Pro MacBook, run the same workload with 1, 2, 4, 6, 8 and 10 workers.
The chip has 10 performance cores and 4 efficiency cores. macOS schedules the
workers. Use 1, 2, 4 and 8 for direct comparisons with the Windows sweep above.

```bash
python3 benchmarks/comparison/setup.py
.venv-benchmark/bin/python -E -s -B benchmarks/comparison/compare.py \
  --offline --datasets pilatus synthetic-4m synthetic-9m --workloads repeated \
  --workers 1,2,4,6,8,10 \
  --variants dioptrin-supersampled azint azint-prepared-mask dioptrin-pyfai_full pyfai-full \
  --modes poisson --bins 2000 --frames 16 --trials 3 --blocks 3 --seconds 0.25 \
  --output benchmark-results/paper-macos-cpu
```

- On DESY Maxwell, pass the same detector and algorithm arguments to
  `sbatch benchmarks/comparison/maxwell.sbatch`. Leave out `--output` so the
  wrapper can name the run by Slurm job. Choose worker budgets within the
  CPU allocation and use matching compute nodes for repeated trials. Request a
  GPU allocation for the GPU comparison if a suitable NVIDIA node is available.
  See the [Maxwell batch job documentation](https://docs.desy.de/maxwell/documentation/jobs/running_batch_jobs/).

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
