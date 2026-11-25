# Final Paper Benchmarks (2026-11-24, Dioptrin v0.15.2)

Cross-machine protocol comparing **Dioptrin (CPU + GPU)**, **pyFAI (CPU + OpenCL GPU)**, and **azint (CPU)** using identical fixtures, radial bins, 2D cake grids (`2000x360`, `1000x72`, χ in `[-180, 180)` deg), masks, polarization (`0.965`), solid-angle correction, and numerical verification across all three machines.

## Shared Settings & Platform Caveats

- **Dioptrin version:** `v0.15.2` (plus PR #86 comparison harness updates; record the exact `git rev-parse HEAD` in `environment.json`).
- **Dimensions:** `1d` and `2d` (`--dimensions 1d 2d`).
- **2D cakes:** `2000x360` and `1000x72` (default `--cakes`).
- **GPU precision:** `f32_compensated` (default `--gpu-precision`): compensated single-precision accumulation with float64 normalization/output (comparable to pyFAI OpenCL's compensated float32 accumulation, and supported identically on both CUDA and Apple Silicon Metal).
- **Input dtype:**
  - Primary pass: `float64` (default `--input-dtype float64`).
  - Optional native-f32 pass: `--input-dtype float32` (asserted lossless on the integer-count fixtures). Note that Dioptrin consumes `float32` natively without upcasting, pyFAI's CSR/OpenCL engines convert to `float32` internally anyway (so `float32` saves a host conversion), and `azint` may convert `float32` to `float64` depending on its binding.
  - Optional CUDA `f64` ablation (CUDA only): `--gpu-precision f64`.
- **Compiler / `azint` build differences to report in the paper:**
  - `maxwell.sbatch` builds `azint` with `AZINT_FAST_MATH=1` (`-ffast-math`) and `-march=native` by default.
  - On macOS, pass `--azint-fast-math` to `setup.py` to match Maxwell (`-ffast-math -flto=thin -mcpu=native`), or omit it for strict `azint`.
  - On native Windows (MSVC), `--azint-fast-math` is not supported and both Dioptrin and `azint` target `x86-64-v3` (AVX2/FMA). Every run records exact compiler flags in `environment.json` and `summary.md`.

---

## 1. Windows Workstation (RTX 5070 Ti, native Windows)

Open a **Visual Studio Developer PowerShell** (MSVC `cl` + Rust MSVC toolchain; no CUDA Toolkit needed):

```powershell
python benchmarks/comparison/setup.py --with-cuda --with-opencl
```

Primary `float64` 1D + 2D CPU & GPU suite:

```powershell
.venv-benchmark\Scripts\python.exe -E -s -B benchmarks/comparison/compare.py `
  --dimensions 1d 2d `
  --opencl-device gpu --cuda-device 0 --gpu-precision f32_compensated `
  --output benchmark-results/2026-11-24-windows-f64 `
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full `
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl `
             dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda `
             dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch
```

Optional `float32` input pass:

```powershell
.venv-benchmark\Scripts\python.exe -E -s -B benchmarks/comparison/compare.py `
  --dimensions 1d 2d --input-dtype float32 `
  --opencl-device gpu --cuda-device 0 --gpu-precision f32_compensated `
  --output benchmark-results/2026-11-24-windows-f32 `
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full `
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl `
             dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda `
             dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch
```

Optional CUDA `f64` precision ablation (GPU rows only):

```powershell
.venv-benchmark\Scripts\python.exe -E -s -B benchmarks/comparison/compare.py `
  --dimensions 1d 2d --cuda-device 0 --gpu-precision f64 `
  --output benchmark-results/2026-11-24-windows-cuda-f64 `
  --variants dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda `
             dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch
```

---

## 2. Maxwell Cluster (DESY, Linux + NVIDIA GPU)

With `v0.15.2`'s embedded `compute_70` PTX, no CUDA Toolkit module is required on Volta+ GPUs (V100, A100, H100, L40S, etc.); on pre-Volta GPUs (e.g. P100), load a matching CUDA toolkit module via `EXTRA_MODULES` for NVRTC fallback. If `pyfai-full-opencl` needs a site OpenCL ICD/driver module on the GPU partition, add it to `EXTRA_MODULES` as well. Increase `--time` beyond the default 2 hours for the full 1D + 2D standard suite.

Primary `float64` 1D + 2D CPU & GPU suite:

```bash
WITH_OPENCL=1 WITH_CUDA=1 \
  sbatch --partition=<gpu-partition> --constraint=<V100|A100|...> --time=08:00:00 \
  benchmarks/comparison/maxwell.sbatch \
  --dimensions 1d 2d \
  --opencl-device gpu --cuda-device 0 --gpu-precision f32_compensated \
  --output benchmark-results/2026-11-24-maxwell-f64 \
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full \
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl \
             dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda \
             dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch
```

Optional `float32` input pass (same `sbatch` with `--input-dtype float32 --output benchmark-results/2026-11-24-maxwell-f32`) and optional CUDA `f64` ablation (`--gpu-precision f64 --output benchmark-results/2026-11-24-maxwell-cuda-f64` with the 4 `-cuda*` variants).

---

## 3. MacBook (Apple Silicon, macOS + Metal)

Install GCC via Homebrew (`brew install gcc`, required for `azint` OpenMP) and build the benchmark environment with Metal + OpenCL (`--azint-fast-math` matches Maxwell's `AZINT_FAST_MATH=1` default):

```bash
python3 benchmarks/comparison/setup.py --with-metal --with-opencl --azint-fast-math
```

Primary `float64` 1D + 2D CPU & GPU suite:

```bash
.venv-benchmark/bin/python benchmarks/comparison/compare.py \
  --dimensions 1d 2d \
  --opencl-device gpu --metal-device 0 --gpu-precision f32_compensated \
  --output benchmark-results/2026-11-24-macos-f64 \
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full \
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl \
             dioptrin-pixel_split-metal dioptrin-pyfai_full-metal \
             dioptrin-pixel_split-metal-batch dioptrin-pyfai_full-metal-batch
```

Optional `float32` input pass (add `--input-dtype float32 --output benchmark-results/2026-11-24-macos-f32`).

---

## Notes

- Run `--suite quick` first on each machine to verify installation, GPU detection, and `pyfai-full-opencl` / `pyfai-full-omp` availability before launching the full `standard` suite.
- Raw `benchmark-results/<timestamp>/` directories stay out of Git (per `AGENTS.md`); summarize final measured tables and hardware conditions here once runs complete.

---

## Measured Results

### Maxwell Cluster (AMD EPYC 9475F + NVIDIA H200 NVL, Linux + CUDA)

- **Revision:** `118a299` (`dioptrin 0.15.2`, `python,cuda` with embedded `compute_70` PTX)
- **Hardware / OS:** `max-exflg047.desy.de` (DESY Maxwell `upex` partition, Slurm jobs `25939264` & `25939280`), AMD EPYC 9475F (96 physical cores / 192 logical CPUs), 754 GiB RAM, NVIDIA H200 NVL (140 GiB, Compute Capability 9.0, driver `595.91.07`, CUDA 13.2), `Linux-5.14.0-687.65.1.el9_8.x86_64`
- **Toolchain & libraries:** Python 3.11, `rustc 1.100.0`, `g++ 11.5.0`, `numpy 2.4.2`, `pyFAI 2026.9.0` (OpenMP enabled via `libgomp`), `pyopencl 2026.1.5` (NVIDIA CUDA OpenCL 3.0), `azint 1.2.6.dev6+g4f78cc3b1` built with `-O3 -march=native -ffast-math` (`azint-cached` bit-identical to stock `azint`).
- **Suite settings:** `standard` `float64` suite (2,000 radial bins in 1D; `2000x360` and `1000x72` cakes in 2D with `subpixels=10`; 32 frames/batch, 3 blocks, 0.3 s/block, 1 trial; CPU budgets 1, 2, 4, 8; GPU at 1 host thread with `f32_compensated`, pageable host NumPy inputs).
- **Status & pyFAI 2D patch note:** Stock run (`2026-11-24-maxwell-f64`, job `25939264`) completed 1,056 / 1,110 cases with 0 skips; 54 multi-worker 2D `pyfai-full` / `pyfai-full-omp` cases failed self-consistency due to an uninitialized scratch buffer (`numpy.empty(256)` in `pyFAI/ext/splitpixel_common.pyx:FullSplitIntegrator`). A follow-up rerun (`2026-11-24-maxwell-f64-pyfai-patched`, job `25939280`) with a one-line patch (`numpy.empty(256)` → `numpy.zeros(256)`, archived as `zero-lut-scratch.patch`) passed all 180 / 180 2D pyFAI cases, accounting for all 54 stock failures (`all_expected_cases_accounted_for: true`; 48 expected 2D `negative_sigma` warnings). Entries marked `*` below come from the patched pyFAI rerun where the stock case failed.
- **Accuracy (`f32_compensated` vs CPU counterpart):** Dioptrin CUDA max relative L2 ≤ `5.70e-9` (1D intensity), ≤ `1.77e-8` (1D sigma), ≤ `1.98e-8` (2D intensity), ≤ `1.90e-8` (2D sigma) — matching Apple Silicon Metal to 3 significant digits. `pyfai-full-opencl` max relative L2 ≤ `6.67e-8` (1D intensity), ≤ `7.37e-8` (1D sigma), and vs patched `pyfai-full` CPU ≤ `6.73e-8` (2D intensity), ≤ `6.27e-8` (2D sigma) (against the unpatched stock 2D CPU reference, the uninitialized buffer caused up to `2.10e-6` intensity / `3.87e-3` sigma disagreement).
- **Compact records archive:** `2026-11-24-maxwell-f64`, `2026-11-24-maxwell-f64-pyfai-patched`, `2026-11-24-maxwell-completion.json`, and Slurm/build records (`maxwell/25939264-*`, `maxwell/25939280-*`) saved outside Git (Dropbox `Science/4 - Scientific Projects/Software/Dioptrin/benchmarks/2026-11-24-paper-final/`).

#### Maxwell `float64` pass — Poisson mode (median frames/s; CPU shown as `1w / 8w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `pyfai-full-omp` | `dioptrin-pixel_split-cuda` | `dioptrin-pyfai_full-cuda` | `dioptrin-pixel_split-cuda-batch` | `dioptrin-pyfai_full-cuda-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 268.1 / 1968.7 | 262.8 / 1899.8 | 277.4 / 1997.5 | 134.8 / 657.3 | 258.8 / 1395.8 | 103.3 / 487.0 | 103.8 / 428.1 | 107.9 / 194.2 | 4481.9 | 4602.6 | 7953.8 | 8014.5 | 1789.0 |
| `synthetic-4m` (frames) | 1D | 95.3 / 706.9 | 95.6 / 710.6 | 101.7 / 760.5 | 21.1 / 165.7 | 27.4 / 206.5 | 13.1 / 100.8 | 14.3 / 98.5 | 13.7 / 48.3 | 947.8 | 957.4 | 1040.1 | 1031.1 | 600.1 |
| `synthetic-9m` (frames) | 1D | 51.5 / 383.2 | 51.7 / 390.4 | 55.2 / 413.7 | 10.7 / 81.2 | 13.5 / 103.0 | 7.4 / 55.5 | 7.4 / 55.4 | 7.3 / 23.7 | 378.8 | 378.0 | 426.5 | 434.4 | 200.9 |
| `pilatus` (repeated) | 2D `2000x360` | 70.4 / 265.5 | 73.9 / 266.5 | 67.7 / 262.1 | 60.1 / 351.6 | 77.6 / 576.4 | 60.2 / 337.6* | 57.9 / 338.8 | 58.8 / 175.8* | 844.3 | 841.8 | 1235.4 | 1233.1 | 420.1 |
| `synthetic-4m` (frames) | 2D `2000x360` | 40.9 / 202.9 | 41.4 / 206.3 | 42.6 / 205.2 | 22.9 / 157.8 | 29.9 / 229.2 | 16.5 / 115.2* | 16.2 / 113.0 | 16.7 / 54.9 | 490.0 | 493.2 | 627.6 | 624.2 | 258.1 |
| `synthetic-9m` (frames) | 2D `2000x360` | 31.2 / 157.7 | 31.2 / 157.6 | 31.1 / 159.5 | 12.2 / 91.1 | 16.9 / 128.9 | 8.9 / 64.1* | 8.4 / 57.7 | 8.9 / 27.2* | 276.9 | 275.3 | 340.8 | 341.0 | 172.7 |
| `pilatus` (repeated) | 2D `1000x72` | 438.2 / 1741.5 | 447.3 / 1786.4 | 505.8 / 1906.1 | 225.3 / 1499.7 | 391.9 / 2252.6 | 133.5 / 748.4 | 108.3 / 641.9 | 135.1 / 258.8* | 3456.1 | 3463.6 | 5661.9 | 5521.9 | 1546.7 |
| `synthetic-4m` (frames) | 2D `1000x72` | 125.4 / 752.3 | 126.1 / 753.1 | 125.3 / 718.6 | 48.7 / 339.8 | 71.6 / 531.9 | 26.5 / 195.9 | 24.1 / 178.2 | 26.1 / 66.3 | 883.1 | 891.4 | 971.8 | 969.4 | 552.1 |
| `synthetic-9m` (frames) | 2D `1000x72` | 74.4 / 453.1 | 74.0 / 452.1 | 74.9 / 416.3 | 24.0 / 165.7 | 37.7 / 273.5 | 13.1 / 95.1 | 12.0 / 68.6 | 13.0 / 30.4 | 368.4 | 368.8 | 414.6 | 420.8 | 205.6 |

#### Maxwell `float64` pass — Intensity-only mode (median frames/s; CPU shown as `1w / 8w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `pyfai-full-omp` | `dioptrin-pixel_split-cuda` | `dioptrin-pyfai_full-cuda` | `dioptrin-pixel_split-cuda-batch` | `dioptrin-pyfai_full-cuda-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 347.7 / 2451.7 | 338.8 / 2398.1 | 357.1 / 2518.6 | 186.3 / 1066.2 | 471.4 / 3256.2 | 120.0 / 461.8 | 120.4 / 464.1 | 110.1 / 199.5 | 4836.9 | 4832.6 | 7994.4 | 7896.2 | 1774.5 |
| `synthetic-4m` (frames) | 1D | 121.5 / 884.4 | 121.1 / 879.4 | 125.5 / 940.4 | 34.8 / 267.2 | 52.3 / 390.4 | 13.7 / 97.1 | 15.2 / 113.5 | 14.0 / 51.1 | 965.9 | 960.5 | 1033.8 | 1032.7 | 591.1 |
| `synthetic-9m` (frames) | 1D | 64.8 / 468.2 | 64.7 / 469.1 | 69.3 / 506.3 | 17.4 / 126.4 | 27.2 / 203.4 | 8.0 / 59.0 | 8.0 / 59.1 | 7.9 / 24.6 | 380.6 | 377.8 | 433.5 | 427.3 | 199.0 |
| `pilatus` (repeated) | 2D `2000x360` | 202.7 / 1179.1 | 201.9 / 1196.8 | 202.2 / 1186.4 | 99.0 / 717.1 | 179.6 / 1296.8 | 62.1 / 382.9 | 78.0 / 461.0 | 62.8 / 182.9 | 1686.2 | 1684.9 | 2901.7 | 2918.4 | 403.2 |
| `synthetic-4m` (frames) | 2D `2000x360` | 70.5 / 505.9 | 70.4 / 506.9 | 73.6 / 485.9 | 36.2 / 262.2 | 59.7 / 445.1 | 17.7 / 124.4* | 19.0 / 129.8 | 18.0 / 58.1* | 704.3 | 698.6 | 849.1 | 852.9 | 265.1 |
| `synthetic-9m` (frames) | 2D `2000x360` | 48.2 / 343.3 | 48.1 / 345.0 | 49.6 / 302.7 | 19.9 / 139.2 | 34.1 / 177.1 | 9.6 / 55.5 | 9.8 / 57.5 | 9.5 / 26.3* | 332.3 | 333.4 | 395.3 | 393.7 | 131.7 |
| `pilatus` (repeated) | 2D `1000x72` | 898.2 / 4262.1 | 879.8 / 4056.8 | 922.0 / 4258.7 | 318.0 / 1837.3 | 780.6 / 4584.7 | 144.5 / 797.6* | 152.4 / 896.3 | 143.1 / 263.2 | 4372.2 | 4419.4 | 8079.3 | 7717.0 | 1370.1 |
| `synthetic-4m` (frames) | 2D `1000x72` | 161.2 / 1040.6 | 160.1 / 1049.2 | 165.2 / 997.8 | 69.2 / 455.0 | 119.2 / 877.0 | 27.7 / 204.2 | 27.9 / 210.4 | 27.8 / 66.8 | 938.5 | 941.5 | 1017.7 | 1035.0 | 519.1 |
| `synthetic-9m` (frames) | 2D `1000x72` | 92.9 / 606.6 | 92.4 / 606.7 | 93.3 / 553.1 | 31.8 / 230.1 | 79.8 / 537.2 | 14.0 / 93.9 | 14.3 / 101.9 | 14.0 / 30.6 | 376.3 | 376.1 | 428.8 | 431.6 | 208.9 |

### MacBook (Apple M4 Pro, macOS + Metal)

- **Revision:** `118a299` (`dioptrin 0.15.2`, `python,metal`)
- **Hardware / OS:** Apple M4 Pro (14 cores: 10P + 4E; 20-core Apple GPU, unified memory), 48 GiB RAM, `macOS-26.6.2-arm64`
- **Toolchain & libraries:** Python 3.13.5, `rustc 1.99.1`, Homebrew `gcc-16`/`g++-16`, `numpy 2.4.2`, `pyFAI 2026.9.0`, `pyopencl 2026.1.5` (OpenCL 1.2 on Apple M4 Pro GPU, no FP64), `azint 1.2.6.dev6+g4f78cc3b1` built with `-O3 -mcpu=native -ffast-math` (subnormals preserved; `azint-cached` bit-identical to stock `azint`).
- **Suite settings:** `standard` suite (2,000 radial bins in 1D; `2000x360` and `1000x72` cakes in 2D with `subpixels=10`; 32 frames/batch, 3 blocks, 0.3 s/block, 1 trial; CPU budgets 1, 2, 4, 8; GPU at 1 host thread with `f32_compensated`).
- **Status & accuracy:** 990 measured rows per pass (330 1D, 660 2D); `pyfai-full-omp` skipped as expected (macOS pyFAI wheel has no OpenMP). All Dioptrin (CPU & Metal), `azint`, `azint-cached`, `pyfai-bbox`, and `pyfai-full-opencl` rows passed with `numerical_status=ok`; `pyfai-full` 2D recorded 24 expected `negative_sigma` warnings per pass. Max relative L2 vs CPU counterpart (`f32_compensated`, identical for `f64` and `f32` inputs): Dioptrin Metal ≤ `5.70e-9` (1D intensity), ≤ `1.77e-8` (1D sigma), ≤ `1.98e-8` (2D intensity), ≤ `1.90e-8` (2D sigma), compared with `pyfai-full-opencl` ≤ `5.25e-8` (1D intensity), ≤ `5.57e-8` (1D sigma), ≤ `5.38e-8` (2D intensity/sigma).
- **Compact records archive:** `2026-11-24-macos-f64`, `2026-11-24-macos-f32`, and `macos-build-record` saved outside Git (Dropbox `Science/4 - Scientific Projects/Software/Dioptrin/benchmarks/2026-11-24-paper-final/`).

#### Primary `float64` pass — Poisson mode (median frames/s; CPU shown as `1w / 8w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `dioptrin-pixel_split-metal` | `dioptrin-pyfai_full-metal` | `dioptrin-pixel_split-metal-batch` | `dioptrin-pyfai_full-metal-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 265.7 / 1942.1 | 263.9 / 1931.6 | 274.6 / 2006.1 | 82.9 / 610.9 | 132.4 / 1002.8 | 88.4 / 647.2 | 86.9 / 649.2 | 1695.7 | 1672.6 | 3528.2 | 3842.9 | 114.8 |
| `synthetic-4m` (frames) | 1D | 94.0 / 563.6 | 93.8 / 587.0 | 100.7 / 619.3 | 33.1 / 238.4 | 52.7 / 403.6 | 30.2 / 211.0 | 29.6 / 213.2 | 878.9 | 878.0 | 1443.0 | 1445.9 | 94.3 |
| `synthetic-9m` (frames) | 1D | 52.3 / 297.4 | 52.2 / 395.9 | 56.3 / 305.3 | 16.0 / 110.7 | 24.3 / 178.4 | 13.4 / 93.7 | 13.4 / 95.2 | 455.2 | 459.6 | 615.5 | 625.5 | 79.4 |
| `pilatus` (repeated) | 2D `2000x360` | 180.8 / 603.2 | 184.1 / 610.4 | 176.5 / 612.8 | 78.0 / 492.1 | 115.1 / 763.7 | 63.2 / 358.2 | 68.9 / 355.3 | 491.3 | 481.3 | 456.1 | 464.7 | 74.0 |
| `synthetic-4m` (frames) | 2D `2000x360` | 115.1 / 416.1 | 113.9 / 422.6 | 118.1 / 428.9 | 39.2 / 260.4 | 59.4 / 424.2 | 28.2 / 173.7 | 28.4 / 161.7 | 338.9 | 343.8 | 452.8 | 452.6 | 61.7 |
| `synthetic-9m` (frames) | 2D `2000x360` | 63.7 / 257.2 | 63.9 / 257.6 | 66.3 / 263.7 | 22.6 / 133.1 | 34.1 / 271.0 | 14.0 / 81.3 | 13.2 / 73.6 | 250.9 | 249.4 | 347.5 | 346.6 | 55.6 |
| `pilatus` (repeated) | 2D `1000x72` | 880.7 / 2986.4 | 847.4 / 2948.9 | 928.6 / 3024.2 | 209.9 / 1216.8 | 380.2 / 2093.2 | 124.9 / 874.2 | 118.4 / 793.5 | 1548.8 | 1560.4 | 2234.3 | 2514.1 | 92.7 |
| `synthetic-4m` (frames) | 2D `1000x72` | 200.7 / 934.9 | 201.5 / 940.6 | 206.2 / 956.7 | 57.7 / 386.4 | 91.9 / 630.0 | 42.2 / 277.6 | 38.7 / 247.8 | 860.4 | 851.2 | 1361.9 | 1315.8 | 91.0 |
| `synthetic-9m` (frames) | 2D `1000x72` | 86.9 / 461.0 | 86.8 / 462.1 | 89.0 / 472.0 | 24.6 / 157.4 | 38.7 / 300.5 | 17.8 / 110.4 | 16.6 / 96.0 | 446.4 | 451.7 | 647.4 | 644.9 | 80.5 |

#### Primary `float64` pass — Intensity-only mode (median frames/s; CPU shown as `1w / 8w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `dioptrin-pixel_split-metal` | `dioptrin-pyfai_full-metal` | `dioptrin-pixel_split-metal-batch` | `dioptrin-pyfai_full-metal-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 350.9 / 2705.2 | 351.3 / 2696.5 | 359.8 / 2738.0 | 120.1 / 915.5 | 268.0 / 2010.0 | 86.4 / 627.4 | 86.0 / 633.6 | 1569.8 | 1584.4 | 3950.6 | 3381.6 | 112.2 |
| `synthetic-4m` (frames) | 1D | 123.8 / 937.2 | 118.0 / 927.7 | 132.7 / 996.4 | 47.9 / 349.9 | 104.2 / 804.4 | 29.9 / 211.4 | 29.6 / 216.0 | 891.0 | 890.3 | 1417.2 | 1437.2 | 94.6 |
| `synthetic-9m` (frames) | 1D | 69.1 / 516.7 | 69.2 / 509.5 | 74.9 / 550.2 | 23.6 / 165.1 | 48.1 / 357.5 | 13.3 / 92.8 | 13.2 / 91.9 | 462.4 | 464.2 | 622.4 | 619.1 | 78.7 |
| `pilatus` (repeated) | 2D `2000x360` | 280.7 / 1741.8 | 261.4 / 1751.1 | 278.1 / 1782.4 | 121.1 / 861.4 | 249.7 / 1674.2 | 66.8 / 394.6 | 77.3 / 432.0 | 656.6 | 525.3 | 969.2 | 997.0 | 71.4 |
| `synthetic-4m` (frames) | 2D `2000x360` | 137.6 / 742.7 | 138.7 / 746.2 | 144.1 / 767.9 | 59.7 / 425.1 | 122.7 / 885.2 | 28.7 / 183.4 | 31.5 / 189.6 | 488.5 | 496.5 | 834.1 | 836.6 | 66.6 |
| `synthetic-9m` (frames) | 2D `2000x360` | 81.0 / 389.6 | 81.7 / 390.0 | 84.6 / 401.7 | 34.4 / 217.5 | 69.6 / 548.0 | 14.3 / 84.7 | 14.6 / 87.3 | 293.6 | 292.3 | 512.9 | 515.1 | 57.4 |
| `pilatus` (repeated) | 2D `1000x72` | 1237.6 / 4350.1 | 1247.8 / 3392.5 | 1297.6 / 4524.5 | 326.5 / 1888.7 | 809.3 / 4451.6 | 130.1 / 848.2 | 128.9 / 902.4 | 2384.8 | 2411.5 | 3413.0 | 3460.7 | 99.8 |
| `synthetic-4m` (frames) | 2D `1000x72` | 220.5 / 1091.2 | 222.5 / 1098.1 | 226.5 / 1127.3 | 83.1 / 565.3 | 182.5 / 1245.0 | 41.2 / 279.9 | 40.7 / 267.3 | 902.4 | 907.2 | 1442.0 | 1439.1 | 88.8 |
| `synthetic-9m` (frames) | 2D `1000x72` | 105.6 / 544.4 | 106.8 / 545.6 | 109.2 / 557.7 | 35.8 / 233.0 | 77.1 / 603.0 | 17.0 / 105.7 | 17.6 / 107.0 | 458.3 | 459.5 | 659.8 | 660.9 | 77.6 |

#### Optional `float32` pass — Poisson mode (median frames/s; CPU shown as `1w / 8w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `dioptrin-pixel_split-metal` | `dioptrin-pyfai_full-metal` | `dioptrin-pixel_split-metal-batch` | `dioptrin-pyfai_full-metal-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 269.3 / 1962.7 | 269.4 / 1941.5 | 279.0 / 1998.8 | 136.4 / 878.5 | 294.7 / 2030.0 | 116.3 / 663.6 | 114.9 / 670.0 | 2850.3 | 2851.8 | 5872.0 | 5815.6 | 115.2 |
| `synthetic-4m` (frames) | 1D | 93.2 / 625.1 | 93.2 / 704.1 | 100.6 / 760.8 | 48.5 / 320.4 | 95.2 / 613.7 | 31.1 / 224.6 | 31.4 / 223.3 | 1142.5 | 1223.7 | 2336.0 | 2347.2 | 106.5 |
| `synthetic-9m` (frames) | 1D | 51.5 / 390.5 | 51.5 / 278.2 | 55.1 / 357.6 | 22.7 / 157.7 | 41.6 / 313.7 | 13.6 / 96.0 | 13.8 / 94.8 | 697.3 | 626.6 | 1010.0 | 1012.9 | 85.6 |
| `pilatus` (repeated) | 2D `2000x360` | 197.5 / 646.8 | 197.7 / 646.6 | 202.0 / 654.7 | 89.7 / 583.7 | 142.8 / 954.2 | 65.5 / 397.3 | 72.6 / 419.8 | 505.9 | 507.9 | 480.6 | 526.1 | 76.2 |
| `synthetic-4m` (frames) | 2D `2000x360` | 132.0 / 470.6 | 132.4 / 475.6 | 136.1 / 483.1 | 52.0 / 342.4 | 90.4 / 580.2 | 29.3 / 188.7 | 29.7 / 182.5 | 471.8 | 472.6 | 507.7 | 514.9 | 72.8 |
| `synthetic-9m` (frames) | 2D `2000x360` | 74.0 / 303.9 | 74.1 / 305.9 | 76.9 / 312.2 | 31.1 / 182.1 | 54.5 / 407.8 | 14.3 / 88.0 | 13.6 / 83.4 | 384.1 | 381.2 | 442.4 | 442.5 | 65.2 |
| `pilatus` (repeated) | 2D `1000x72` | 885.8 / 3264.2 | 888.8 / 3283.9 | 933.6 / 3371.9 | 324.1 / 1731.4 | 667.4 / 3158.4 | 174.7 / 936.4 | 131.3 / 875.9 | 2463.4 | 2463.1 | 4296.1 | 4249.3 | 110.3 |
| `synthetic-4m` (frames) | 2D `1000x72` | 255.0 / 1151.6 | 256.5 / 1159.4 | 265.1 / 1178.7 | 81.3 / 510.8 | 154.9 / 960.6 | 43.2 / 289.8 | 39.6 / 267.0 | 1272.4 | 1261.8 | 1986.5 | 1975.8 | 105.8 |
| `synthetic-9m` (frames) | 2D `1000x72` | 108.9 / 582.0 | 109.4 / 582.3 | 112.3 / 597.4 | 35.0 / 223.2 | 66.2 / 517.0 | 18.4 / 114.7 | 16.5 / 105.5 | 706.3 | 696.1 | 970.4 | 977.5 | 89.5 |

#### Optional `float32` pass — Intensity-only mode (median frames/s; CPU shown as `1w / 8w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `dioptrin-pixel_split-metal` | `dioptrin-pyfai_full-metal` | `dioptrin-pixel_split-metal-batch` | `dioptrin-pyfai_full-metal-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 361.8 / 2707.7 | 362.9 / 2719.1 | 371.0 / 2788.0 | 177.7 / 1179.3 | 591.1 / 4019.6 | 101.7 / 663.2 | 103.3 / 670.2 | 2793.9 | 2780.2 | 5114.5 | 5104.6 | 115.6 |
| `synthetic-4m` (frames) | 1D | 123.6 / 933.4 | 123.9 / 938.5 | 130.6 / 989.9 | 63.1 / 438.1 | 180.6 / 1184.9 | 31.0 / 224.4 | 31.0 / 222.1 | 1254.8 | 1258.7 | 2308.3 | 2320.9 | 105.6 |
| `synthetic-9m` (frames) | 1D | 66.9 / 504.7 | 67.0 / 498.5 | 72.5 / 549.0 | 30.8 / 218.5 | 82.0 / 624.6 | 13.4 / 94.1 | 13.3 / 94.2 | 670.8 | 630.7 | 966.9 | 972.0 | 86.2 |
| `pilatus` (repeated) | 2D `2000x360` | 289.8 / 1835.8 | 297.0 / 1844.4 | 298.7 / 1838.4 | 139.4 / 965.8 | 316.3 / 2151.9 | 69.4 / 438.6 | 83.5 / 489.2 | 694.8 | 695.1 | 1161.9 | 1167.6 | 76.8 |
| `synthetic-4m` (frames) | 2D `2000x360` | 168.6 / 912.3 | 169.2 / 923.1 | 174.7 / 944.0 | 74.7 / 525.4 | 189.8 / 1335.8 | 30.2 / 191.0 | 32.1 / 198.4 | 627.7 | 642.6 | 1046.3 | 1036.0 | 73.1 |
| `synthetic-9m` (frames) | 2D `2000x360` | 98.2 / 490.9 | 99.1 / 493.2 | 102.8 / 511.3 | 44.2 / 282.8 | 111.6 / 848.8 | 14.6 / 88.9 | 14.9 / 91.0 | 479.3 | 496.4 | 700.8 | 701.8 | 65.5 |
| `pilatus` (repeated) | 2D `1000x72` | 1275.3 / 5099.3 | 1275.9 / 5143.9 | 1325.7 / 5398.9 | 437.2 / 2495.5 | 1392.9 / 6962.9 | 167.4 / 942.1 | 173.2 / 954.7 | 2811.6 | 2817.3 | 5485.8 | 5458.6 | 112.2 |
| `synthetic-4m` (frames) | 2D `1000x72` | 265.4 / 1375.9 | 243.4 / 1376.9 | 268.5 / 1413.4 | 98.1 / 724.6 | 254.6 / 1941.2 | 41.6 / 289.0 | 40.5 / 285.5 | 1281.9 | 1292.8 | 2120.5 | 2144.0 | 102.9 |
| `synthetic-9m` (frames) | 2D `1000x72` | 129.9 / 697.1 | 131.1 / 696.9 | 134.9 / 712.9 | 47.6 / 312.0 | 131.5 / 1032.4 | 17.8 / 109.6 | 18.0 / 113.3 | 745.5 | 764.1 | 982.3 | 984.7 | 91.4 |
