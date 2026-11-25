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
