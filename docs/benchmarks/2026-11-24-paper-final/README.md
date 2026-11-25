# Final Paper Benchmarks (2026-11-24, Dioptrin v0.15.2)

Cross-machine protocol comparing **Dioptrin (CPU + GPU)**, **pyFAI (CPU + OpenCL GPU)**, and **azint (CPU)** using identical fixtures, radial bins, 2D cake grids (`2000x360`, `1000x72`, χ in `[-180, 180)` deg), masks, polarization (`0.965`), solid-angle correction, and numerical verification across all three machines.

The rerun of these benchmarks with Dioptrin 0.16.0, with one run script per machine, is described in [`../2026-11-25-paper-0.16.0/README.md`](../2026-11-25-paper-0.16.0/README.md).

## Shared Settings & Platform Caveats

- **Dioptrin version:** `v0.15.2` (plus PR #86 comparison harness updates; record the exact `git rev-parse HEAD` in `environment.json`).
- **Dimensions:** `1d` and `2d` (`--dimensions 1d 2d`).
- **2D cakes:** `2000x360` and `1000x72` (default `--cakes`).
- **GPU precision:** `f32_compensated` (default `--gpu-precision`): compensated single-precision accumulation with float64 normalization/output (comparable to pyFAI OpenCL's compensated float32 accumulation, and supported identically on both CUDA and Apple Silicon Metal).
- **Input dtype:**
  - Primary pass: `float64` (default `--input-dtype float64`).
  - Optional native-f32 pass: `--input-dtype float32` (asserted lossless on the integer-count fixtures). Note that Dioptrin consumes `float32` natively without upcasting, pyFAI's CSR/OpenCL engines convert to `float32` internally anyway (so `float32` saves a host conversion), and `azint` may convert `float32` to `float64` depending on its binding.
  - Optional CUDA `f64` ablation (CUDA only): `--gpu-precision f64`.
- **Timed Dioptrin calls:** CPU 1D rows time the library's default `batch1d(frames, n, errors=...)` on the global Rayon pool of the CPU budget (`--dioptrin-1d-path global`, the default; `split` keeps the earlier `num_workers=B, num_threads=1` call), and every Dioptrin row (CPU 1D/2D, CUDA, Metal) ends its timed call with NumPy float64 intensity (and Poisson sigma) for every frame, as pyFAI's and azint's calls do. Results record `dioptrin_1d_path` and `dioptrin_result_conversion` (`array` for the zero-copy accessors, `list` for `np.asarray` of lists). See `benchmarks/comparison/README.md` ("Dioptrin CPU 1D call", "Result conversion"). **The measured results below predate this:** their CPU 1D Dioptrin rows timed `batch1d(..., num_workers=B, num_threads=1)`, which in 0.15.2 is the slower `split` path, and no Dioptrin row timed the conversion of its results to NumPy arrays. Reruns under the new harness need a Dioptrin build that has the `*_array` accessors (unreleased at the time of writing; 0.15.2 wheels lack them): the harness refuses a build without them unless `--allow-list-conversion` is given, and then times list building rather than integration.
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

### Cross-Machine Snapshot (`float64`, Poisson, median frames/s)

| Dataset / Mode | Machine | `dioptrin-pyfai_full` (8 CPU) | `dioptrin-pyfai_full-gpu` (per-frame) | `dioptrin-pyfai_full-gpu-batch` | `azint-cached` (8 CPU) | `pyfai-full` (8 CPU) | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` 1D | Windows (9800X3D + RTX 5070 Ti) | 1538.7 *(2061.2 @16w)* | 2858.9 | **4273.9** | 1717.0 *(2543.9 @16w)* | 243.2 | 707.9 |
| `pilatus` 1D | Maxwell (EPYC 9475F + H200 NVL) | 1899.8 | 4602.6 | **8014.5** | 1395.8 | 487.0 | 1789.0 |
| `pilatus` 1D | MacBook (M4 Pro + Metal) | 1931.6 | 1672.6 | **3842.9** | 1002.8 | 647.2 | 114.8 |
| `synthetic-9m` 1D | Windows (9800X3D + RTX 5070 Ti) | 225.8 *(318.2 @16w)* | 277.6 | **300.6** | 149.5 *(207.8 @16w)* | 24.8 | 92.3 |
| `synthetic-9m` 1D | Maxwell (EPYC 9475F + H200 NVL) | 390.4 | 378.0 | **434.4** | 103.0 | 55.5 | 200.9 |
| `synthetic-9m` 1D | MacBook (M4 Pro + Metal) | 395.9 | 459.6 | **625.5** | 178.4 | 93.7 | 79.4 |
| `synthetic-9m` 2D `2000x360` | Windows (9800X3D + RTX 5070 Ti) | 80.6 | 156.3 | **179.2** | 147.6 *(185.2 @16w)* | — *(bbox: 22.4)* | 60.3 |
| `synthetic-9m` 2D `2000x360` | Maxwell (EPYC 9475F + H200 NVL) | 157.6 | 275.3 | **341.0** | 128.9 | 64.1* | 172.7 |
| `synthetic-9m` 2D `2000x360` | MacBook (M4 Pro + Metal) | 257.6 | 249.4 | **346.6** | 271.0 | 81.3 | 55.6 |
| `synthetic-9m` 2D `1000x72` | Windows (9800X3D + RTX 5070 Ti) | 174.7 | 260.1 | 285.3 | **316.3** *(327.8 @16w)* | — *(bbox: 26.7)* | 99.3 |
| `synthetic-9m` 2D `1000x72` | Maxwell (EPYC 9475F + H200 NVL) | **452.1** | 368.8 | 420.8 | 273.5 | 95.1 | 205.6 |
| `synthetic-9m` 2D `1000x72` | MacBook (M4 Pro + Metal) | 462.1 | 451.7 | **644.9** | 300.5 | 110.4 | 80.5 |

*All three GPUs (RTX 5070 Ti CUDA, H200 NVL CUDA, Apple M4 Pro Metal) achieve identical `f32_compensated` accuracy vs CPU counterpart to 3 significant digits: max relative L2 ≤ `5.70e-9` (1D intensity), ≤ `1.77e-8` (1D sigma), ≤ `1.98e-8` (2D intensity), ≤ `1.90e-8` (2D sigma).*

### Windows Workstation (AMD Ryzen 7 9800X3D + NVIDIA GeForce RTX 5070 Ti, native Windows)

- **Revision:** `118a299` (`dioptrin 0.15.2`, `python,cuda` loading embedded `compute_70` PTX via display driver only — `nvrtc_version: null`, no CUDA Toolkit installed)
- **Hardware / OS:** AMD Ryzen 7 9800X3D (8 physical cores / 16 logical CPUs with SMT), 31.2 GiB RAM, NVIDIA GeForce RTX 5070 Ti (16 GiB, Blackwell Compute Capability 12.0, driver `616.64`, CUDA 13.4), `Windows-11-10.0.26200`
- **Toolchain & libraries:** Python 3.14.10, `rustc 1.100.0` (`RUSTFLAGS="-C target-cpu=x86-64-v3"`), MSVC `19.52.37029` (`CL="/arch:AVX2"`), `numpy 2.4.2`, `pyFAI 2026.9.0` (OpenMP via MSVC `_vcomp_fork`), `pyopencl 2026.1.5` (NVIDIA CUDA OpenCL 3.0), `azint 1.2.6.dev6+g4f78cc3b1` (`azint-cached` bit-identical to stock `azint`).
- **Suite settings & completeness:** Split across 1D/2D + SMT extensions and ablations: `windows-f64-1d` (370 rows, 1–8w), `windows-f64-1d-smt` (240 rows, 1/12/16w), `windows-f64-2d` (580 rows, 1–8w), `windows-f64-2d-smt` (360 rows, 1/12/16w), `windows-f32-1d` (530 rows, 1–16w), `windows-f32-2d` (820 rows, 1–16w), and `windows-cuda-f64` (120 rows, 1D+2D GPU precision `f64`). **0 failures, 0 skips, and 0 non-`ok` numerical rows** across all 3,020 rows. In 2D (`windows-f64-2d`, `windows-f32-2d`), stock `pyfai-full` / `pyfai-full-omp` were omitted (leaving `pyfai-bbox` and `pyfai-full-opencl`).
- **Accuracy (`f32_compensated` vs CPU counterpart):** Dioptrin CUDA max relative L2 ≤ `5.70e-9` (1D intensity), ≤ `1.77e-8` (1D sigma), ≤ `1.98e-8` (2D intensity), ≤ `1.90e-8` (2D sigma); `pyfai-full-opencl` 1D max relative L2 ≤ `6.67e-8` (intensity), ≤ `7.37e-8` (sigma).
- **Compact records archive:** `2026-11-24-windows-{f64-1d,f64-1d-smt,f64-2d,f64-2d-smt,f32-1d,f32-2d,cuda-f64}` and `windows-build-record` saved outside Git (Dropbox `Science/4 - Scientific Projects/Software/Dioptrin/benchmarks/2026-11-24-paper-final/`).

#### Windows `float64` pass — Poisson mode (median frames/s; CPU shown as `1w / 8w / 16w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `pyfai-full-omp` | `dioptrin-pixel_split-cuda` | `dioptrin-pyfai_full-cuda` | `dioptrin-pixel_split-cuda-batch` | `dioptrin-pyfai_full-cuda-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 272.7 / 1608.6 / 2007.8 | 274.0 / 1538.7 / 2061.2 | 283.8 / 1641.6 / 2083.1 | 103.8 / 469.8 / 608.1 | 260.5 / 1717.0 / 2543.9 | 73.9 / 243.2 / 319.0 | 72.6 / 246.0 / 336.8 | 72.3 / 96.9 / 78.9 | 2905.5 | 2858.9 | 4222.6 | 4273.9 | 707.9 |
| `synthetic-4m` (frames) | 1D | 84.0 / 474.2 / 636.8 | 85.7 / 469.8 / 643.4 | 93.5 / 514.2 / 680.8 | 27.5 / 127.3 / 154.6 | 40.5 / 351.0 / 451.3 | 12.9 / 62.2 / 76.2 | 12.7 / 64.8 / 78.6 | 12.7 / 22.8 / 18.4 | 673.4 | 665.0 | 782.4 | 773.4 | 234.5 |
| `synthetic-9m` (frames) | 1D | 44.3 / 238.1 / 380.9 | 46.3 / 225.8 / 318.2 | 50.7 / 237.7 / 363.7 | 12.0 / 51.5 / 64.4 | 18.6 / 149.5 / 207.8 | 5.3 / 24.8 / 32.4 | 5.4 / 25.2 / 35.6 | 5.4 / 7.9 / 8.1 | 280.2 | 277.6 | 296.1 | 300.6 | 92.3 |
| `pilatus` (repeated) | 2D `2000x360` | 70.3 / 156.3 / 167.9 | 70.5 / 159.6 / 165.0 | 72.8 / 158.6 / 167.9 | 52.7 / 206.6 / 245.0 | 94.0 / 382.4 / 469.7 | — | 35.6 / 126.2 / 158.0 | — | 299.7 | 296.1 | 356.9 | 345.0 | 123.6 |
| `synthetic-4m` (frames) | 2D `2000x360` | 43.9 / 119.5 / 103.0 | 44.3 / 117.7 / 111.8 | 44.9 / 119.8 / 117.6 | 25.6 / 97.4 / 111.6 | 38.8 / 254.3 / 253.8 | — | 11.9 / 49.4 / 56.4 | — | 230.9 | 228.3 | 269.5 | 269.8 | 94.5 |
| `synthetic-9m` (frames) | 2D `2000x360` | 30.3 / 80.6 / 79.8 | 31.6 / 80.6 / 81.3 | 31.9 / 81.7 / 80.2 | 13.8 / 49.3 / 56.4 | 21.9 / 147.6 / 185.2 | — | 5.7 / 22.4 / 27.8 | — | 161.3 | 156.3 | 182.6 | 179.2 | 60.3 |
| `pilatus` (repeated) | 2D `1000x72` | 491.9 / 1210.3 / 1171.6 | 481.4 / 1226.3 / 1183.3 | 494.5 / 1243.0 / 1200.5 | 143.8 / 583.1 / 656.7 | 418.2 / 2277.0 / 2456.8 | — | 69.5 / 237.5 / 294.8 | — | 1417.5 | 1451.4 | 2185.6 | 2227.5 | 442.2 |
| `synthetic-4m` (frames) | 2D `1000x72` | 135.6 / 370.8 / 383.0 | 132.8 / 374.2 / 379.0 | 136.4 / 381.7 / 390.3 | 47.5 / 153.6 / 173.6 | 91.8 / 691.3 / 773.4 | — | 16.4 / 64.3 / 72.0 | — | 547.2 | 503.8 | 639.4 | 649.0 | 210.8 |
| `synthetic-9m` (frames) | 2D `1000x72` | 74.2 / 175.9 / 171.8 | 73.8 / 174.7 / 170.4 | 75.2 / 176.0 / 174.6 | 20.7 / 72.0 / 75.9 | 46.1 / 316.3 / 327.8 | — | 7.2 / 26.7 / 32.6 | — | 267.7 | 260.1 | 288.9 | 285.3 | 99.3 |

#### Windows `float64` pass — Intensity-only mode (median frames/s; CPU shown as `1w / 8w / 16w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `pyfai-full-omp` | `dioptrin-pixel_split-cuda` | `dioptrin-pyfai_full-cuda` | `dioptrin-pixel_split-cuda-batch` | `dioptrin-pyfai_full-cuda-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 357.5 / 2156.7 / 2700.8 | 358.9 / 2134.6 / 2750.7 | 380.7 / 2236.2 / 2828.0 | 148.5 / 670.0 / 799.2 | 536.9 / 3363.5 / 5021.2 | 75.5 / 237.6 / 327.5 | 76.2 / 256.7 / 308.1 | 74.7 / 97.1 / 77.1 | 2742.5 | 2935.3 | 4643.3 | 4465.6 | 706.0 |
| `synthetic-4m` (frames) | 1D | 118.7 / 615.2 / 823.9 | 117.4 / 599.8 / 922.5 | 126.2 / 654.6 / 1050.2 | 39.7 / 175.1 / 207.2 | 76.0 / 698.3 / 907.8 | 12.9 / 59.2 / 73.6 | 13.1 / 64.1 / 81.2 | 13.2 / 21.8 / 17.9 | 678.2 | 669.7 | 787.5 | 785.2 | 232.3 |
| `synthetic-9m` (frames) | 1D | 64.4 / 288.8 / 408.6 | 65.5 / 282.6 / 364.9 | 69.7 / 313.7 / 442.5 | 18.2 / 70.0 / 84.9 | 39.1 / 294.3 / 441.8 | 5.8 / 25.6 / 33.6 | 5.5 / 25.4 / 35.4 | 5.4 / 7.8 / 8.1 | 285.2 | 283.4 | 300.7 | 295.9 | 92.6 |
| `pilatus` (repeated) | 2D `2000x360` | 143.5 / 806.2 / 956.8 | 168.8 / 795.2 / 898.1 | 153.3 / 772.5 / 890.9 | 78.2 / 363.5 / 477.2 | 211.7 / 978.2 / 1253.9 | — | 42.1 / 148.8 / 202.8 | — | 772.1 | 740.3 | 1002.7 | 945.4 | 121.0 |
| `synthetic-4m` (frames) | 2D `2000x360` | 72.8 / 249.8 / 240.8 | 72.2 / 264.3 / 242.9 | 74.5 / 274.7 / 261.5 | 37.7 / 152.3 / 174.4 | 74.6 / 587.9 / 675.8 | — | 13.5 / 62.3 / 79.1 | — | 410.5 | 412.8 | 497.5 | 498.7 | 88.0 |
| `synthetic-9m` (frames) | 2D `2000x360` | 45.7 / 129.3 / 124.4 | 45.3 / 130.4 / 130.2 | 47.2 / 133.9 / 128.0 | 19.8 / 71.2 / 80.9 | 43.4 / 330.5 / 400.6 | — | 6.4 / 28.8 / 38.6 | — | 233.3 | 234.1 | 260.0 | 259.1 | 62.3 |
| `pilatus` (repeated) | 2D `1000x72` | 823.2 / 3850.7 / 3996.4 | 823.9 / 3847.3 / 3932.5 | 847.7 / 3806.7 / 3940.9 | 204.1 / 781.4 / 882.6 | 884.7 / 4602.0 / 5537.0 | — | 84.4 / 323.7 / 396.3 | — | 2129.2 | 2081.6 | 3282.1 | 3275.3 | 462.9 |
| `synthetic-4m` (frames) | 2D `1000x72` | 162.8 / 483.8 / 505.1 | 160.6 / 500.7 / 505.8 | 164.9 / 504.2 / 521.8 | 60.3 / 203.2 / 235.7 | 151.5 / 1333.7 / 1519.4 | — | 19.4 / 89.8 / 94.2 | — | 609.1 | 616.3 | 721.1 | 728.5 | 219.3 |
| `synthetic-9m` (frames) | 2D `1000x72` | 86.4 / 199.2 / 190.0 | 85.1 / 196.1 / 190.2 | 86.8 / 198.1 / 192.2 | 26.3 / 91.8 / 94.1 | 86.8 / 654.7 / 721.6 | — | 8.3 / 36.7 / 45.2 | — | 273.0 | 281.2 | 298.3 | 293.1 | 96.0 |

#### Windows `float32` pass — Poisson mode (median frames/s; CPU shown as `1w / 8w / 16w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `pyfai-full-omp` | `dioptrin-pixel_split-cuda` | `dioptrin-pyfai_full-cuda` | `dioptrin-pixel_split-cuda-batch` | `dioptrin-pyfai_full-cuda-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 275.1 / 1573.6 / 1925.3 | 272.1 / 1563.0 / 1926.3 | 284.2 / 1615.6 / 1946.6 | 126.6 / 603.4 / 872.1 | 187.8 / 1493.9 / 2439.1 | 62.8 / 277.1 / 357.8 | 67.2 / 278.3 / 357.6 | 59.5 / 102.1 / 84.0 | 3688.3 | 3641.8 | 4888.1 | 4857.5 | 1353.4 |
| `synthetic-4m` (frames) | 1D | 83.9 / 477.8 / 684.8 | 86.3 / 479.5 / 668.3 | 94.3 / 519.7 / 774.6 | 34.7 / 177.3 / 209.4 | 51.8 / 442.6 / 648.3 | 15.1 / 70.3 / 88.1 | 15.1 / 72.6 / 88.4 | 15.0 / 26.1 / 19.6 | 967.8 | 958.0 | 1264.8 | 1273.5 | 564.6 |
| `synthetic-9m` (frames) | 1D | 46.0 / 244.5 / 351.3 | 47.0 / 250.1 / 369.3 | 51.4 / 272.3 / 359.3 | 14.7 / 81.1 / 97.6 | 23.8 / 210.0 / 268.3 | 6.0 / 32.9 / 37.6 | 6.0 / 33.2 / 41.5 | 6.0 / 10.8 / 8.6 | 403.4 | 437.6 | 524.9 | 516.0 | 278.1 |
| `pilatus` (repeated) | 2D `2000x360` | 79.8 / 174.2 / 172.9 | 76.4 / 176.7 / 179.5 | 74.6 / 177.4 / 175.4 | 58.7 / 265.7 / 297.6 | 93.8 / 440.2 / 419.1 | — | 43.7 / 166.0 / 167.5 | — | 316.3 | 325.1 | 401.9 | 397.8 | 141.1 |
| `synthetic-4m` (frames) | 2D `2000x360` | 50.8 / 121.2 / 118.5 | 50.7 / 131.1 / 118.9 | 52.2 / 131.8 / 120.5 | 30.2 / 119.4 / 137.5 | 44.3 / 282.7 / 291.9 | — | 13.0 / 57.2 / 62.9 | — | 256.2 | 252.7 | 315.7 | 299.0 | 124.4 |
| `synthetic-9m` (frames) | 2D `2000x360` | 35.2 / 94.2 / 90.0 | 35.6 / 94.0 / 85.1 | 36.0 / 97.1 / 90.4 | 15.0 / 59.9 / 83.9 | 25.0 / 185.9 / 183.5 | — | 6.5 / 30.1 / 34.0 | — | 198.7 | 200.3 | 236.4 | 234.5 | 101.0 |
| `pilatus` (repeated) | 2D `1000x72` | 507.0 / 1237.1 / 1138.8 | 502.6 / 1252.8 / 1214.7 | 527.9 / 1263.0 / 1217.6 | 188.4 / 769.7 / 884.2 | 401.8 / 2176.6 / 2303.0 | — | 80.4 / 288.9 / 345.4 | — | 1549.2 | 1558.2 | 2049.4 | 1995.7 | 680.4 |
| `synthetic-4m` (frames) | 2D `1000x72` | 148.1 / 492.8 / 489.2 | 147.9 / 490.4 / 482.8 | 151.4 / 501.4 / 485.1 | 51.1 / 244.5 / 269.1 | 92.1 / 688.5 / 900.9 | — | 18.0 / 84.7 / 90.1 | — | 699.0 | 724.5 | 882.5 | 898.6 | 420.6 |
| `synthetic-9m` (frames) | 2D `1000x72` | 95.0 / 220.7 / 217.7 | 94.6 / 217.4 / 212.9 | 94.8 / 219.6 / 225.5 | 25.3 / 105.0 / 119.1 | 54.2 / 391.3 / 412.7 | — | 8.3 / 37.8 / 41.9 | — | 369.2 | 369.6 | 459.4 | 458.0 | 253.5 |

#### Windows `float32` pass — Intensity-only mode (median frames/s; CPU shown as `1w / 8w / 16w`)

| Dataset (workload) | Mode | `dioptrin-pixel_split` | `dioptrin-pyfai_full` | `dioptrin-supersampled` | `azint` | `azint-cached` | `pyfai-full` | `pyfai-bbox` | `pyfai-full-omp` | `dioptrin-pixel_split-cuda` | `dioptrin-pyfai_full-cuda` | `dioptrin-pixel_split-cuda-batch` | `dioptrin-pyfai_full-cuda-batch` | `pyfai-full-opencl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 380.6 / 2186.4 / 2633.7 | 374.5 / 2239.3 / 2737.5 | 394.1 / 2133.7 / 2838.3 | 175.0 / 868.2 / 1148.4 | 479.7 / 3109.9 / 4370.7 | 64.5 / 276.3 / 346.5 | 62.9 / 269.0 / 376.8 | 61.4 / 96.0 / 82.0 | 3168.6 | 3562.7 | 5004.6 | 5039.0 | 1229.2 |
| `synthetic-4m` (frames) | 1D | 115.1 / 639.1 / 855.5 | 117.6 / 638.5 / 887.4 | 128.0 / 719.4 / 1047.9 | 46.8 / 242.6 / 299.2 | 90.7 / 834.9 / 1216.6 | 15.9 / 72.5 / 88.7 | 15.5 / 72.2 / 93.6 | 15.7 / 24.8 / 19.1 | 1004.2 | 958.4 | 1261.9 | 1287.7 | 527.7 |
| `synthetic-9m` (frames) | 1D | 65.2 / 304.1 / 454.7 | 65.2 / 297.3 / 416.0 | 68.8 / 341.3 / 403.0 | 21.7 / 105.5 / 126.5 | 44.1 / 416.9 / 592.0 | 6.1 / 33.1 / 38.6 | 6.1 / 33.5 / 41.9 | 6.1 / 10.7 / 8.5 | 420.0 | 432.9 | 532.7 | 526.7 | 269.5 |
| `pilatus` (repeated) | 2D `2000x360` | 187.8 / 870.7 / 883.7 | 188.1 / 910.2 / 924.6 | 194.1 / 903.6 / 936.8 | 92.8 / 470.8 / 547.0 | 238.4 / 1123.2 / 1265.5 | — | 48.1 / 193.3 / 207.6 | — | 821.5 | 813.2 | 1103.7 | 1037.9 | 142.3 |
| `synthetic-4m` (frames) | 2D `2000x360` | 89.8 / 308.9 / 281.9 | 89.7 / 325.4 / 286.4 | 92.1 / 343.1 / 311.0 | 43.9 / 197.2 / 240.8 | 86.6 / 662.1 / 705.7 | — | 15.0 / 71.1 / 82.8 | — | 507.1 | 492.9 | 640.7 | 630.0 | 124.5 |
| `synthetic-9m` (frames) | 2D `2000x360` | 52.4 / 154.7 / 149.0 | 52.5 / 158.1 / 152.5 | 53.7 / 161.0 / 154.6 | 22.1 / 104.1 / 110.1 | 47.3 / 413.6 / 409.1 | — | 7.1 / 36.3 / 43.6 | — | 314.2 | 318.8 | 383.0 | 384.3 | 99.4 |
| `pilatus` (repeated) | 2D `1000x72` | 843.5 / 4074.7 / 3827.8 | 835.3 / 4130.0 / 3883.9 | 901.7 / 4361.0 / 4137.6 | 265.3 / 1085.6 / 1140.4 | 817.7 / 4362.1 / 5279.1 | — | 94.6 / 377.5 / 423.2 | — | 2326.3 | 2317.8 | 3304.3 | 3293.3 | 727.2 |
| `synthetic-4m` (frames) | 2D `1000x72` | 178.9 / 742.7 / 722.3 | 178.1 / 745.1 / 733.1 | 182.0 / 777.7 / 779.9 | 65.0 / 340.0 / 361.0 | 170.9 / 1331.9 / 1778.5 | — | 21.7 / 99.3 / 118.2 | — | 823.1 | 841.2 | 1113.3 | 1098.7 | 412.7 |
| `synthetic-9m` (frames) | 2D `1000x72` | 108.9 / 248.2 / 247.7 | 108.8 / 248.0 / 244.3 | 109.0 / 252.7 / 256.7 | 32.4 / 136.3 / 167.0 | 98.8 / 782.9 / 883.0 | — | 9.4 / 45.9 / 52.6 | — | 418.4 | 401.6 | 467.5 | 491.3 | 258.9 |

#### Windows CUDA `f64` precision ablation (`float64` inputs, `f32_compensated` → `f64`, median frames/s)

| Dataset (workload) | Mode | `pixel_split-cuda` (Poisson) | `pyfai_full-cuda` (Poisson) | `pixel_split-cuda-batch` (Poisson) | `pyfai_full-cuda-batch` (Poisson) | `pyfai_full-cuda-batch` (Intensity) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pilatus` (repeated) | 1D | 2905.5 → 1865.4 | 2858.9 → 1841.7 | 4222.6 → 2572.9 | 4273.9 → 2495.3 | 4465.6 → 2664.5 |
| `synthetic-4m` (frames) | 1D | 673.4 → 485.0 | 665.0 → 502.2 | 782.4 → 603.7 | 773.4 → 608.2 | 785.2 → 608.2 |
| `synthetic-9m` (frames) | 1D | 280.2 → 209.4 | 277.6 → 226.9 | 296.1 → 261.8 | 300.6 → 270.8 | 295.9 → 269.4 |
| `pilatus` (repeated) | 2D `2000x360` | 299.7 → 289.6 | 296.1 → 288.6 | 356.9 → 370.2 | 345.0 → 369.3 | 945.4 → 957.6 |
| `synthetic-4m` (frames) | 2D `2000x360` | 230.9 → 193.6 | 228.3 → 195.5 | 269.5 → 245.6 | 269.8 → 245.3 | 498.7 → 412.3 |
| `synthetic-9m` (frames) | 2D `2000x360` | 161.3 → 133.4 | 156.3 → 133.7 | 182.6 → 160.5 | 179.2 → 160.7 | 259.1 → 225.4 |
| `pilatus` (repeated) | 2D `1000x72` | 1417.5 → 1110.1 | 1451.4 → 1132.6 | 2185.6 → 1528.7 | 2227.5 → 1556.8 | 3275.3 → 2120.7 |
| `synthetic-4m` (frames) | 2D `1000x72` | 547.2 → 412.0 | 503.8 → 420.9 | 639.4 → 509.0 | 649.0 → 510.0 | 728.5 → 571.6 |
| `synthetic-9m` (frames) | 2D `1000x72` | 267.7 → 208.2 | 260.1 → 207.5 | 288.9 → 246.9 | 285.3 → 250.4 | 293.1 → 261.0 |

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
