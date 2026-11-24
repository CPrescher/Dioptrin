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
- Raw `benchmark-results/<timestamp>/` directories stay out of Git (per `AGENTS.md`); summarize final measured tables and hardware conditions here once all three runs complete.
