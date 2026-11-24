# Final Paper Benchmarks (2026-11-24, Dioptrin v0.15.2)

Unified cross-machine protocol comparing **Dioptrin (CPU + GPU)**, **pyFAI (CPU + OpenCL GPU)**, and **azint (CPU)** using identical fixtures, radial bins, 2D cake grids (`2000x360`, `1000x72`, χ in `[-180, 180)` deg), masks, polarization (`0.965`), solid-angle correction, and numerical verification across all three machines.

## Shared Settings

- **Dioptrin version:** `v0.15.2` (includes embedded `compute_70` PTX for driver-only CUDA on Volta+ GPUs, and Metal on macOS).
- **Dimensions:** `1d` and `2d` (`--dimensions 1d 2d`).
- **2D cakes:** `2000x360` and `1000x72` (default `--cakes`).
- **GPU precision:** `f32_compensated` (default `--gpu-precision`, matches pyFAI OpenCL's compensated single-precision accumulation + float64 division/output, and works identically on both CUDA and Apple Silicon Metal).
- **Input dtype:**
  - Primary pass: `float64` (default `--input-dtype float64`).
  - Optional native-f32 pass: `--input-dtype float32` (halves host-to-device and CPU memory bandwidth).
  - Optional CUDA f64 ablation (CUDA only): `--gpu-precision f64`.

---

## 1. Windows Workstation (RTX 5070 Ti, native Windows)

Open a **Visual Studio Developer PowerShell** (MSVC `cl` + Rust MSVC toolchain; no CUDA Toolkit needed):

```powershell
python benchmarks/comparison/setup.py --with-cuda --with-opencl
```

Run the full 1D + 2D CPU & GPU suite:

```powershell
.venv-benchmark\Scripts\python.exe -E -s -B benchmarks/comparison/compare.py `
  --dimensions 1d 2d `
  --opencl-device gpu `
  --cuda-device 0 `
  --gpu-precision f32_compensated `
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full `
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl `
             dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda `
             dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch
```

Optional `float32` input pass (append `--input-dtype float32` and `--output benchmark-results/2026-11-24-windows-f32`):

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

---

## 2. Maxwell Cluster (DESY, Linux + NVIDIA GPU)

With `v0.15.2`'s embedded `compute_70` PTX, no CUDA Toolkit module is required on Volta+ GPUs (only the node's NVIDIA display driver). Submit from the repository root:

```bash
WITH_OPENCL=1 WITH_CUDA=1 \
  sbatch --partition=<gpu-partition> benchmarks/comparison/maxwell.sbatch \
  --dimensions 1d 2d \
  --opencl-device gpu \
  --cuda-device 0 \
  --gpu-precision f32_compensated \
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full \
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl \
             dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda \
             dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch
```

*(For an interactive allocation or non-Slurm Linux node, run `python3 benchmarks/comparison/setup.py --with-cuda --with-opencl` followed by `.venv-benchmark/bin/python benchmarks/comparison/compare.py ...` with the same flags.)*

---

## 3. MacBook (Apple Silicon, macOS + Metal)

Install GCC via Homebrew (`brew install gcc`, required for azint OpenMP) and build the benchmark environment with Metal + OpenCL:

```bash
python3 benchmarks/comparison/setup.py --with-metal --with-opencl
```

Run the full 1D + 2D CPU & GPU suite:

```bash
.venv-benchmark/bin/python benchmarks/comparison/compare.py \
  --dimensions 1d 2d \
  --opencl-device gpu \
  --metal-device 0 \
  --gpu-precision f32_compensated \
  --variants dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full \
             azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl \
             dioptrin-pixel_split-metal dioptrin-pyfai_full-metal \
             dioptrin-pixel_split-metal-batch dioptrin-pyfai_full-metal-batch
```

---

## Notes

- Run `--suite quick` first on each machine to verify installation and GPU detection before launching the full `standard` suite.
- Raw `benchmark-results/<timestamp>/` directories stay out of Git (per `AGENTS.md`); summarize final measured tables and hardware conditions here once all three runs complete.
