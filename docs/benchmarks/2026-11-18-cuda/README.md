# Native CUDA fixed-mask 1D integration

The follow-up [pyFAI CPU and OpenCL comparison](PYFAI.md) uses
the same workstation and real fixtures.

The prototype gathers each bin from Dioptrin’s prepared CSR table. The CPU
applies the fixed mask, builds the table and computes its f64 denominators.
CUDA uploads that table once and uses the same kernels for `counting`,
`pixel_split` and `pyfai_full`. Each launch handles up to 256 frames. Both
intensity-only and Poisson standard-deviation kernels return zero for bins
with zero normalization.

All three accumulation policies read f32 frames and weights. Plain f32
uses explicit fused multiply-add in the per-thread loop. Compensated f32
retains product residuals with explicit FMA and merges high/low pairs within
and across warps, as in the Metal prototype. The double policy widens the
operands before multiplication and accumulates in f64. Each policy uses the
CPU’s f64 denominator and returns f64 output, so the comparison includes
input quantization and the arithmetic of the selected policy. NVRTC compiles
with implicit FMA contraction disabled and without fast math.

The two GPU layouts assign either a 128-thread block to one bin or each
of four warps in a block to its own bin. The warp layout eliminates block
synchronization and reduces the number of blocks for short rows. The block
layout gives longer rows four times as many threads. Both consume the same CSR
table. CSC scatter would need contended atomics for signal and variance and
would complicate the compensated reduction.

## Reproduce

The benchmark is an ignored test under the optional `cuda` feature. The
driver and NVRTC libraries load at runtime. The default crate, documentation
and Python builds have no CUDA dependency. On the RTX 5070 Ti, native
`sm_120` support starts with CUDA 12.8, and NVIDIA introduced support for
the card in Windows driver 572.47. For runtime PTX compilation, use an
NVRTC release supported by the installed driver.
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html),
[RTX 5070 Ti launch driver](https://www.nvidia.com/en-au/geforce/news/geforce-rtx-5070-ti-game-ready-driver/).

```sh
cargo test --features cuda --lib integration::cuda::tests -- --nocapture
python docs/benchmarks/2026-11-18-cuda/run.py --out benchmark-results/cuda/windows.json
```

Set `DIOPTRIN_REQUIRE_CUDA=1` to require a working GPU and NVRTC for the
correctness test. Pass `--nvrtc` with the directory containing the NVRTC
libraries if they are outside the normal library search path. The benchmark
requires a GPU and writes samples and metadata to JSON at the chosen path.
Use `--cases pilatus,eiger` with the matching `eiger_0000.tiff`,
`eiger_0000.poni` and `eiger_0000.mask` in `test_data/`, or use
`--eiger-dir` to point at another fixture directory. The same Rust TIFF
loader orients both image and mask. The Eiger PONI takes its pixel sizes
from the derived calibration used by Metal. Use `eiger-synthetic` to
exercise the Eiger dimensions, geometry and mask with generated signal.
Use a square side such as `256` as a standalone synthetic case.

For the recorded matrix, use the following `run.py` arguments with a
distinct `--out` for each row. The standard sweeps use both CUDA layouts
and CPU thread counts 1, 4, 8 and 16. Capacity is recalculated from the
current device memory each time.

| Cases | Methods | Batches | Layouts | CPU threads | Repeats |
| --- | --- | --- | --- | --- | --- | ---: |
| pilatus | all | 1,8,64,256 | block,warp | 1,4,8,16 | 5 |
| eiger | all | 1,8,64 | block,warp | 1,4,8,16 | 3 |
| 256 | all | 1,8,64,256 | block,warp | 1,4,8,16 | 3 |
| eiger | pixel_split,pyfai_full | 256 | block | 16 | 3 |
| eiger | counting | 256 | warp | 16 | 3 |
| eiger | pixel_split | max | block | 16 | 3 |
| pilatus | pixel_split | max | warp | 16 | 3 |

## Measurement method

- Native Windows 11 Pro, build 26200, on an AMD Ryzen 7 9800X3D
  (8 cores, 16 logical processors), 32 GB DDR5 and an NVIDIA GeForce
  RTX 5070 Ti (16 GB). Driver 616.64 reported CUDA 13.4. Compilation
  used NVIDIA’s CUDA 13.4.59 NVRTC redistributable directly. The GPU drove
  the Windows desktop during the run. NVIDIA’s compute-process query
  listed no competing compute jobs before measurement.
  [NVIDIA NVRTC redistributables](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/).
- Each case uses 3,000 two-theta bins in degrees, a fixed mask and polarization
  factor 0.965. For real images the fixed mask includes negative and
  nonfinite samples, matching the Metal fixture. Synthetic signals are
  deterministic and include fractional values, negative samples and sparse
  high-intensity pixels. Each timed batch contains contiguous copies of its
  case’s base image.
- GPU runs separate synchronized wall times for upload and download, a CUDA
  event time for the kernel, and synchronized wall times for the resident,
  f32-input and f64-input paths. The f64-input path includes conversion
  into the preallocated f32 transfer buffer. Both host-input paths return
  results to host memory. Buffer allocation and input generation occur before
  timing. Both pageable and cacheable pinned host buffers are measured.
- CPU times use the public `Integrator` slice methods with the default fast
  CSC, fast CSR and strict CSR policies. Rayon pools have 1, 4, 8 or 16
  threads and process independent frames and the integration work on the
  same pool. Each configuration gets two warmups and at least three timed
  repeats. The result summaries give the median and full observed range.
- The largest batch uses reported free device memory after table upload
  with 128 MiB of headroom. When available, dedicated free memory from
  `nvidia-smi` also bounds the CUDA allocation budget. The host budget is
  capped at 15 GB and reserves 2 GiB of available system memory. For
  batches that exceed the host budget, the harness reuses a slab of up
  to 64 host frames. It uploads the slab into distinct device offsets,
  including any final partial slab. The f64 path converts each slab
  before uploading it. The resident path retains the whole batch on
  the device. The raw rows record both the device batch and host slab
  sizes.
- Accuracy uses the existing strict f64 CSR `Integrator` result for the
  same image and table. Maximum absolute error includes every bin.
  Maximum relative error is over nonzero CPU reference bins.

The prototype branches from `v0.14.0` (`7340d59`). The measurement JSON
records the source commit and file hashes. The Metal record on `main`
uses the same Pilatus and Eiger images. Their SHA-256 hashes and the Eiger mask’s
hash matched the Metal record. Both PONI calibrations matched after normalizing
the Windows checkout’s line endings. Counting on Pilatus produced the same
949,623 nonzero contributions. Pilatus pixel splitting produced 6,023,814 here
and 6,023,815 in the Metal record. The tables are prepared by the local Dioptrin
revision, so agreement between the two backends is evaluated against each
run’s own strict CPU reference.
[Metal benchmark record](https://github.com/CPrescher/Dioptrin/blob/7cf3727db04d1557b10fab670bc5e3e46d227e71/docs/benchmarks/2026-11-10-metal/README.md).

## Native Windows throughput

The CPU point for a batch is the lowest median of the tested layout and
thread combinations. The 16-thread pool uses all 16 logical processors
and shares them between independent frames and the work within each
integration call. Eiger pixel splitting with std improved from 7.016
ms for a single CPU frame to 4.273 ms/frame over 64 frames. The latter
is about 273 ms for the batch. GPU entries are also divided by the
number of frames in the actual batch. All CPU measurements use Dioptrin’s
f64 integration. The f32 and f64 input columns describe the CUDA path.

### 64-frame batches

Times are median [min–max] in ms/frame. CUDA uses compensated f32 with f32
host input. CPU layout and thread count identify the winning configuration.

| Image | Method | Poisson std | CPU path | CPU | CUDA |
| --- | --- | --- | --- | ---: | ---: |
| Synthetic | counting | No | fast CSR, 4 | 0.030 [0.030–0.032] | 0.015 [0.015–0.015] |
| Synthetic | counting | Yes | strict CSR, 4 | 0.036 [0.031–0.036] | 0.021 [0.020–0.021] |
| Synthetic | pixel_split | No | fast CSR, 16 | 0.048 [0.047–0.052] | 0.016 [0.016–0.020] |
| Synthetic | pixel_split | Yes | fast CSR, 16 | 0.056 [0.056–0.058] | 0.022 [0.021–0.026] |
| Synthetic | pyfai_full | No | fast CSR, 16 | 0.048 [0.046–0.051] | 0.016 [0.016–0.017] |
| Synthetic | pyfai_full | Yes | fast CSR, 16 | 0.059 [0.053–0.060] | 0.021 [0.021–0.022] |
| Pilatus | counting | No | fast CSC, 16 | 0.164 [0.163–0.167] | 0.097 [0.096–0.098] |
| Pilatus | counting | Yes | fast CSR, 16 | 0.192 [0.186–0.194] | 0.097 [0.096–0.099] |
| Pilatus | pixel_split | No | fast CSC, 16 | 0.427 [0.416–0.470] | 0.126 [0.122–0.132] |
| Pilatus | pixel_split | Yes | fast CSR, 16 | 0.565 [0.548–0.581] | 0.125 [0.122–0.127] |
| Pilatus | pyfai_full | No | fast CSR, 16 | 0.501 [0.491–0.523] | 0.125 [0.123–0.129] |
| Pilatus | pyfai_full | Yes | fast CSR, 16 | 0.550 [0.545–0.559] | 0.122 [0.117–0.123] |
| Eiger | counting | No | strict CSR, 16 | 1.924 [1.918–1.961] | 0.919 [0.917–0.921] |
| Eiger | counting | Yes | fast CSR, 16 | 1.982 [1.951–1.998] | 0.922 [0.916–0.922] |
| Eiger | pixel_split | No | fast CSC, 16 | 3.325 [3.091–3.679] | 1.100 [1.100–1.105] |
| Eiger | pixel_split | Yes | fast CSC, 16 | 4.273 [4.249–4.376] | 1.108 [1.099–1.111] |
| Eiger | pyfai_full | No | fast CSC, 16 | 3.079 [3.042–3.283] | 1.093 [1.090–1.095] |
| Eiger | pyfai_full | Yes | fast CSC, 16 | 4.468 [4.139–4.591] | 1.105 [1.102–1.108] |

### CPU scheduling headroom

A separate CPU-only run alternated 14 and 16 Rayon workers, with two
warmups and seven timing samples per setup. The best f64 median at each
worker count is shown for 64 frames with Poisson std. Entries are
ms/frame [min–max]. The JSON also includes single frames, intensity-only,
and all three layouts at both thread counts.

| Image | Method | 14 workers | 16 workers |
| --- | --- | ---: | ---: |
| Pilatus | counting | 0.228 [0.213–0.255] | 0.224 [0.205–0.248] |
| Pilatus | pixel_split | 0.600 [0.585–0.616] | 0.587 [0.567–0.641] |
| Pilatus | pyfai_full | 0.648 [0.623–0.728] | 0.620 [0.565–0.694] |
| Eiger | counting | 2.159 [2.083–2.452] | 2.172 [2.099–2.413] |
| Eiger | pixel_split | 4.370 [4.053–5.311] | 4.191 [4.034–4.397] |
| Eiger | pyfai_full | 4.517 [4.371–4.732] | 4.589 [4.405–5.079] |

For Eiger pixel splitting, 14 workers took 4.3% longer per frame. All
six ranges overlapped. A 14-worker pool is a practical desktop setting
on this system. Windows may schedule those workers on any of the
16 logical processors because the benchmark uses no affinity mask.
The 16-worker result remains the full-machine benchmark point.

```sh
python docs/benchmarks/2026-11-18-cuda/run.py --cpu-only \
    --cases pilatus,eiger --cpu-threads 14,16 --batches 1,64 --repeats 7 \
    --out benchmark-results/cuda/windows-cpu-headroom.json
```

### Pixel splitting by batch size

These are median ms/frame with Poisson std. Each CUDA cell gives the
best f32-input / f64-input wall medians for its accumulation policy.
The larger staged runs used the layouts selected from the standard
64-frame sweep and compared all three CPU policies at 16 threads.

| Image | Frames | Best CPU | Plain | Compensated | Double |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilatus | 1 | 0.352 | 0.262 / 0.328 | 0.273 / 0.326 | 0.377 / 0.457 |
| Pilatus | 8 | 0.546 | 0.182 / 0.283 | 0.175 / 0.280 | 0.303 / 0.431 |
| Pilatus | 64 | 0.565 | 0.124 / 0.406 | 0.125 / 0.404 | 0.250 / 0.537 |
| Pilatus | 256 | 0.561 | 0.121 / 0.407 | 0.118 / 0.406 | 0.246 / 0.530 |
| Pilatus | 3,469 (capacity) | 0.652 | 0.122 / 0.414 | 0.119 / 0.404 | 0.258 / 0.541 |
| Eiger | 1 | 7.016 | 1.365 / 2.869 | 1.511 / 2.742 | 1.571 / 3.020 |
| Eiger | 8 | 5.795 | 1.115 / 3.958 | 1.140 / 3.945 | 1.424 / 4.232 |
| Eiger | 64 | 4.273 | 1.106 / 3.925 | 1.108 / 3.923 | 1.405 / 4.215 |
| Eiger | 256 | 5.921 | 1.109 / 3.995 | 1.121 / 4.037 | 1.515 / 4.424 |
| Eiger | 341 (capacity) | 5.504 | 1.098 / 3.853 | 1.103 / 3.853 | 1.413 / 4.160 |

The 256-frame Eiger tests cover all three methods and both output modes.
The capacity trials focus on pixel splitting. For Eiger, CUDA reported
15.57 GB free after table upload and the dedicated-memory reading was
14.00 GB. The latter bounded the run to 341 frames after headroom.
The 64-frame host slab and final partial slab used 8.65 GB of host
buffers with std. Pilatus fit 3,469 frames with std, based on 14.50 GB
of free dedicated memory and 1.01 GB of host buffers. At Eiger capacity,
compensated CUDA was about 5.0×
the measured f64 CPU throughput from f32 input and 1.43× from f64
input. The three compensated f32-input repetitions ranged from
1.103 to 1.104 ms/frame after rounding.

### CUDA phases

Timings below are median ms/frame for 64-frame pixel splitting with std.
Pinned results use the lowest f32 end-to-end median for each mode. The
pageable column uses the best pageable f32 configuration for that mode.
Upload, kernel and download are timed separately from the complete
host-input operations.

| Image | Accumulation / layout | Upload | Device kernel | Download | Resident wall | Pinned f32 input | Pinned f64 input | Pageable f32 input |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pilatus | Plain / warp | 0.079 | 0.041 | 0.001 | 0.045 | 0.124 | 0.406 | 0.352 |
| Pilatus | Compensated / warp | 0.080 | 0.041 | 0.002 | 0.036 | 0.125 | 0.404 | 0.343 |
| Pilatus | Double / warp | 0.080 | 0.166 | 0.002 | 0.166 | 0.250 | 0.537 | 0.442 |
| Eiger | Plain / block | 0.797 | 0.328 | 0.002 | 0.318 | 1.106 | 3.925 | 2.506 |
| Eiger | Compensated / block | 0.781 | 0.327 | 0.002 | 0.325 | 1.108 | 3.923 | 2.491 |
| Eiger | Double / warp | 0.777 | 0.624 | 0.002 | 0.627 | 1.405 | 4.215 | 2.777 |

Pinned input cut the Eiger compensated f32 round trip by about 2.25×.
Double accumulation took about 4.0× the compensated device time on
Pilatus and 1.9× on Eiger, with each mode using its selected layout.
The warp mapping was faster on Pilatus. For Eiger weighted plain and
compensated kernels, the block mapping was faster at this batch size.

The standard runs recorded the following one-time preparation and upload
times. Plan preparation includes the separate upload time in the last
column.

| Image | Method | Plan preparation (ms) | CUDA table upload (ms) |
| --- | --- | ---: | ---: |
| Pilatus | counting | 54.9 | 5.8 |
| Pilatus | pixel_split | 298.2 | 14.7 |
| Pilatus | pyfai_full | 450.1 | 16.8 |
| Eiger | counting | 488.0 | 7.8 |
| Eiger | pixel_split | 2,137.3 | 43.7 |
| Eiger | pyfai_full | 2,539.1 | 75.4 |

The normal-cache Pilatus process spent 28.5 ms in NVRTC and 0.5 ms in
module loading. The Eiger process used `CUDA_CACHE_DISABLE=1` and recorded
340.2 ms in NVRTC and 318.2 ms in module loading. Those individual runs
include the state of the driver and host caches at launch.

### Single-frame comparison with Metal

Both GPU columns are compensated f32 from an f64 host image with Poisson
std. Each speedup divides the CPU time on that machine by its own GPU
time. CUDA uses discrete transfers and GPU normalization. Metal uses
shared buffers and normalizes on the CPU. The published Metal numbers
come from an M4 Pro. [Metal benchmark results](https://github.com/CPrescher/Dioptrin/blob/7cf3727db04d1557b10fab670bc5e3e46d227e71/docs/benchmarks/2026-11-10-metal/README.md#results).

| Image / method | CUDA (ms) | CUDA speedup | Metal (ms) | Metal speedup |
| --- | ---: | ---: | ---: | ---: |
| Pilatus counting | 0.804 | 0.12× | 0.241 | 0.38× |
| Pilatus pixel splitting | 0.326 | 1.08× | 0.434 | 1.00× |
| Eiger counting | 2.775 | 0.43× | 1.712 | 0.80× |
| Eiger pixel splitting | 2.742 | 2.56× | 2.400 | 1.22× |

Counting is the main loss case. For 64 Eiger frames with std, compensated
CUDA took 0.922 ms/frame with f32 input and 3.736 ms/frame with f64
input, against 1.982 ms/frame on the CPU. For a single synthetic counting
frame with std, the CPU took 0.043 ms and compensated CUDA took 0.081 ms
even with f32 input. Eiger weighted batches showed a larger f32 speedup
than the f64-input path, where conversion consumed much of the margin.

### Fresh process to first pattern

`startup.py` starts a fresh executable for each trial and times from
process launch until the first intensity and Poisson std arrive in host
memory. Both paths read the TIFF, PONI and applicable mask and build
their integration plan inside the interval. The input is f64.
Compilation of the Rust executable and fixture hashing occur outside
the measured interval. The code records both total time and time
after loading the fixture.

CPU layout and thread count come from the lowest single-frame
median in the steady-state trials. CUDA uses compensated f32 with the
layout and host buffer type selected by its f64-input single-frame
median. The cache-enabled run primes the driver before measurement.
The cache-disabled run uses `CUDA_CACHE_DISABLE=1`. Both use the normal
filesystem cache. The randomized trial order helps spread changes in
machine load. The CUDA phase samples separate driver/context setup,
NVRTC, module loading, table preparation and upload, buffer allocation,
and the first f64-input frame.

```sh
python -B docs/benchmarks/2026-11-18-cuda/startup.py \
    --steady-results benchmark-results/cuda/steady.json \
    --out benchmark-results/cuda/startup.json
```

The table reports median [min–max] milliseconds from process launch to
the first pattern, with five fresh processes per cell.

| Image | Method | CPU path | CPU | CUDA, cache enabled | CUDA, cache disabled |
| --- | --- | --- | ---: | ---: | ---: |
| Pilatus | counting | fast CSR, 8 | 61.1 [60.0–62.2] | 881.5 [878.9–893.3] | 884.6 [852.0–898.7] |
| Pilatus | pixel_split | fast CSR, 16 | 320.7 [300.1–328.6] | 1,129.2 [1,109.3–1,135.8] | 1,130.3 [1,125.6–1,145.3] |
| Pilatus | pyfai_full | fast CSR, 16 | 388.2 [370.0–397.4] | 1,199.8 [1,178.7–1,233.2] | 1,214.5 [1,173.5–1,292.6] |
| Eiger | counting | fast CSR, 16 | 582.3 [544.7–599.0] | 1,369.6 [1,364.5–1,416.7] | 1,385.9 [1,369.9–1,460.7] |
| Eiger | pixel_split | fast CSR, 16 | 2,116.4 [2,041.9–2,158.5] | 2,888.0 [2,825.4–2,975.7] | 3,063.2 [2,846.4–3,117.9] |
| Eiger | pyfai_full | strict CSR, 16 | 2,353.8 [2,272.1–2,416.8] | 3,112.7 [3,089.3–3,132.0] | 3,130.8 [3,080.5–3,190.7] |

For Eiger pixel splitting with the cache enabled, median after-fixture
time was 2,047 ms for CPU and 2,816 ms for CUDA. CUDA spent 138 ms
in driver and context initialization, 360 ms in NVRTC, 319 ms in
module loading, and 1,995 ms in plan preparation. Plan preparation
includes 18 ms of table upload. Buffer and kernel preparation took
5.3 ms and the first f64-input frame took 4.2 ms. These are medians
of individual phases. The cache-disabled trials also spent about
315 ms loading the module, so enabling the cache did not remove
module load time on this machine. CPU reached the first pattern
sooner in every matched case.

## Numerical accuracy

Each cell gives maximum absolute / maximum relative error against strict
f64 CSR. Maxima are over the tested layouts for each input, with Poisson
std enabled. The synthetic frame is 256×256, Pilatus is 1043×981 and
Eiger is 3262×3108.

| Image | Method | Plain intensity | Plain std | Compensated intensity | Compensated std | Double intensity | Double std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | counting | 0.00640 / 1.83e-7 | 4.09e-6 / 9.16e-8 | 0.00380 / 5.43e-8 | 2.85e-6 / 2.72e-8 | 0.00380 / 5.43e-8 | 2.85e-6 / 2.72e-8 |
| Synthetic | pixel_split | 0.00105 / 2.66e-7 | 1.46e-6 / 1.53e-7 | 0.000641 / 7.93e-8 | 6.97e-7 / 5.80e-8 | 0.000641 / 7.93e-8 | 6.97e-7 / 5.80e-8 |
| Synthetic | pyfai_full | 0.00205 / 2.89e-7 | 1.59e-6 / 2.20e-7 | 0.000627 / 7.89e-8 | 1.09e-6 / 6.35e-8 | 0.000627 / 7.89e-8 | 1.09e-6 / 6.35e-8 |
| Pilatus | counting | 0 / 0 | 1.78e-15 / 2.83e-16 | 0 / 0 | 1.78e-15 / 2.83e-16 | 0 / 0 | 1.78e-15 / 2.83e-16 |
| Pilatus | pixel_split | 0.000447 / 1.52e-7 | 3.68e-7 / 1.05e-7 | 2.38e-5 / 3.47e-8 | 2.98e-7 / 3.95e-8 | 2.38e-5 / 3.47e-8 | 2.98e-7 / 3.95e-8 |
| Pilatus | pyfai_full | 0.000612 / 1.69e-7 | 4.28e-7 / 1.06e-7 | 2.55e-5 / 3.96e-8 | 4.08e-7 / 3.96e-8 | 2.55e-5 / 3.96e-8 | 4.08e-7 / 3.96e-8 |
| Eiger | counting | 116 / 3.01e-6 | 0.000883 / 1.51e-6 | 3.55 / 4.66e-10 | 3.84e-5 / 2.33e-10 | 3.55 / 4.66e-10 | 3.84e-5 / 2.33e-10 |
| Eiger | pixel_split | 309 / 2.48e-6 | 0.00666 / 8.92e-7 | 300 / 5.24e-8 | 0.00652 / 5.21e-8 | 300 / 5.24e-8 | 0.00652 / 5.21e-8 |
| Eiger | pyfai_full | 218 / 2.46e-6 | 0.00543 / 8.70e-7 | 194 / 5.57e-8 | 0.00415 / 5.59e-8 | 194 / 5.57e-8 | 0.00415 / 5.59e-8 |

Compensated and double give the same maxima to the precision shown. The
remaining error is dominated by f32 input and weight quantization. For
example, the Eiger frame includes unmasked int32 saturation values that
round by one count when converted to f32. The error in weighted Eiger
patterns has a small relative maximum even where absolute intensity
differences are large.

The feature-gated test checks both output modes and all three arithmetic
policies against the public `Integrator`, with partial and full masks,
negative pixels, distinct frames and a batch that crosses a launch
boundary.

## Run records and follow-up

The native Windows timings use commits `3dbf31c`, `a18c720`,
`a035cef` and `c843ec8` on the prototype branch. The standard Pilatus run completed
its Pilatus section before stopping the queued synthetic Eiger case.
The subsequent Eiger run used the matching TIFF. Eiger at 256 frames
and the two capacity runs used the staging implementation. The output
JSON preserves all repeats, phase timings, configuration choices,
input and source hashes, and the GPU telemetry before and after each
run. The raw records live in `benchmark-results/`, outside Git.
Across the longer runs, recorded GPU temperatures were 44–59 °C and
spot checks showed SM clocks of about 2.86 GHz when busy. Clocks varied
at lighter loads. All measurements were native Windows.

For throughput, the next experiment would overlap PCIe transfers and
kernel execution using two pinned staging buffers. For the f64-input
path, the first target is the host conversion cost. The large difference
between resident and host-input costs also motivates testing a producer
that emits f32 directly. For latency, a retained CUDA context, compiled
module and prepared table would amortize the startup work.
