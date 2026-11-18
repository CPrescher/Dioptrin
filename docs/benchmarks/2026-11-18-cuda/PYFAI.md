# pyFAI CPU and OpenCL comparison on Windows

The comparison uses pyFAI 2026.9.0, the latest stable PyPI release at
measurement time, and PyOpenCL 2026.1.5. It ran on the same Windows
11 Pro workstation as the [CUDA prototype](README.md), with a Ryzen
7 9800X3D and RTX 5070 Ti. The OpenCL result identifies
`pyFAI.opencl.azim_csr.OCL_CSR_Integrator` on the RTX 5070 Ti under
NVIDIA driver 616.64. The CPU rows use native Cython engines.
[pyFAI release](https://pypi.org/project/pyFAI/2026.9.0/).

## Measurement

- Both libraries read the original Pilatus and Eiger TIFFs. The
  comparator uses the same row flip, uniform-pixel PONI geometry,
  fixed masks, polarization factor 0.965, solid-angle correction and
  3,000 two-theta bins in degrees. TIFF and mask hashes are recorded.
- pyFAI's no-split method is compared with Dioptrin counting.
  pyFAI's full split is paired with Dioptrin `pyfai_full`. Each
  library builds its own table and uses its own Poisson implementation.
  In particular, pyFAI's native Poisson model imposes a variance floor
  of one. The comparison measures speed for the corresponding
  algorithms and correction settings.
  [pyFAI engine documentation](https://pyfai.readthedocs.io/en/stable/api/engines/index.html).
- pyFAI Cython CSR and CSC use one caller with 14 OpenMP threads.
  Another CPU configuration uses four concurrent frame callers with
  one OpenMP thread each. It gives each caller its own initialized CSR
  engine and scratch storage. The benchmark verifies the effective
  OpenMP settings and selected implementation. The GPU path uses
  OpenCL CSR on the explicitly selected NVIDIA card.
- Warm measurements call pyFAI's public `integrate1d` on 1, 8 or 64
  contiguous host frames, with both intensity-only and Poisson output.
  Stable geometry and masks allow `safe=False` after an initial
  checked call. The four-worker CPU path schedules frames concurrently.
  The GPU path submits one frame per call and returns each result
  to host memory. CUDA uses its native multi-frame launches. File
  reading, host buffer construction, and NumPy dtype preparation
  happen outside warm timing.
- There are three timed repeats after two warmups and three fresh
  processes per setup configuration. The reported
  startup interval is fresh process launch to first f64-host Poisson
  pattern, including Python imports, fixture reading, and first-call
  plan preparation. Additional CPU-worker setup is recorded separately.

## Warm throughput

The table reports median [min–max] ms/frame with Poisson std. The CPU
column uses the fastest tested pyFAI configuration for each row. A
four-worker CPU entry uses one OpenMP thread per frame caller.

| Image | Split | Frames | pyFAI CPU f64 | CPU configuration | pyFAI OpenCL f64 | pyFAI OpenCL f32 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Pilatus | None | 1 | 9.154 [9.019–10.175] | CSR, 1 OpenMP thread | 1.485 [1.466–1.790] | 0.698 [0.596–0.738] |
| Pilatus | None | 8 | 5.199 [4.857–5.256] | CSR, 4 frame workers | 1.502 [1.461–1.612] | 0.858 [0.842–1.040] |
| Pilatus | None | 64 | 5.184 [4.997–5.285] | CSR, 4 frame workers | 1.473 [1.398–1.530] | 0.677 [0.650–0.689] |
| Pilatus | Full | 1 | 13.070 [12.081–13.613] | CSR, 14 OpenMP threads | 1.359 [1.353–1.634] | 0.806 [0.741–0.846] |
| Pilatus | Full | 8 | 8.848 [8.706–8.891] | CSR, 4 frame workers | 1.564 [1.492–1.599] | 0.782 [0.734–0.975] |
| Pilatus | Full | 64 | 8.560 [8.504–8.603] | CSR, 4 frame workers | 1.729 [1.577–1.734] | 0.864 [0.861–0.933] |
| Eiger | None | 1 | 115.003 [113.091–120.092] | CSR, 14 OpenMP threads | 11.497 [10.198–11.661] | 3.006 [2.990–3.193] |
| Eiger | None | 8 | 58.023 [57.513–59.057] | CSR, 4 frame workers | 10.272 [9.596–10.634] | 3.511 [3.449–3.554] |
| Eiger | None | 64 | 57.190 [56.973–57.451] | CSR, 4 frame workers | 9.998 [9.949–11.263] | 3.486 [3.461–3.661] |
| Eiger | Full | 1 | 121.875 [120.694–124.345] | CSR, 14 OpenMP threads | 10.053 [10.008–10.119] | 3.698 [3.696–3.704] |
| Eiger | Full | 8 | 77.035 [76.266–77.316] | CSR, 4 frame workers | 10.001 [9.986–10.361] | 3.941 [3.826–4.452] |
| Eiger | Full | 64 | 74.145 [73.974–74.846] | CSR, 4 frame workers | 10.556 [10.446–10.617] | 3.830 [3.777–4.505] |

For the 64-frame Eiger full-split batch with Poisson std, the
single-caller CSR and CSC CPU medians were 121.768 and 148.230 ms/frame.
Parallel frame callers lowered CSR to 74.145 ms/frame, a 1.64×
throughput gain over the 14-thread CSR configuration. Preparing the
other three engines took 4.846 s in the timed-throughput process.

### Same-machine comparison at 64 frames

Times are median ms/frame with Poisson std. Dioptrin CPU uses its
best 14-worker f64 path. Dioptrin CUDA uses compensated accumulation.
GPU values include transfer and return a host result.

| Image / method | Dioptrin CPU f64 | pyFAI CPU f64 | Dioptrin CUDA f64 / f32 | pyFAI OpenCL f64 / f32 |
| --- | ---: | ---: | ---: | ---: |
| Pilatus counting | 0.228 | 5.184 | 0.376 / 0.097 | 1.473 / 0.677 |
| Pilatus full split | 0.648 | 8.560 | 0.402 / 0.122 | 1.729 / 0.864 |
| Eiger counting | 2.159 | 57.190 | 3.736 / 0.922 | 9.998 / 3.486 |
| Eiger full split | 4.517 | 74.145 | 3.920 / 1.105 | 10.556 / 3.830 |

For Eiger full splitting, pyFAI OpenCL was about 7.0× as fast as the
best tested pyFAI CPU configuration with f64 host input. Dioptrin CUDA
was 2.69× as fast as pyFAI OpenCL from f64 host input and 3.47× from f32.
These are public-API throughput comparisons on this workstation. The
results include each backend's data preparation and submission schedule.

With intensity alone at 64 frames, the pyFAI CPU, OpenCL f64 and OpenCL
f32 medians were respectively 5.341, 1.586 and 0.730 ms/frame for
Pilatus counting, and 8.377, 1.548 and 0.957 for Pilatus full splitting.
For Eiger they were 58.469, 10.236 and 3.441 for counting, and
74.100, 10.225 and 3.755 for full splitting. The JSON includes
the individual repeats and the results at 1 and 8 frames.

## Startup

These are median [min–max] milliseconds from a fresh process to the
first f64-input Poisson pattern. For CPU, the process uses the settings
of the winning 64-frame configuration. Its first pattern uses one
engine. Additional frame-worker preparation happens after that result.

| Image / method | pyFAI CPU | pyFAI OpenCL |
| --- | ---: | ---: |
| Pilatus counting | 2,316 [2,304–2,338] | 2,455 [2,438–3,260] |
| Pilatus full split | 2,621 [2,561–2,702] | 2,719 [2,712–2,722] |
| Eiger counting | 3,652 [3,636–3,712] | 3,700 [3,639–4,375] |
| Eiger full split | 5,165 [5,113–5,172] | 5,275 [5,211–5,330] |

Eiger full splitting spent a median 2,751 ms in pyFAI's first CPU
integration call and 2,846 ms in its first OpenCL call, including
plan initialization. The rest of the process-to-first time includes
Python imports and reading the fixture. The corresponding Dioptrin
full-split first-pattern medians were 2,354 ms CPU and 3,113 ms CUDA
with the driver cache enabled. See the [startup phase breakdown](README.md#fresh-process-to-first-pattern).

## Reproduce

Use a Python environment with the recorded pyFAI, PyOpenCL and
`threadpoolctl` packages. The NVIDIA driver supplies OpenCL.

```sh
python -m pip install pyFAI==2026.9.0 pyopencl==2026.1.5 threadpoolctl
python docs/benchmarks/2026-11-18-cuda/pyfai_compare.py \
    --backends cpu --cpu-threads 14 --cpu-frame-workers 4 --dtypes float64 \
    --gpu-name "RTX 5070 Ti" --out benchmark-results/cuda/pyfai-cpu.json
python docs/benchmarks/2026-11-18-cuda/pyfai_compare.py \
    --backends gpu --gpu-name "RTX 5070 Ti" \
    --out benchmark-results/cuda/pyfai-opencl.json
```

Use `--eiger-dir` if the Eiger TIFF and mask are stored outside
`test_data/`. The generated JSON includes package and driver versions,
source and input hashes, raw timings, and selected engines.
