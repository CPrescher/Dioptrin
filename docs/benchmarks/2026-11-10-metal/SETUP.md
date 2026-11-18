# Complete startup, break-even, and reuse after a pause

Measured on the same M4 Pro on November 10, 2026. All results below include
intensity and Poisson std, 3,000 bins, fixed masks, solid-angle correction, and
polarization factor 0.965. The Metal kernel uses compensated accumulation.

## Complete startup measurement

`run_setup.py` measures the wall interval from launching a fresh process to its
first completed pattern. This includes executable/framework loading, the Rayon
pool, image/PONI/mask reading, geometry, lookup-table construction, mask packing,
normalization, allocations, and the first integration. For GPU execution it also
includes Metal device/queue creation, buffer filling, shader/pipeline creation,
input conversion, dispatch/wait, and result readback/normalization.

The new benchmark-only Rust static-library bridge supplies native arrays to the
Swift Metal harness in the same process. There are no intermediate fixture
exports or duplicate fixture-file reads. CPU lazy plan construction happens in
its first integration and is fully inside the measured interval. GPU numerical
reference computation occurs after timing. Only the GPU pipeline actually used
is prepared; the earlier prototype's full variant sweep is unnecessary here.

Seven fresh-process trials per backend/case are run in randomized backend order.
Each process also measures 51 subsequent integrations after at least 150 ms of
warm-up. Timings below use the median startup and the median of the seven
per-process steady-state medians. Compilation, benchmark warm-up, reference
validation, and process teardown are excluded from startup. Normal filesystem
and shader caches are retained; these are fresh-process starts, not starts after
purging all system caches or rebooting. The very first GPU process took 337 ms
for Pilatus counting; its seven-run median was 68 ms. All samples are retained.

Both engines use the same executable and bridge. CPU configuration and GPU
threadgroup size were selected from the earlier tuning sweep and held fixed.
The CPU runs the existing multithreaded `Integrator` API. Per-frame costs are
remeasured in this harness, and can differ from the earlier standalone sweep;
these startup measurements are paired with the newly measured frame costs.

| Image / method | Tuned CPU: startup through first pattern | GPU: startup through first pattern | Tuned CPU: subsequent frame | GPU: subsequent FP64 frame |
| --- | ---: | ---: | ---: | ---: |
| Pilatus counting | 37.25 ms | 67.73 ms | 0.128 ms | 0.431 ms |
| Pilatus pixel splitting | 150.08 ms | 180.23 ms | 0.438 ms | 0.438 ms |
| Eiger counting | 286.55 ms | 344.57 ms | 1.468 ms | 1.718 ms |
| Eiger pixel splitting | 741.85 ms | 799.44 ms | 3.004 ms | 2.395 ms |

Tuned CPU: fast CSR with 10 threads for Pilatus, strict CSR with 14 threads for
Eiger counting, and fast CSR with 14 threads for Eiger splitting. The default
CSC path is also measured below. Eiger default CSC was marginally faster per
subsequent frame than the selected CSR configuration in this run, but its
startup was slower; comparing both complete paths avoids mixing the startup
of one configuration with the recurring cost of another.

“Setup excluding the first image calculation” is not directly isolatable from
the CPU's public lazy API. An effective setup intercept can be derived as
`startup_to_first_pattern - steady_frame_time`: respectively **37.12, 149.64,
285.08, and 738.84 ms for the tuned CPU**, and **67.30, 179.80, 342.85, and
797.05 ms for the GPU**. These are derived intercepts, not separate timer
measurements. The direct complete measurement is the first-pattern column.

## Break-even for consecutive frames

Let `F` be complete time to the first pattern and `t` the measured time per
subsequent frame. For a reused plan:

```text
T_CPU(N) = F_CPU + (N - 1) * t_CPU
T_GPU(N) = F_GPU + (N - 1) * t_GPU

If t_CPU > t_GPU:
N_break_even = 1 + ceil(max(0, F_GPU - F_CPU) / (t_CPU - t_GPU))
```

If the GPU is slower per frame and also starts slower, there is no break-even
in this model. A faster GPU startup with slower recurring execution would
instead produce an initial-only advantage; the analysis handles that case too.

| Image / method | Break-even against tuned CPU | Interpretation |
| --- | ---: | --- |
| Pilatus counting | None | GPU starts slower and integrates slower |
| Pilatus pixel splitting | No reliable break-even | Frame times are essentially tied; GPU has about 30 ms more startup |
| Eiger counting | None in the median model | GPU starts slower and integrates slower |
| Eiger pixel splitting | **96 images; approximately 100** | About 58 ms additional startup, recovered at about 0.61 ms/image |

The Eiger splitting estimate is variable: six paired trials predict 68–138
images, while one trial had slower GPU recurring execution and no break-even.
Thus 96 is an estimate under the measured conditions, not an exact threshold.

The model was checked with actual uninterrupted batches in fresh processes,
without artificial warm-up inserted between setup and the batch:

| Number of Eiger pixel-split images | CPU total, three trials | GPU total, three trials | Observed winner |
| --- | --- | --- | --- |
| 48 | 857, 913, 911 ms | 896, 942, 935 ms | CPU, all three trials |
| 192 | 1,383, 1,316, 1,308 ms | 1,311, 1,289, 1,246 ms | GPU, all three trials |

This brackets the observed change in winner between 48 and 192 images and
supports an approximate 100-image crossover. Each GPU batch's first and last
patterns passed the numerical checks used in the original experiment.

For completeness, the default CSC path gives different startup/recurring costs:

| Image / method | Default CSC startup | Default CSC per frame | GPU break-even against default CSC, median model |
| --- | ---: | ---: | --- |
| Pilatus counting | 40.73 ms | 0.320 ms | None |
| Pilatus pixel splitting | 159.52 ms | 1.316 ms | 25 images |
| Eiger counting | 348.16 ms | 2.536 ms | First image |
| Eiger pixel splitting | 806.68 ms | 2.953 ms | First image |

The Eiger GPU/default-CPU startup differences are small compared with run-to-run
variation; “first image” describes the median model, not a guaranteed startup
win. The tuned CPU comparison is the more conservative amortization test.

## Another frame after 30 seconds

The GPU idle test prepares one plan and keeps its Rust context, Metal device,
queue, pipeline, lookup tables, normalization, and buffers alive. It dispatches
no integrations for 30 seconds, then times the next integration. This is
repeated three times in the same process. The desktop can still use the GPU.

The next Eiger pixel-split pattern took **12.54, 29.25, and 14.95 ms**, including
FP64 input conversion, dispatch/wait, and readback. No setup routine was called
again. The initial complete startup was approximately **800 ms**. The medians
of the next ten immediate frames were 2.82, 5.01, and 4.79 ms; fully warmed
integration before the pauses was 2.37 ms in that process.

This directly demonstrates plan reuse after the pause. The additional latency
can include cache, clock, and scheduling effects; no device counters were
collected to assign it to one cause. A 30-second pause does not invalidate the
plan. Geometry, mask, binning, or correction changes require updating the plan;
restarting the process also discards the in-memory state. The current prototype
does not automatically rebuild on changes.

The matched retained CPU plan took **6.73, 23.46, and 13.12 ms** after
its three 30-second pauses (median 13.12 ms). The GPU median was
14.95 ms. These few samples overlap substantially and were measured in separate
processes; they do not establish a statistically reliable difference. Both
paths clearly avoid repeating the hundreds-of-milliseconds startup. With one
frame every 30 seconds, this experiment demonstrates **no GPU speed advantage**;
the median GPU idle-frame cost was higher than the CPU's, so applying the hot
batch crossover to that workflow would be misleading.

The approximately 100-image break-even above applies to consecutive frames.
Frames separated by 30 seconds have different recurring costs and should use
the idle measurements instead of the warmed throughput numbers. Waiting time
itself is common to both paths and is excluded from these compute-time totals.

## Reproduce and verify

```sh
python3 docs/benchmarks/2026-11-10-metal/run_setup.py --repeats 7 --samples 51
python3 docs/benchmarks/2026-11-10-metal/run_idle_cpu.py
```

Both scripts need the Eiger TIFF `test_data/eiger_0000.tiff`, which is not in
the repository (see [README.md](README.md#reproduce)); `run_setup.py
--small-only` runs the Pilatus cases alone.

Source: `setup/Cargo.toml`, `setup/src/lib.rs`, `setup/SetupBench.swift`, and the
unchanged `integrate.metal` kernel. The benchmark adds no production API or
dependency changes. Build outputs go under the repository's Cargo target
directory and temporary executable directories are cleaned up.

Raw evidence: `setup-results.json` (84 fresh-process trials, six measured batch
pairs, and the GPU idle test) and `idle-cpu-results.json` (matched CPU idle
follow-up). These record source hashes, first-pattern timing events, individual
frame samples, and accuracy results. GPU checks compare first/last patterns
with the original FP64 public CSR implementation and require intensity error
below 0.01 reference Poisson std. CPU execution uses the existing library API,
whose 202 unit tests passed in the original prototype validation.
