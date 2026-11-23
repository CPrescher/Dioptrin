# Experimental CUDA 1D integration

```{warning}
This feature is experimental. Its Python and Rust APIs are excluded from
semantic-versioning guarantees and may change or be removed in any release.
```

Dioptrin can integrate fixed-mask 1D patterns on an NVIDIA GPU through an
explicit plan. Nothing else changes: `integrate1d`, `batch1d` and every other
method keep running on the CPU, and no setting moves them to the GPU.

## Scope

- 1D integration only, with `mask_mode="fixed"` and the `counting`,
  `pixel_split` and `pyfai_full` methods. A dynamic mask, the `histogram` and
  `supersampled` methods, 2D integration, `azimuthal_std` and
  `MultiIntegrator` are not supported.
- Opt-in through `Integrator.prepare_cuda1d`, which returns a `Cuda1dPlan`.
- Not included in the PyPI wheels. A wheel without the feature still has every
  name below; `prepare_cuda1d` then raises `CudaUnavailableError`.

## Building

The `cuda` cargo feature needs only the Rust toolchain at build time: no CUDA
toolkit, `nvcc` or GPU. The CUDA libraries are loaded when the program runs.

```sh
maturin build --release --features python,cuda   # wheel in target/wheels/
maturin develop --release --features python,cuda # into the active environment
```

Rust users enable the feature in `Cargo.toml`:

```toml
dioptrin = { version = "...", features = ["cuda"] }
```

Do not set `CUDARC_CUDA_VERSION` while building: the bindings are generated
for the CUDA 13.4 API, and that variable overrides the choice.

## Run-time requirements

- An NVIDIA driver (`libcuda`).
- NVRTC (`libnvrtc` and `libnvrtc-builtins`) where the dynamic loader finds it,
  for example on `LD_LIBRARY_PATH`. On clusters with environment modules,
  loading a CUDA toolkit module usually provides it. On Windows NVRTC is
  `nvrtc64_*.dll` and must be on `PATH`. The pip wheel `nvidia-cuda-nvrtc`
  installs the libraries inside the Python environment, where the loader does
  not look: add its library directory to `LD_LIBRARY_PATH` (`PATH` on
  Windows) explicitly.
- The driver must accept the PTX that this NVRTC emits: in practice the
  driver's CUDA version (shown by `nvidia-smi`) must be at least the NVRTC
  version.
- The GPU's compute capability must be supported by that NVRTC (CUDA 13
  requires 7.5 or newer).

The bindings are generated for the CUDA 13.4 API. Using them with a 12.x
driver or NVRTC 12 is untested.

The kernels are compiled by NVRTC the first time a device is used in a
process and cached for the rest of it. A CUDA-enabled build runs on machines
without a GPU: `cuda_available()` returns `False` and `prepare_cuda1d` raises
`CudaUnavailableError`.

## Python usage

```python
import dioptrin
import numpy as np

print(dioptrin.cuda_info())          # build and device description, never raises
if dioptrin.cuda_available():
    ai = dioptrin.Integrator("calibration.poni", method="pyfai_full",
                             unit="2th_deg", radial_range=(1.0, 30.0))
    ai.set_mask(mask)
    plan = ai.prepare_cuda1d(image.shape, 2000)   # precision="f32_compensated"

    result = plan.integrate1d(image, errors=True)        # one frame
    results = plan.batch1d(frames)                       # list of 2D arrays
    results = plan.batch1d(np.stack(frames))             # or one 3D array
```

- `cuda_available(device=0)` and `cuda_info(device=0)` report whether the build
  has CUDA support and whether the device can be used. `cuda_info` returns a
  dict with the keys `compiled`, `available`, `reason`, `device`,
  `device_name`, `compute_capability`, `total_memory_bytes`,
  `driver_version`, `nvrtc_version` and `bindings_version`.
- `Integrator.prepare_cuda1d(shape, n_points, *, precision="f32_compensated",
  device=0, max_batch_bytes=None)` builds the lookup table on the CPU and
  uploads it.
- `Cuda1dPlan.integrate1d(image, *, errors=False)` returns an
  `IntegrationResult`; `Cuda1dPlan.batch1d(images, *, errors=False)` returns
  a list of them. The fields are lists, as for CPU `integrate1d`.
- Inputs are C-contiguous `float64`, `float32`, `uint16`, `uint32` or `int32`
  arrays of the plan's shape, or a list or tuple of such arrays of one dtype, or
  one C-contiguous 3D array. They are borrowed without copying for the call and
  must not be mutated concurrently; read-only arrays and `np.memmap` views are
  accepted. Integer frames are uploaded natively and converted in-register on
  the device; `float64` frames are rounded to `float32` into pinned staging
  unless the precision is `"f64"`.
- `Cuda1dPlan.alloc_pinned_batch(n_frames, dtype="float32")` allocates a
  writable 3D NumPy array `(n_frames, dim1, dim2)` backed by CUDA page-locked
  (pinned) host memory. External readers such as `dioptrin-io` or `h5py`
  (`dataset.read_direct(buf)`) can decode directly into it; `batch1d` detects
  pinned memory automatically and performs zero-copy async PCIe DMA.
- `Cuda1dPlan.batch1d_bslz4(chunks, dtype="uint32", *, errors=False)` accepts a
  sequence of raw HDF5 Bitshuffle+LZ4 compressed frame chunks (for example from
  `h5py.Dataset.id.read_direct_chunk` or `dioptrin-io`), decompresses them on
  the GPU, and integrates the resulting frames directly in VRAM.
- As on the CPU, the mask of an `np.ma.MaskedArray` input is ignored: only its
  data is integrated. Use `set_mask` before preparing the plan instead.
- `batch1d` processes frames in chunks of at most `plan.max_batch_frames`. A
  `batch1d` call fails as a whole; there are no per-frame errors.
- `max_batch_bytes` caps the device memory of one chunk. One frame takes
  `pixels × 4` bytes for float32 storage plus `pixels × 8` for the float64
  staging buffer (`pixels × 8` in all for `"f64"`), plus `n_points × 16` for
  the results. The default is `min(1 GiB, half of the free device memory
  after the table upload)`. An explicit value is not checked against the free
  device memory: a chunk that does not fit raises `RuntimeError` when
  `batch1d` runs, and the plan stays usable with smaller calls.
- The GIL is released while preparing and integrating. A plan can be shared
  between threads; calls on one plan run one at a time.

Plan properties: `shape`, `n_points`, `precision`, `device`, `device_name`,
`radial`, `max_batch_frames` and `table_entries`.

Errors:

| Condition | Exception |
| --- | --- |
| Built without the `cuda` feature, or no usable driver, NVRTC or device | `CudaUnavailableError` (a `RuntimeError`) |
| Dynamic mask, unsupported method, `n_points == 0`, shape not matching the mask or detector, unknown precision, budget too small | `ValueError` |
| Input dtype other than float64/float32, or mixed dtypes in one call | `TypeError` |
| Input not C-contiguous, not aligned, or of the wrong shape | `ValueError` |
| CUDA driver error at run time, such as running out of device memory | `RuntimeError` |

The Rust API mirrors this: `Integrator::prepare_cuda1d(shape, n_points,
&Cuda1dOptions)` returns a `Cuda1dPlan` with `integrate_frame`,
`integrate_frames` and `integrate_stack`, and `cuda_info(device)` returns a
`CudaInfo`. Errors are `CudaError` values.

## Precision and agreement with the CPU

`precision` selects the storage and accumulation precision on the device:

| Precision | Pixels and weights | Accumulation |
| --- | --- | --- |
| `"f32"` | float32 | float32 |
| `"f32_compensated"` (default) | float32 | double-word compensated float32 |
| `"f64"` | float64 | float64 |

In every mode the bin denominators (solid angle, polarisation, mask) are summed
on the CPU in float64 when the plan is prepared, and the division, error
propagation and output are float64. The radial axis is the one CPU
`integrate1d` returns for the same integrator and `n_points`.

The GPU gathers the table in its own fixed order, so its results agree with
CPU `integrate1d` within the tolerance of the chosen precision, not bit for
bit. They are deterministic, and `batch1d` gives bit-identical results to
`integrate1d` on each frame. Both are properties of the design (a fixed
reduction order per bin, no atomics, the same kernel for one frame and many)
that the device tests check.

`fast_math` and `sparse_layout` do not affect the GPU.

## Plans are snapshots

A plan captures the integrator's mask, corrections, method, unit and radial
range when it is prepared. Later calls such as `set_mask` or
`set_polarization_factor` do not change it; prepare a new plan instead. Device
memory is freed when the plan is garbage-collected.

## Processes

CUDA is not fork-safe once initialised. Do not fork a process after it has
used CUDA, which includes calling `cuda_info`, `cuda_available` or
`prepare_cuda1d`. With `multiprocessing`, use the `spawn` or `forkserver`
start method, or initialise CUDA only in the child processes.
