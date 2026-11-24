# Dioptrin

Dioptrin provides fast 1D and 2D diffraction integration using a compiled Rust
engine. It supports pixel splitting, supersampling, detector corrections,
Poisson uncertainties, azimuthal intensity spread, and batch integration.

See the [user manual](https://dioptrin.readthedocs.io/en/latest/) for API examples,
HDF5 workflows and performance measurements.

## Installation

```bash
python -m pip install dioptrin
```

Dioptrin is distributed as binary wheels for standard CPython 3.11–3.15 on
Windows x86-64, macOS Intel and Apple silicon, and Linux x86-64 and ARM64.
Free-threaded Python is not supported.
If no compatible wheel is available, installation from the source distribution
requires a Rust toolchain. The source is available on
[GitHub](https://github.com/CPrescher/Dioptrin).

## License

Dioptrin is open-source software under the [MIT license](https://github.com/CPrescher/Dioptrin/blob/main/LICENSE).
Starting with 0.7.0, both source builds and pre-built wheels work without
activation, license keys, or expiration checks. Upgrade older installations:

```bash
python -m pip install --upgrade dioptrin
```

## Use with Dioptas

Install Dioptrin in the same Python environment as Dioptas and restart Dioptas. The **Use Dioptrin** checkbox becomes available in its
integration options. Dioptas continues to work with pyFAI without Dioptrin.

### Multi-detector 2D integration (0.11.0)

`MultiIntegrator.integrate2d` combines detector images on a shared radial and
azimuthal grid. Detectors retain their individual calibrations, masks and
corrections. Fractional polygon splitting is the default for `pixel_split`,
with periodic seam handling and detector-plane sampling at the beam center.

```python
# int1 and int2 have the same radial unit/range and their own geometry/mask.
multi = dioptrin.MultiIntegrator([int1, int2])
result = multi.integrate2d(
    [img1, img2], 2000, 360, errors=True, normalization="absolute",
)
cake = np.asarray(result.intensity).reshape(360, 2000)
```

Import `dioptrin` and `numpy as np` in your application. Radial coordinates use
the configured unit; azimuthal coordinates and sector edges use radians. Raw
signal, normalization and Poisson variance are merged before final division.
Absolute scaling includes each detector's pixel area divided by distance
squared, matching pyFAI MultiGeometry's solid-angle scale. Applications must
call the new API to use it; upgrading the library alone does not switch an
existing Dioptas multi-detector workflow. See the
[multi-detector guide](https://dioptrin.readthedocs.io/en/latest/MULTI_DETECTOR.html)
for complete setup, normalization, units, masks, uncertainties and benchmarks.

### Integration arithmetic

Fast accumulation is enabled by default starting in 0.6.0
(`fast_math=True`). It uses four partial float64 sums and hardware fused
multiply-add where available. Set `fast_math=False` in `Integrator` or
`Integrator.from_poni_dict` to select strict accumulation. The two modes may
differ in floating-point rounding. Geometry, float64 pixel weights and correction
formulas are unchanged. Histogram and azimuthal-spread algorithms keep their
existing behavior.


### File-batch workers

Starting in 0.6.1, file batches reuse typed TIFF decoding buffers and retain their
workers between calls. Standard 1D/2D CSR file tasks convert individual pixel
values to float64 inside the existing arithmetic. `Integrator.clear_file_workers()`
releases cached workers and buffers; already admitted streams finish normally.
Changing the worker/thread configuration replaces the pool. Cached workers retain
one decoded image buffer each until cleared or the integrator is released.


### Shared dynamic-mask plans (0.9.0)

For application-level frame parallelism, prepare one immutable plan and create
a lightweight worker for each application worker:

```python
plan = integrator.prepare_dynamic1d(image_shape, 2000)
worker = plan.worker(num_threads=1)
result = worker.integrate1d(image_float64, mask_uint8, errors=True)
```

All workers share the plan's geometry, float64 CSR/CSC tables and correction
arrays. Each call has its own mask, normalization sums and accumulation results.
The complete effective mask must be passed for each frame; the source
integrator's stored mask is excluded. Images and masks must be C-contiguous and
remain unchanged until the call returns. Numerical work releases the GIL and
borrows those arrays without copying. Changes to the source integrator leave
existing plans unchanged; prepare a new plan when scientific settings change.

Dioptrin owns the scientific plan and kernels. Applications retain responsibility
for reading files, preparing images, mask plugins, frame ordering and cancellation.
Use one compute thread per worker when frames are already processed in parallel.


### Fast 1D sparse layout

Starting in 0.9.0, fast 1D integration defaults to CSC for both mask policies.
Fixed masks cache their active pixels and bin normalization; dynamic masks
recompute normalization per frame using the same geometry-only coefficients.
Choose `sparse_layout="csr"` in either Integrator constructor, or use
`integrator.set_sparse_layout("csr")`, to request CSR. All weights and sums remain
float64. Fast accumulation can change rounding. Strict arithmetic, azimuthal
spread and 2D preserve their existing compatible paths. Mode/layout switches
retain valid tables; mask/correction/geometry edits invalidate dependent caches.

### Prebuilt wheels

Release wheels cover CPython 3.11–3.15 on Linux x86-64/ARM64, macOS Intel/Apple
Silicon and Windows x64. Starting with 0.10.0, Windows ARM64 wheels are also
available for CPython 3.13–3.15. Use `pip install --only-binary=:all: dioptrin` to
require an available wheel.

### pyFAI-compatible full pixel splitting (0.12.0)

Use `method="pyfai_full"` for approximate pyFAI-compatible full splitting in
both 1D patterns and 2D cakes:

```python
ai = dioptrin.Integrator("calibration.poni", method="pyfai_full", unit="2th_deg")
pattern = ai.integrate1d(image, 2000, errors=True)
cake = ai.integrate2d(image, 2000, 360, errors=True)
```

The default `pixel_split` method retains its fractional polygon convention.
`histogram` remains the separate unsplit legacy method. Singular beam pixels
and negligible boundary coverage are handled safely rather than reproduced
bit for bit. See the [validation notes](https://github.com/CPrescher/Dioptrin/blob/v0.12.0/docs/validation/pyfai-full.md).

### Azimuthal brackets and corrected sector ranges (0.13.0)

`integrate2d`, `batch2d`, `batch2d_iter` and `MultiIntegrator.integrate2d`
accept the keyword `azimuthal_bracket` (radians) instead of `n_azimuthal`,
giving rows as centre ± half-width:

```python
cake = ai.integrate2d(image, 2000, azimuthal_bracket=np.radians(2.5))
# 72 rows centred on 0°, 5°, …, 355°
```

Behaviour changes to check when upgrading:

- An explicit `azimuthal_range` in 2D calls is now the true `(min, max)` range
  in radians, sectors included. It was previously read as (first-bin centre,
  step); code that worked around this must pass `(min, max)`.
- 1D `pixel_split` weights each pixel by the fraction of its own area in a bin,
  so intensities change slightly (closer to pyFAI full splitting).
  `MultiIntegrator` 1D `pixel_split` and `supersampled` patterns change where
  detectors share a bin with unequal coverage.
- Invalid input raises `ValueError` or `TypeError` instead of being ignored,
  e.g. `method="histogram"` with `radial_range`, a `polarization_factor`
  outside [-1, 1], or a q unit without a positive wavelength.

See the [changelog](https://github.com/CPrescher/Dioptrin/blob/v0.13.0/CHANGELOG.md)
for the full list.

### Batch streams and input checks (0.14.0)

`batch1d_iter` and `batch2d_iter` over a list of arrays now run on
`num_workers` workers with `num_threads` threads each, like a generator-fed
stream, and copy each frame only when the stream reaches it. With 8 workers
on 1M frames this was 2.1–2.5 times faster on an Apple M4 Pro. Generator-fed
streams reuse the workers of the previous finished stream on the same
`Integrator` (`clear_file_workers()` releases them), and 2D `pixel_split`
cakes are identical for every thread count.

Behaviour changes to check when upgrading:

- Do not change arrays in place while a list-fed stream is open. Lists of
  small frames can be slower when `num_workers` leaves cores idle; raise it
  or use `batch1d`.
- A single path (`str`, `bytes`, `os.PathLike`) passed where a list is
  expected raises `TypeError`; wrap it in a list.
- Colour (RGB, RGBA, CMYK) TIFFs are rejected with `ValueError`, or the
  file's `BatchResult.error` in batches, instead of being integrated with
  meaningless values. Read one channel with `tifffile` and pass it as a
  float64 array.
- A TIFF in a batch whose shape does not match the mask fails as its own item
  with both shapes in `BatchResult.error`.

The Rust API has breaking changes (streaming batch functions, `Geometry`
shape accessors, `histogram_parallel` removed); see the [changelog](https://github.com/CPrescher/Dioptrin/blob/v0.14.0/CHANGELOG.md)
for these and the full list.

### Native dtypes, tiled CPU batches, and optional GPU backends (0.15.0)

- CPU `integrate1d`, `integrate2d`, `batch1d`, and `batch2d` accept native
  `float32`, `uint16`, `uint32`, and `int32` arrays directly alongside
  `float64`, and fixed-mask CSR batches use 4-frame SIMD tiling.
- `radial_range` now applies to 2D integration across all methods, and
  `integrate1d` / `integrate2d` release the GIL during integration.
- Experimental **CUDA** and macOS **Metal** 1D/2D plans with native dtypes,
  4-frame tiling, `InputOrder::FlipVertical`, and GPU Bitshuffle/LZ4 decode.
  Starting with 0.15.2, prebuilt Linux (x86-64/ARM64) and Windows x64 wheels
  include `cuda` with embedded forward-compatible `compute_70` PTX (requiring
  only an NVIDIA display driver on Volta+ GPUs, no CUDA Toolkit), and macOS
  wheels include `metal`.
- **Breaking:** the legacy no-op `validate_license()` and `License*Error`
  names have been removed.

See the [changelog](https://github.com/CPrescher/Dioptrin/blob/v0.15.2/CHANGELOG.md)
for the full list.
