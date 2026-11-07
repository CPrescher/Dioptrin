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
