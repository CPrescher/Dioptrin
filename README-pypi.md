# Dioptrin

Dioptrin provides fast 1D and 2D diffraction integration using a compiled Rust
engine. It supports pixel splitting, supersampling, detector corrections,
Poisson uncertainties, azimuthal intensity spread, and batch integration.

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

Fast CSR accumulation is enabled by default starting in 0.6.0
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
