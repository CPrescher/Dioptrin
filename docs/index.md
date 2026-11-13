# Dioptrin user manual

Dioptrin integrates diffraction images into 1D patterns and 2D cakes using a
Rust engine, with bindings for Python. This manual brings together the guides,
examples and API documentation maintained in this repository.

Start with [installation](installation.md) and the [Python API](python-api.md),
or the [Rust integration examples](rust-api.md). The
[performance guide](PERFORMANCE.md) explains fixed and dynamic masks, float64
arithmetic, shared plans and matched integration timings.

```{toctree}
:caption: Getting started
:maxdepth: 1

Installation <installation>
python-api
rust-api
examples
```

```{toctree}
:caption: Guides
:maxdepth: 1

HDF5_USAGE
MULTI_DETECTOR
detectors
validation/pyfai-full
validation/pyfai-full-seam
PERFORMANCE
```

```{toctree}
:caption: Reference
:maxdepth: 1

changelog
BUILDING
```

Dioptrin performs diffraction integration. The separate
[dioptrin-io](https://github.com/CPrescher/dioptrin-io) library provides native
file readers for Rust applications. Its [performance guide](https://dioptrin-io.readthedocs.io/en/latest/PERFORMANCE.html)
contains file-reading comparisons, CPU decoder backends and reader measurements.

Historical [integration benchmark records](https://github.com/CPrescher/Dioptrin/tree/main/docs/benchmarks)
remain in the repository. Native reader measurements and their provenance are
maintained in the dioptrin-io manual.
