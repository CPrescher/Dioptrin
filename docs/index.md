# Dioptrin user manual

Dioptrin integrates diffraction images into 1D patterns and 2D cakes using a
Rust engine, with bindings for Python. This manual brings together the guides,
examples and API documentation maintained in this repository.

Start with [installation](installation.md) and the [Python API](python-api.md),
or the [Rust integration examples](rust-api.md). The
[performance guide](PERFORMANCE.md) explains sparse and dense images, CPU
backends, worker counts and the conditions behind the measured rates.

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
file readers for Rust applications. The performance guide identifies which
library and pipeline each setting or measurement belongs to.

Historical [integration benchmark records](https://github.com/CPrescher/Dioptrin/tree/main/docs/benchmarks)
remain in the repository. The performance chapter includes downloadable Windows
reader measurements and provenance.
