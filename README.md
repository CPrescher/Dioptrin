# Dioptrin — Unified LUT API

This repository provides lookup table (LUT) utilities for fast 1D/2D azimuthal integration.

## Unified CSR LUTs

All LUTs are represented in Compressed Sparse Row (CSR) format and share unified structs:

- `lut::Lut1d` — 1D LUT (rows = histogram bins)
- `lut::Lut2d` — 2D LUT (rows = `azi * n_tth + tth`)

Weighted variants (supersampled / pixel-split) use the same structs with `values: Some(weights)`;
unweighted (counting) LUTs have `values: None` and integrate via the mean of contributing pixels.

### 1D Counting

```rust
use dioptrin::lut::{build_lut1d_csr, integrate_lut1d};
let tth = vec![0.1, 0.2, 0.3, 0.4];
let intensity = vec![10.0, 20.0, 30.0, 40.0];
let lut = build_lut1d_csr(&tth, 1024);
let (_bins, y) = integrate_lut1d(&intensity, &lut);
```

### 1D Supersampled

```rust
use dioptrin::lut::{build_lut1d_ss_csr, integrate_lut1d};
use dioptrin::geometry::Geometry;
// let geometry: Geometry = ...;
// let tth: Vec<f64> = geometry.calc_tth_all();
// let lut = build_lut1d_ss_csr(&geometry, &tth, 1024, 4);
// let (_bins, y) = integrate_lut1d(&intensity, &lut);
```

### 1D Pixel-Split

```rust
use dioptrin::lut::{build_lut1d_ps_csr, integrate_lut1d};
let lut = build_lut1d_ps_csr(&tth, 1024);
let (_bins, y) = integrate_lut1d(&intensity, &lut);
```

### 2D Counting

```rust
use dioptrin::lut::{build_lut2d_csr, integrate_lut2d};
let lut2d = build_lut2d_csr(&tth, &chi, 1024, 360);
let y = integrate_lut2d(&intensity, &lut2d);
```

### 2D Supersampled

```rust
use dioptrin::lut::{build_lut2d_ss_csr, integrate_lut2d};
use dioptrin::geometry::Geometry;
// let lut2d = build_lut2d_ss_csr(&geometry, &tth, &chi, 1024, 360, 4);
// let y = integrate_lut2d(&intensity, &lut2d);
```

### 2D Pixel-Split

```rust
use dioptrin::lut::{build_lut2d_ps_csr, integrate_lut2d};
let lut2d = build_lut2d_ps_csr(&tth, &chi, 1024, 360);
let y = integrate_lut2d(&intensity, &lut2d);
```

## Migration

- All old `lut_*` builders have been renamed to `build_*` and unified to return CSR-based `Lut1d`/`Lut2d`.
- HashMap- and Vec-of-Vec-based LUTs are removed from the public API.
- Use `integrate_lut1d`/`integrate_lut2d` for both weighted and unweighted LUTs.

## Integrator Method Selection

Use enum-based selection for 1D/2D in `Integrator`:

```rust
use dioptrin::integration::{Integrator, LutMethod};
// let (x, y) = integrator.integrate_lut_with(&frame, 1250, LutMethod::Counting);
// let y2d = integrator.integrate2d_lut_with(&frame, 1250, 360, LutMethod::Supersampled { n_ss: 4 });
```

## Module Layout

- `src/lut/counting_csr.rs`: unweighted CSR LUT builders (`build_lut1d_csr`, `build_lut2d_csr`) and CSR types.
- `src/lut/pixel_split.rs`: pixel-splitting CSR LUT builders (`build_lut1d_ps_csr`, `build_lut2d_ps_csr`).
- `src/lut/supersampled.rs`: supersampled CSR LUT builders (`build_lut1d_ss_csr`, `build_lut2d_ss_csr`).
- `src/lut/integrate.rs`: shared CSR integrators (`integrate_lut1d`, `integrate_lut2d`) used by any LUT.
