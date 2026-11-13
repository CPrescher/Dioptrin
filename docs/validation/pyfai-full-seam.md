# Azimuth seam dependence of pyFAI-style full splitting — 2026-11-13

A full-circle cake should not depend on where the circle is cut: integrating
over `(0, 2π)` or `(-2π, 0)` instead of `(-π, π)` should only rotate the cake
by half a turn. This holds for Dioptrin's own methods, but not for
`pyfai_full`, and it does not hold for pyFAI itself either. `pyfai_full`
reproduces pyFAI's behaviour, including this dependence, on purpose.

## Setup

- Image: `test_data/CeO2_Pilatus1M.tif`, calibration `test_data/v1.poni`.
- Cake: 300 radial × 72 azimuthal bins, 2θ in degrees.
- Reference: the cake with the seam at ±π (Dioptrin range `(-π, π)`; pyFAI
  `setChiDiscAtPi`, `azimuth_range=(-180, 180)`).
- Comparison: the cake with the seam at 0 (Dioptrin `(0, 2π)` and `(-2π, 0)`;
  pyFAI `setChiDiscAtZero`, `(0, 360)`), rows shifted back by 36.
- Metric per bin: `|got − ref| / max(|ref|, 1)`.
- Sector check (Dioptrin only): `(2.5, 2.5 + π/2)` against the same range
  shifted by −2π.
- Versions: Dioptrin 0.12.0 at commit `17cf5ab` (PR 16), pyFAI 2026.9.0,
  NumPy 2.5.4, Python 3.13.5, macOS arm64.

Raw numbers: [`pyfai-full-seam.json`](pyfai-full-seam.json). To reproduce,
run this from the repository root with a Dioptrin wheel, pyFAI and fabio
installed:

```bash
python scripts/compare_azimuth_seam.py docs/validation/pyfai-full-seam.json
```

## Results

Full circle, seam at 0 vs seam at ±π (21,600 bins):

| Integrator | Bins > 10⁻⁹ | Bins > 1% | Largest relative difference |
|---|---:|---:|---:|
| Dioptrin `counting` | 0 | 0 | 0 |
| Dioptrin `supersampled` (n_ss = 3) | 0 | 0 | 0 |
| Dioptrin `pixel_split` | 0–1 | 0 | 1.0e-9 |
| Dioptrin `pyfai_full` | 896 | 83 | 48.8 |
| pyFAI `full/histogram/cython` | 14,686 | 85 | 48.8 |
| pyFAI `full/csr/cython` | 17,486 | 1,184 | 42.6 |
| pyFAI `bbox/histogram/cython` | 14,330 | 92 | 49.6 |

- Most of pyFAI's small differences come from float32 accumulation: their
  median is 4.7e-7. Dioptrin's `pyfai_full` accumulates in float64; the
  median of its 896 differing bins is 1.2e-3.
- **All 83 Dioptrin `pyfai_full` bins that change by more than 1% are among
  the 85 such bins of pyFAI `full/histogram`.** The dependence is the same
  effect in both codes, not something the port added.
- The large differences reach radial bin 205 (2θ ≈ 21° on pyFAI's grid).
  They are not confined to the beam centre or the seam rows: in Dioptrin,
  33 of 72 azimuthal rows have differing bins.
- Quarter-circle sector `(2.5, 2.5 + π/2)` vs the same range shifted by −2π:
  - counting and supersampled: identical;
  - `pixel_split`: 2 bins differ, by up to 2.3e-9;
  - `pyfai_full`: 276 bins differ, 118 of them by more than 1%, by up to
    1.44 relative, out to radial bin 260 (2θ ≈ 27°).

## Cause

pyFAI's full splitting must decide, for each pixel, how to unwrap the four
corner azimuths of a pixel that may straddle the seam before splitting it
over bins. It does not test geometrically where the pixel is. It uses a
heuristic: the sign of a diagonal "area" built from the corners, then a
2π shift when the mean corner angle exceeds π. The heuristic runs on angles
taken relative to the current seam (π or 0). Moving the seam changes which
pixels it shifts, and some pixels away from the seam are unwrapped wrongly
and spread over the wrong bins. The effect is large in nearly empty bins,
such as those at module gaps, where one misplaced pixel fraction dominates
the bin.

This description comes from Dioptrin's port in `src/lut/pyfai_split.rs`,
adapted from pyFAI's `splitPixel.pyx` and `regrid_common.pxi` (v2025.12.1).
The port keeps the heuristic for compatibility. The measured agreement with
pyFAI above supports reading it as pyFAI's own behaviour; the pyFAI source
was not instrumented.

Dioptrin's own methods avoid the problem:
- `counting` and `supersampled` bin points (pixel centres or subpixels), so
  only each point's azimuth matters.
- `pixel_split` unwraps each pixel polygon around its own centre azimuth
  before clipping it against the bins, so the seam position does not affect
  it.

## Consequences

- `pyfai_full` is kept as is: it exists to match pyFAI, and with the default
  seam (`(-π, π)`) it is bit-identical to Dioptrin's default `pyfai_full`
  cake.
- For sectors, or for full circles cut elsewhere than ±π, use `pixel_split`.
  Its result does not depend on how the range is written: differences are at
  most a few 1e-9, from floating-point rounding.
- This applies to every code that uses pyFAI's full-splitting engines: a
  cake's per-bin values near module gaps can change by large factors when
  only the azimuth convention (`chiDiscAtPi` vs `chiDiscAtZero`) changes.
