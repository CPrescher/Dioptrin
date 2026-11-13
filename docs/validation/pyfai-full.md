# pyFAI-compatible full splitting — 2026-11-08

`IntegrationMethod::PyfaiFull` and Python `method="pyfai_full"` add the
previously missing 1D full-histogram convention and select the existing 2D
compatibility kernel. Default polygon splitting remains unchanged.

## Evidence

- Rust golden weights from independent pyFAI 2026.9.0 `fullSplit1D_engine`,
  plus fixed CSC, block workers, dynamic masks, method/cache switching,
  ordinary/shared-grid cakes, and a singular beam-pixel variance test.
- `python_examples/test_pyfai_full.py`, run against a separately built wheel:
  eight matched cases spanning 2θ radians/degrees, Q in Å⁻¹/nm⁻¹ and
  polarization 0/0.6. Both 1D intensity/counting std and 2D intensity/counting
  std are compared with pyFAI full-splitting engines on identical effective
  edges. Typed TIFF and method switching must reproduce array results.
  The maximum observed absolute 1D intensity difference was 3.98e-5 for this
  synthetic 24 × 32 image (signals about 30–250 counts).
- The desktop's independent reference script also covers 2θ/Q/d,
  solid-angle on/off, sectors, masks, warm plans, saved projects, Batch,
  and combined detectors. Three additional centered-detector cases include
  a populated sector crossing ±180°.

Compatibility is approximate. Geometry/corrections remain float64; radial
raster buffers match pyFAI's float32 accumulation. Singular beam pixels use
nonnegative detector-plane fractions, and negative numerical fractions are
rejected. One centered-detector cake comparison has two different populated
bins associated with pyFAI reference weights −5.55e-9 and 3.75e-8; the native
and reference intensities there are ratios of negligible/invalid coverage sums.
All bins with absolute reference weight ≥1e-6 pass the stated cake comparison;
all differences are retained in the desktop receipt. This is not a universal
binwise accuracy bound or bitwise reproduction claim.

The new 1D sector rule includes a whole pixel when its azimuthal extent
intersects the selected interval, following `fullSplit1D_engine`; it is not
fractional sector clipping. The existing `pixel_split` convention is unchanged.

2D `pyfai_full` cakes depend on where the azimuth seam lies, as pyFAI's own
full-splitting engines do; see
[Azimuth seam dependence](pyfai-full-seam.md).

Primary algorithm sources (MIT; notice retained in the implementation):
[splitPixel.pyx](https://github.com/silx-kit/pyFAI/blob/v2025.12.1/src/pyFAI/ext/splitPixel.pyx),
[regrid_common.pxi](https://github.com/silx-kit/pyFAI/blob/v2025.12.1/src/pyFAI/ext/regrid_common.pxi).
