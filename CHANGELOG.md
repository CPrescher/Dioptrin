# Changelog

All notable changes to Dioptrin are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The Rust crate is prepared for crates.io. `Cargo.toml` now lists the
  published files (`src/`, README, changelog, `LICENSE` and the pyFAI notices
  in `licenses/`), which shrinks the package from 336 files and 22.1 MiB to
  83 files and 4.2 MiB; test data, docs, examples, benches and integration
  tests stay in the repository only. It also gains keywords, categories, the
  manual as homepage and docs.rs settings. The library code is unchanged.
  maturin builds the PyPI sdist from the same list, so the sdist shrinks from
  335 to 85 files; it keeps everything a source build needs, and
  `[tool.maturin] include` adds the `dioptrin.pyi` stub.
- Pushed release tags now also publish the crate to crates.io (trusted
  publishing), after the GitHub release and the PyPI upload; a PyPI failure
  does not stop it. A new `crate` job runs `cargo publish --dry-run` on pull
  requests and tags, and on tags checks that the tag, `Cargo.toml` and
  `pyproject.toml` versions agree; the GitHub release and both uploads wait
  for it. A re-run skips a version that is already on crates.io.

### Fixed

- Generator-fed `batch1d_iter` and `batch2d_iter` streams no longer build a
  new worker pool (one thread and one Rayon pool per worker) each time
  (#32): a stream reuses the workers of the previous generator-fed stream on
  the same `Integrator` once that stream's workers are done. A stream that
  is still open keeps its workers, and `clear_file_workers()` releases the
  cached ones too. On an Apple M4 Pro (256×256 frames, 8 workers,
  `pixel_split`) opening and closing a stream costs 0.27 ms instead of
  0.50 ms, as in 0.12.0, so 400-frame streams are 4.1% below 0.12.0 instead
  of 6.8%. Long streams of small frames stay about 6% below 0.12.0: that is
  the cost of releasing the GIL while the producer thread waits, which 0.13.0
  introduced so other Python threads keep running; keeping the GIL there (as
  0.12.0 did) costs 10–12% on 4M frames instead.
- `batch1d_iter` and `batch2d_iter` over a list of arrays now run like a
  generator-fed stream (#33): on `num_workers` workers of their own with
  `num_threads` threads each (both were ignored before; 1D used every core of
  the global Rayon pool), reading at most `4 * num_workers` frames ahead.
  Arrays are still checked when the stream is opened, with the same errors,
  but are copied only when the stream reaches them instead of all at once
  under the GIL, so do not change them in place while the stream is open (a
  changed shape raises at that item). Deleting a list-fed stream now stops
  its workers and frees the integrator before `del` returns; 1D used to
  integrate all remaining frames first. A panic while integrating a frame of
  a 1D list stream now gives an error result at that item, as in every other
  stream, instead of ending the stream with `RuntimeError`. Results are
  bit-identical. On an Apple M4 Pro (`pixel_split`) list-fed streams now run
  as fast as generator-fed ones: with 8 workers 1M frames go from 1430 to
  3570 frames/s in 1D and from 1450 to 3080 in 2D, 4M frames from 560 to
  860. Lists of small frames get slower where an explicit or default
  `num_workers` leaves cores idle that the old 1D path used: 256×256 frames
  with the default workers go from 39 500 to 29 300 frames/s, with
  `num_workers=2` from 38 300 to 10 500; raise `num_workers` (or use
  `batch1d`) for those.
- The legacy 2D `pixel_split` cake (`IntegrationMethod::PixelSplit`,
  `build_lut2d_ps_csr`, `build_lut2d_ps_csr_with_azimuthal_range`) is now
  bit-identical across runs and thread counts (#34). Its table was filled from
  parallel threads in scheduling order, so the order of a row's entries, its
  normalization sum and the cake's addition order depended on thread timing;
  on the Pilatus 1M test frame with 14 threads 6–9% of the cake's bins
  differed from the single-threaded result, by up to 3e-13 relative. Each row
  now lists its entries in pixel order, which is the table a single-threaded
  build already produced: single-threaded results are unchanged,
  multi-threaded results now equal them. The 1D `pixel_split` table and the
  other methods' tables were already deterministic.
- `math::histogram::histogram_parallel` now panics when `weights` and `data`
  differ in length, like `histogram` and `histogram_with_errors` (#42). It
  paired them chunk by chunk and silently ignored the extra values: longer
  weights were dropped, and with shorter weights the trailing data was not
  binned at all, so bins reported wrong means (or NaN, 0 with a mask, when
  left empty). A mask shorter than the data still panics and a longer one is
  still accepted, as in the other two.
- Rust projects that depend on the crate no longer build and link an unused
  `cdylib` next to the `rlib` (#45). `Cargo.toml` no longer declares
  `crate-type = ["rlib", "cdylib"]`; maturin already built the Python
  extension with `cargo rustc --crate-type cdylib`, so the Python build runs
  the same command and the macOS wheel's extension module is byte-identical
  to the one built from 0.13.0's manifest. Without the `cdylib`, cargo also
  emits the crate's metadata early, so a dependent can start compiling
  before the crate's code generation finishes.
- A `Geometry` whose public `dim1`/`dim2` were set by hand to a shape its
  attached detector cannot represent now panics with the shape message the
  `Integrator` gives (#43). Before, the two-theta, azimuth, solid-angle and
  corner arrays, `detector_position`, the LUT builders and `refine` returned
  results for a truncated binning on a non-divisor shape, NaN arrays for a
  larger shape on a uniform detector, or panicked with an integer overflow or
  a division by zero. A crop of a uniform detector with a fixed shape is now
  rejected too, as `with_detector` and the `Integrator` already did, although
  one larger than half the detector on each axis used to give correct arrays.
  Shapes `with_detector` accepts give bit-identical arrays, and a geometry
  without a detector, or with one without a fixed shape, accepts every shape
  as before.
- `pyfai_full` decides a detector's corner sign correctly for pixels far
  smaller than any real detector's (#44). The vote over the pixels' signed
  areas in (radial, azimuth) used the shoelace sum of absolute coordinates,
  which turned single votes into rounding noise once a pixel spanned less
  than about 3e-8 rad (pixel size / distance), until the majority flipped: a
  64×64 detector with 1e-12 m pixels at 1 m,
  `rot1 = 1.2` and the PONI 0.5 mm beside it voted −1 (974 pixels against
  962) where +1 is right. The area is now half the cross product of the
  diagonals, the same quantity computed from coordinate differences, which
  keeps every pixel's vote down to about 1e-15 rad. Real detectors (1e-6 rad
  and up) were not affected: over 2,174 detector, orientation, transform,
  binning and beam cases every pixel's vote and every 1D, 2D, shared-grid and
  `MultiIntegrator` result is bit-identical to before.

## [0.13.0] - 2026-11-15

### Added

- Azimuthal brackets for texture analysis: 2D rows can be given as centre ±
  half-width instead of a bin count. Python `integrate2d`, `batch2d`,
  `batch2d_iter` and `MultiIntegrator.integrate2d` accept the keyword
  `azimuthal_bracket` (radians) in place of `n_azimuthal`; the first row is
  centred on the start of `azimuthal_range` (default `(0, 2π)`, end
  exclusive), e.g. `azimuthal_bracket=np.radians(2.5)` gives 72 rows
  centred on 0°, 5°, …, 355°. Rows are contiguous and the span must be a whole
  number of rows. In Rust: `Integrator::integrate2d_on_azimuthal_brackets`
  (and `_from_slice`) and the helper `integration::azimuthal_bracket_bins`.
  `n_azimuthal` is now optional in those Python signatures; existing calls are
  unchanged.
- `Integrator::integrate2d_on_azimuthal_range` (and `_from_slice`) integrates
  2D on true azimuthal edges `(min, max)` in radians, including sectors and
  sectors crossing the ±π cut, and returns intensity, optional errors and the
  radial and azimuthal bin centres (`AzimuthalRange2dResult`). It bins through
  the cached shared-grid tables and keeps the default 2D radial grid. The batch
  task `batch::Integration2DOnAzimuthalRange` runs it per file. The existing
  `integrate2d_with_azimuthal_range*` functions keep their meaning (first-bin
  centre and step, wrapping periodically).
- Accept finite, resolvable azimuthal starts outside ±2π on shared grids and
  true-range 2D APIs. For example, (-3π, -π) uses the equivalent (-π, π)
  calculation and returns the corrected (-π, π) axis. Spans above one full
  turn, such as (-4π, 4π), remain rejected. Previously accepted ranges and
  their documented pyFAI seam behavior are unchanged.

### Fixed

- 1D `pixel_split` (`IntegrationMethod::PixelSplit`, `build_lut1d_ps_csr`,
  `build_lut1d_ps_csr_with_range`) now weights every pixel by the **fraction
  of its own area** that falls into a radial bin (clipped polygon area divided
  by the pixel's quadrilateral area). It
  previously used the clipped polar-plane area itself, which weighted pixels
  within one bin by roughly 1/r and biased bins towards their low-angle side.
  A pixel partly outside the radial range still contributes only its in-range
  fraction.
- The pixel containing or touching the point of normal incidence has no single
  polar quadrilateral. The 1D polygon builder now uses the same detector-plane
  sampling fallback as the shared-grid 2D and `pyfai_full` paths; before, the
  innermost bins could receive no weight from it at all.
- `MultiIntegrator` 1D merging with `pixel_split` or `supersampled` gave every
  detector the same total weight in a bin, however many pixels it contributed
  there, because the 1D tables were normalized per bin. The tables
  (`build_lut1d_ps_csr*`, `build_lut1d_ss_csr*`) now hold plain pixel fractions,
  so merged bins weight every pixel by its in-range fraction. Counting and
  `pyfai_full` tables were never normalized per bin.
- 1D `counting` and `supersampled` with an explicit `radial_range`
  (`build_lut1d_csr_with_range`, `build_lut1d_ss_csr_with_range`) moved pixels
  and subpixels below the range into the first bin and those above it into the
  last bin. They are now dropped, as `pixel_split` and `pyfai_full` already
  did. A supersampled pixel partly outside the range keeps only its in-range
  fraction. Both range limits are included: a value exactly at `radial_max`
  belongs to the last bin. Integration without `radial_range` is not affected.
- `IntegrationMethod::Histogram` (and `math::histogram::histogram`,
  `histogram_with_errors`, `histogram_parallel`) panicked with an index out of
  bounds when the largest radial value rounded to bin index `n_points`. This
  happened for many bin counts when the radial maximum was large (roughly 30
  or more), for example 2θ in degrees or q in nm⁻¹ on the Pilatus 1M test
  geometry. The maximum now lands in the last bin; all other values are binned
  as before. `histogram_parallel`, which the integrator does not use, did not
  panic in that case but silently lost the sixteenth of the data that contained
  the maximum; it now returns the complete result.
- 2D `counting` and `supersampled` without an explicit azimuthal range
  (`build_lut2d_csr`, `build_lut2d_ss_csr`) wrapped pixels and subpixels at or
  slightly beyond the top of the automatic azimuthal grid into the first
  azimuthal bin; they now go to the last. Values reach that point by rounding
  to bin index `n_azimuthal`, or, on detectors rotated about the beam, by
  interior subpixels lying above the grid maximum by a small fraction of a
  bin. Counting 2D is not affected on real geometries (the range padding
  covers radian azimuths). Supersampled 2D changes the first and last
  azimuthal bin at one radius, e.g. one subpixel at +π on the `v1.poni`
  Pilatus 1M geometry; every other table entry is unchanged. The
  `_with_azimuthal_range` variants still wrap periodically.
- `math::histogram::histogram_parallel` never binned the last `len % 16`
  values (and returned all NaN for fewer than 16 values) because it split the
  input into 16 equal chunks and dropped the remainder. It now bins every
  value with rayon, without copying the input or silently swallowing worker
  panics, and keeps at most 32 partial histograms whatever the input length.
  Bins and counts match `histogram`; the bin sums hold the same terms but
  are added in a different order, so they can differ by floating-point
  rounding (relatively more where weights within a bin nearly cancel). The
  result is now identical from run to run and for any thread count.
- `IntegrationMethod::Histogram` in 2D (every `Integrator::integrate2d*`
  method, the typed TIFF 2D path, the `batch` helpers with a 2D task, and
  Python `integrate2d`, `batch2d` and `batch2d_iter`) ignored `n_azimuthal`
  and the azimuthal range and returned the 1D radial profile: `n_radial`
  values instead of `n_radial * n_azimuthal`, which could not be reshaped
  into a cake (or was silently reshaped to a single row). Histogram has no
  azimuthal binning, so the `integrate2d*` methods now panic with
  "`<method>` is not supported for Histogram method"; `batch::process_files`,
  `process_files_sequential`, `process_files_streaming` and
  `process_queue_streaming` return that message as an `Err` for every input
  without integrating; and Python raises `ValueError` before any work starts.
  1D Histogram integration is unchanged.
- Python `Integrator.integrate2d`, `batch2d` and `batch2d_iter` (path and
  array input, with and without errors) passed the documented
  `azimuthal_range=(min, max)` to Rust as (first-bin centre, step), so
  `(0, 1.57)` with 90 bins put the whole circle into five bins of 1.57 rad.
  They now integrate the true range: the bins split `[min, max)` evenly, pixels
  outside are left out, and `result.azimuthal` holds the bin centres. Invalid
  ranges (non-finite, `max <= min`, span above 2π) raise
  `ValueError`; batches check before starting.
  Non-finite pixels behave as without a range: they make only the bins they
  contribute to NaN or infinite. Radially nothing the default path keeps is
  lost: with counting, the pixel at the largest radial value lands in the last
  bin, also for units whose maximum is large (2θ in degrees, q in nm⁻¹); with
  supersampling, subpixels below or above the default radial edges (next to
  the beam centre, or a few ulp past the outermost one) go into the first or
  last radial bin, as in the default table. A pixel or subpixel inside
  the range whose bin index rounds to `n` lands in the last bin instead of
  being dropped. A full circle may start anywhere: `(0, 2π)` and `(-2π, 0)`
  give the `(-π, π)` cake rotated by half a turn (identical for counting and
  supersampling, within about 1e-9 relative for `pixel_split` on the CeO2
  image).
  With `pyfai_full`, individual sector bins depend on how the range is
  written: on the CeO2 Pilatus 1M image (300×72 bins, 2θ in degrees),
  `(2.5, 2.5 + π/2)` and the same range shifted by −2π differ in 276 of 21,600
  bins (up to 2θ ≈ 27°, by up to 1.44 relative), because pyFAI's full
  splitting unwraps pixels with a heuristic that depends on where the azimuth
  seam lies (pyFAI itself behaves the same). `pixel_split`
  differs in 2 bins by at most 2.3e-9 there and is the method to use for sectors.
  The same applies to where a `pyfai_full` full circle starts: `(0, 2π)`
  differs from the rotated `(-π, π)` cake in 896 of 21,600 bins there (median
  0.1%, 83 bins by more than 1%). pyFAI 2026.9.0's own full splitting shows the
  same: all 83 bins are among the 85 that change by more than 1% between
  `setChiDiscAtPi` and `setChiDiscAtZero`. See
  `docs/validation/pyfai-full-seam.md`.
- 1D `pixel_split` and `pyfai_full` without an explicit `radial_range` lost
  part or all of the pixel containing the point of normal incidence: the
  automatic range starts at the smallest pixel-corner value, and the part of
  that pixel closer to the beam was dropped (the whole pixel when the beam sits
  at its centre). That part now goes into the first bin, so the pixel keeps its
  full weight, as with counting and supersampling. The automatic bin edges are
  unchanged. With an explicit `radial_range` (and so in `MultiIntegrator`) the
  pixel still contributes only its in-range fraction.
- `integrate2d_with_azimuthal_range*` (first-bin centre and step) with
  `pyfai_full` panicked with "valid pyFAI cake grid" when the first bin edge
  lay outside `[-2π, 2π]` or the bins spanned more than a full turn. A start
  outside that window is now reduced modulo 2π (a start too large to reduce
  exactly panics with a message that says so). A span above 2π still
  panics, with a message that says so: `pyfai_full` treats the bins as the
  sector `[start - step/2, start + (n - 0.5)·step)` and, unlike the other
  methods, does not fold angles outside it into the bins. This was
  undocumented and is now stated in the function docs.
- The seam test of 1D `pyfai_full` used the signed diagonal area
  `0.5·(X − Y)` where pyFAI (and the 2D port) use `0.5·X − Y`. It now matches
  pyFAI. The two differ in sign only when the pixel's diagonals do not cross
  in (radial, azimuth) space, i.e. for folded quadrilaterals, so patterns are
  not expected to change.
- `pyfai_full` (1D and 2D) was wrong for a `Detector` with orientation 2 or 4
  attached through `Geometry::with_detector` (Rust only; not reachable from
  Python or the detector catalogue), since 0.12.0. These orientations mirror
  one detector axis, so every pixel's corners run round it the other way in
  (radial, azimuth) space. pyFAI multiplies its seam test by a corner sign for
  this (`_corner_sign` in `regrid_common.pxi`); the port did not. Pixels
  crossing the ±π seam were therefore not unwrapped: in 1D their weights were
  misplaced by up to 100% between neighbouring radial bins, which moved
  patterns by about 1e-5 relative (up to about 1e-3 on small images)
  whenever the seam crosses the detector. In 2D, in addition, every raster
  area came out negative and was discarded, so almost every pixel fell back
  to exact polygon clipping instead of pyFAI's raster rule, and seam pixels
  were smeared around the whole circle: nearly every cake bin was wrong. With
  the seam on the detector this reached 200% with the beam on it; with the
  beam off the detector but the seam on it, cakes differed from pyFAI by
  5.6e-2 to 0.18 on the bins both fill, and two thirds of the bins (1024 of
  1536 in a test) were filled in one and empty in the other. With the seam
  off the detector they differed by about 2e-2.
  This affected every `pyfai_full` path: 1D (with or without `radial_range`,
  and `build_lut1d_pyfai_csr` sectors), cakes, `integrate2d_with_azimuthal_range*`,
  `integrate2d_on_azimuthal_range*`, brackets, `integrate2d_raw_on_grid*` and
  `MultiIntegrator`, plus `PixelSplitMode::PyfaiHistogram` shared grids. The
  sign is now measured on the pixel corners the kernels receive, which equals
  pyFAI's orientation rule for uniform and separable detectors and also covers
  transformed maps, whose `orientation` does not describe their corner order.
  Orientations 1–4 now agree with the default orientation on the flipped image
  to f32 rounding (about 1e-7), and with pyFAI 2026.9.0
  (`("full", "csr", "cython")` and its histogram engines) to about 1e-6 when
  no pixel contains the point of normal incidence. With the beam on the
  detector every orientation, the default included, differs from pyFAI by
  1e-3 to 4e-3 in 1D and 4e-3 to 5e-3 in 2D, because such singular pixels
  are sampled on the detector plane (unchanged).
  Results for the default orientation are unchanged, bit for bit.
- An `Integrator` with a `Detector` of fixed shape (`max_shape`) attached
  through `Geometry::with_detector` accepted images of any shape, although
  only the detector's shape and its integer binnings (with a defined pitch,
  if the detector lists pitches) are defined; the
  binning was checked once, for the shape given to `Geometry::new`. Rust only
  (Python attaches no detector), since 0.12.0. A larger image panicked in
  `Detector::corners` with an arithmetic overflow (debug) or indexed out of
  bounds (release) for separable, corner-map and transformed detectors and
  gave degenerate results or unrelated errors for spline and uniform ones; a
  shape that is not an
  integer divisor, e.g. 7 rows of a 12-row detector, was integrated silently
  with truncated binning. `Integrator`, `MultiIntegrator` and the batch
  helpers now reject such an image with a message naming both shapes (and,
  for a crop, the remedy; see "Compatibility and validation"): methods that
  return a `Result`
  (shared-grid, azimuthal range and bracket, `MultiIntegrator` 2D,
  `prepare_fixed1d`, `prepare_dynamic1d`, `prepare_fixed2d_on_grid`) return
  it, the others panic with it, as they do for a mask of the wrong shape, and
  the batch helpers report it for that item (without running the task) and
  integrate the others.
  Detectors without `max_shape` still accept any shape, and results for valid
  shapes are unchanged, bit for bit. Not checked: `Geometry` methods and the
  public LUT builders on a `Geometry` whose public `dim1`/`dim2` are changed
  after `with_detector`, and `Detector::corners`/`position` called directly
  with another shape.
- `Detector::position` on a transformed map (`Detector::transformed` of a
  non-uniform or spline detector, i.e. Dioptas image flips and rotations)
  shifted an offset inside a pixel instead of flipping or rotating it with the
  pixel, so positions on pixel edges were one pixel off (and the last edge
  folded back) and points inside a pixel were misplaced by up to a pixel. On
  binned images (`max_shape` an integer multiple of the image shape) the
  binned pixel's corners were taken in the base detector's order but
  interpolated in the image's, with the same effect. Binned spline detectors
  under a transform were wrong for every method, since the binned branch
  interpolated spline-mapped corners and so displaced pixel centres too
  (in tests, counting 1D raw sums by up to 1e-1, normalized 1D patterns with
  an explicit range by 1.7e-2, supersampled shared-grid tables by 1e-2 and
  solid angles by 9e-6). Otherwise pixel centres and per-pixel corners were
  right, so 1D tables (except
  supersampled automatic ranges, below), the legacy 2D counting and
  `pixel_split` cakes and the counting and supersampled
  shared-grid tables (`integrate2d_raw_on_grid*`, ranges, brackets,
  `MultiIntegrator` 2D) were not affected. Wrong by order one were the shared
  corner grid, which only `pixel_split` and `pyfai_full` use (every
  `pyfai_full` cake and, for both methods, `integrate2d_raw_on_grid*`,
  `integrate2d_on_azimuthal_range*`, brackets and `MultiIntegrator` 2D), and
  the legacy supersampled cake (`integrate2d` with `Supersampled`, wrong by
  0.77–0.98 of the largest bin for some transforms in tests), whose
  automatic radial and azimuthal ranges are taken from the outermost
  subpixels. The same radial range moved 1D supersampled patterns (by up to
  a few 1e-2 in tests). Calibration refinement evaluates 2θ at sub-pixel peak
  positions, so it fitted transformed maps against peaks misplaced by up to a
  pixel along a flipped axis. Rust only, since 0.12.0. A transformed map now
  evaluates its base detector at the inverse-mapped point of the base image
  with the same binning, so a binned pixel spanning a module gap or several
  unequal or curved base pixels has the base's binned quadrilateral, matching
  its corners and mask. One exception remains: pyFAI's seam test, used by
  `pyfai_full` 2D and `PixelSplitMode::PyfaiHistogram`, is invariant under
  mirrors and 180° rotations of the corner order but not under a quarter
  turn, so with a net quarter-turn transform (for example Left on an
  orientation-3 base or Transverse on orientation 2) a pixel next to the beam
  can be split differently than on the base detector (one pixel in a test,
  moving raw sums by about 1e-1 in the bins it touches). That is pyFAI's
  rule, not a position error, and not new.
- Solid angles of a transformed non-flat detector (curved separable or NeXus
  corner maps) whose base has orientation 1, 2 or 4 did not match the base
  detector. The base detector, like pyFAI's `cos_incidence` (which takes the
  normal from `get_pixel_corners()` without re-ordering it for the
  orientation), takes the normal of the pixel at the same image index, which
  for these orientations is not the physical pixel's normal; the transformed
  map took the physical one. It now follows the base's (pyFAI's) convention,
  so a transformed image integrates exactly like the original one. This is a
  consistency fix: the solid angles it reproduces are pyFAI's convention, not
  the physical normal. The change grows with curvature (8e-2 relative in one
  test, 0.4 to 7.5 in strongly curved ones). Rust only, since 0.12.0.
- `MultiIntegrator` 2D and `integrate2d_raw_on_grid*` dropped a value inside
  the grid whose bin index rounded to `n` (for example the value one ulp below
  the upper limit of `(0.5, 25)` with 1500 bins, or an azimuth a few 1e-16
  below 0 on a `(0, 2π)` grid). It now lands in the last bin. The limits stay
  half-open: values at or above the upper limit, and outside a sector, are
  still dropped.
- Python: invalid input now raises `ValueError` or `TypeError` instead of a
  Rust panic (`PanicException`) or being ignored:
  - `batch1d_iter` and `batch2d_iter` stopped silently at an item they could
    not read (a list of non-float64 arrays yielded nothing; a bad item from a
    generator truncated the stream; an exception raised by the generator
    produced a bogus extra result). Lists are now checked before work starts,
    and for generators the error, or the generator's own exception, is raised
    from the iterator at that item after the earlier results. A panic while
    integrating a frame of `batch1d_iter` over a list of arrays ended that
    iteration silently; it now raises `RuntimeError` at that item, after the
    earlier results, once the frames still being integrated are done. All
    other streams (TIFF paths, generator input, `batch2d_iter` over arrays)
    already turned such a panic into an error `BatchResult` ("integration
    worker panicked") at its position and went on; they still do.
  - An image whose shape differs from the mask raises `ValueError` for array
    inputs on every `Integrator` and `MultiIntegrator` call. TIFF path batches
    still report it per item as a worker failure.
  - `MultiIntegrator.integrate1d` and `batch1d` accept lists that mix TIFF
    paths and arrays, as `integrate2d` already did.
  - Zero bin counts (`n_points`, `n_radial`, `n_azimuthal`), `n_ss=0`, an empty
    image, a `radial_range` that is not finite with min < max, a
    `polarization_factor` outside [-1, 1], and q units without a wavelength in
    the calibration are rejected. The last four are checked when the
    integrator is built or reconfigured, not at integration time.
  - `method="histogram"` together with a `radial_range` raises `ValueError`
    (at construction and in `set_method`). The Histogram method always uses
    the automatic radial range, so the range was silently ignored and the
    full pattern returned. The Rust API still ignores it; this is now
    documented on `Integrator::with_radial_range`.
- Synchronous batches (`batch1d`, `batch2d`, `MultiIntegrator.integrate1d`
  and `batch1d`) release the GIL while they wait, and `batch1d_iter` /
  `batch2d_iter` fed by a Python iterator run on their own worker pool, so an
  open or abandoned stream no longer blocks a later batch on the same
  integrator.
- Python `batch1d_iter` / `batch2d_iter` fed by a generator or other iterator:
  - Deleting the `BatchIterator` (or reaching its end, including by an
    exception) now stops the stream's workers: they release the integrator
    before it returns, so its `set_*` methods work again at once, and then
    shut down their threads.
    Previously the workers waited for the generator's next item, which a
    generator blocked on a live feed never yields: each abandoned stream kept
    its threads (13 with `num_workers=4, num_threads=2`) and left the
    integrator "in use by a batch operation" for the rest of the process. The
    one thread reading the generator cannot be interrupted while it is inside
    the generator's own `__next__`; it stays there until that call returns,
    then exits without pulling more, and holds neither the integrator nor the
    workers meanwhile.
  - A stream now reads at most `4 * num_workers` items ahead of the results
    handed out. An endless generator that nobody consumed was pulled as fast
    as the workers ran, its results accumulating in memory without bound
    (tens of thousands of frames within seconds). The workers themselves
    never wait on the consumer, so an open stream still does not block other
    batches.
  - A Rust panic while reading the next item (for example a pyo3
    `PanicException` raised in the generator, as when it calls the
    `Integrator` from the stream's thread) ended the stream as if the
    generator were exhausted. It now raises `RuntimeError` at that item,
    after the earlier results, and so does any other loss of the reading
    thread: a generator stream only ends cleanly when the generator does.
- Python `batch1d_iter` / `batch2d_iter` over a list of arrays or TIFF paths
  release the integrator before they report the end of the stream, so its
  `set_*` methods work right after the loop. This depended on the unspecified
  order in which a finished worker dropped its captures; 1D over arrays could
  still raise "Integrator is in use by a batch operation" for a moment after
  `StopIteration`.
- The documented range of `polarization_factor` was 0 to 1; it is −1 to 1
  (+1 horizontal, −1 vertical, 0 unpolarised), which is what the code accepts.
- `dioptrin.pyi` now covers the calibration API (`Calibrant`,
  `CalibrationPoints`, `refine`, `auto_calibrate`, `find_peaks_image` and the
  related functions and result classes), which was missing from the stubs.
- One `Integrator` used from several threads with images of **different
  shapes** (Python file batches and list/generator `_iter` calls with workers,
  or Rust threads sharing an `Arc<Integrator>`) could crash with a segmentation fault, panic with
  "Lut2dGuard created without LUT being built", hang, or silently integrate an
  image with the lookup table of another shape. The cached geometry and lookup
  tables belonged to whichever shape was requested last, and a call could pick
  up a table that another thread had just rebuilt for its own image. Every call
  now works on the geometry and table of its own image shape, so mixed shapes
  give the same results as one integrator per shape. Each 1D/2D cache still
  keeps one table, so switching shapes can rebuild it; warm, fixed-shape batches
  continue to share their prepared tables. Mixed-shape array lists in Python
  `batch1d` now use parallel processing too; they previously ran serially.
  Histogram prewarming retains its shared geometry, and Histogram integration
  borrows the immutable radial snapshot instead of copying it for every frame.
- Calling `with_radial_unit` on an already-used Rust integrator now refreshes
  its cached radial coordinates, like `set_radial_unit`. Previously it could
  rebuild tables using the old unit's coordinates. A wrong-shape mask is also
  rejected before publishing geometry for the rejected image.
- The lookup-table kernels read pixels without bounds checks and trusted the
  table to fit the image. Prepared tables now verify once per call
  that the image, mask, normalization and variance arrays cover every pixel
  the table addresses; public caller-built tables validate each gathered index.
  Invalid inputs panic with a clear message. This also covers the public
  `lut::integrate1d*`/`integrate2d*` functions, which accept caller-built
  `Lut1d`/`Lut2d` values and could read out of bounds from safe code.
  Public caller-built tables pay for a check per gathered index; cached
  Integrator/plan tables retain constant-time array-length checks.
- The built-in circular mask of a detector with `circular_mask` under
  `Detector::transformed` differed from the base detector's mask, mapped
  through the same flip or rotation, on binned images since detectors were
  added in 0.12.0. Affected were MAR 345 at odd binned shapes, and MAR 345 and
  Rayonix 133 / SX165 at unequal binnings under `Transpose`, `Transverse`,
  `Left` or `Right` (and custom circular detectors of odd or non-square shape
  at any binning other than 1): the circle was drawn at full resolution and reduced
  block-wise, or redrawn on the transformed image with the radius of the
  wrong axis, whereas pyFAI draws it on each binned image with an
  integer-division centre that is not symmetric under flips. Examples of
  wrong pixels: MAR 345 at binning 2 (1725 × 1725) under `Half`: 4,878;
  MAR 345 at binning [1, 3] (3450 × 1150) under `Left`: 2,854,004;
  Rayonix 133 at binning [1, 2] (4096 × 2048) under `Left`: 4,730,336. The
  transformed detector now evaluates the base's circle on the base image of
  the same binning, so it masks exactly the base's pixels at every binning;
  uniform maps record the operation in a new serialized field,
  `circular_mask_transform`, written only for transformed circular detectors.
  Masks of untransformed detectors are unchanged; like the mask intervals,
  the circle follows the detector orientation, whereas pyFAI's does not
  (catalogue detectors, with orientation 3, are unaffected). Transformed
  detectors serialized by earlier versions: uniform ones (catalogue MAR 345
  and Rayonix) and those of odd or non-square shape load with their old mask
  (transform the base again for the corrected one); non-uniform ones of even
  square shape (separable, corner or spline maps, e.g. a MAR 345 with a Fit2D
  spline) now load with the corrected mask on binned images, unchanged
  unbinned.
- `Detector::clear_spline` on a transformed detector whose base becomes
  uniform now gives the geometry of clearing the spline on the base and then
  transforming: a uniform map in the base orientation. It previously used
  orientation 3, which also changed positions and corners of non-circular
  detectors with base orientation 0, 1, 2 or 4. Fields set on the
  transformed detector (name, aliases, config, pixel sizes, binning pitches,
  `max_shape`, `circular_mask`, mask intervals) are kept; the intervals are
  re-encoded for the new orientation so they mask the same pixels. Nothing
  in the crate calls `clear_spline`.
- Preserve non-finite signal, zero normalization, and signed/zero weights when
  dynamic workers fall back to CSR with supplied variances. The optional spread
  calculation retains its documented sample filtering.
- `math::histogram::histogram` and `histogram_with_errors` panic with a clear
  message when `weights` and `data`, or `variances` and `weights`, differ in
  length, instead of panicking out of bounds or silently ignoring the extra
  data values.

### Compatibility and validation

- **Behaviour change:** 2D integration with the Histogram method now fails
  instead of returning a 1D profile; use `counting`, `pixel_split`,
  `pyfai_full` or `supersampled` for 2D. `IntegrationTask` has a new provided
  method, `integrate2d_method`, which the built-in 2D tasks override; custom
  tasks are unaffected.
- **Behaviour change:** Python `Integrator` 2D calls with an explicit
  `azimuthal_range` return different cakes. Ranges used to be read as (first-bin
  centre, step); they are now true `(min, max)` ranges, sectors included, so
  code that passed (centre, step) to work around the bug must pass
  `(min, max)` instead. An explicit full circle `(-π, π)` follows exactly ±π,
  while the default (no range) grid spans the detector's own azimuth extent, so
  values differ slightly from the default cake; on the CeO2 Pilatus 1M test
  image (500×360 bins, 2θ in degrees) the median relative difference per bin is
  2e-5 for counting, 1.8e-3 for `pixel_split`, 1.0e-3 for `supersampled` and
  0 for `pyfai_full`, with larger differences in sparsely filled bins at module
  gaps. On the default path's own edges, counting gives identical values and
  supersampling values equal to rounding, except in the first and last
  azimuthal rows when a subpixel lies just outside those edges (the default
  table clamps it into the end row; a true range does not: up to 0.7% in one
  bin of the CeO2 image at `n_ss = 3`). `pixel_split` differs
  slightly (fractional polygon areas vs the legacy 2D table; pixels touching
  the beam centre are sampled densely). Calls without
  `azimuthal_range` are unchanged.
- **Behaviour change:** multi-detector 1D `pixel_split` and `supersampled`
  patterns change wherever detectors share a bin with unequal coverage.
  Single-detector intensities and errors are unchanged up to floating-point
  rounding, but the raw sums (`integrate1d_raw*`: `sum_signal`, `sum_norm`,
  `sum_variance`, `count`) and the 1D table values are now on the pixel-fraction
  scale instead of normalized per bin.
- **Behaviour change:** with an explicit `radial_range` that does not cover the
  whole detector, the first and last bin of 1D `counting` and `supersampled`
  patterns change, for one detector and for `MultiIntegrator`: they no longer
  contain intensity from outside the range. A detector lying entirely outside
  the shared range now contributes nothing to a merged pattern. The other bins
  are unchanged.
- **Behaviour change:** 1D `pixel_split` intensities and errors change
  slightly for every 1D path that uses the polygon table (ordinary, typed,
  fixed/dynamic plans, CSR/CSC layouts, batch and `MultiIntegrator` 1D). On the
  CeO2 Pilatus 1M test image (1000 bins, 1–30° 2θ) the relative L2 difference
  to pyFAI full splitting drops from 1.0e-3 to 1.2e-4, and to 3.4e-5 (maximum
  deviation 0.16σ, previously 4.8σ) when the two bins at each end of the range
  are left out; there the range-clipping conventions differ on purpose.
  The polygon change itself does not touch counting, supersampled,
  `pyfai_full` or any 2D output.
  The legacy single-detector 2D polygon builder (`build_lut2d_ps_csr*`) is
  unchanged and still uses row-normalized polar areas.
- **Behaviour change:** Python calls that used to run now raise `ValueError`
  (see "Python: invalid input" under Fixed): `method="histogram"` together
  with `radial_range`, at construction and in `set_method` (the range was
  ignored); a `polarization_factor` outside [-1, 1] or non-finite, at
  construction and in `set_polarization_factor` (it was applied as given); a q
  unit without a positive wavelength in the calibration, at construction and
  in `set_unit` (it panicked at the first integration instead). A
  `batch1d_iter` / `batch2d_iter` list that mixes valid items with one that is
  neither a path nor a 2D float64 array (an `int`, an `int32` array) raises
  `TypeError` at the call, before anything is integrated; before, the items
  ahead of it were integrated and yielded and the stream stopped there
  without an error.
- **Behaviour change:** cropped images on an attached `Detector` with a fixed
  shape (`max_shape`, as every catalogue detector has) now fail with the
  "does not fit detector" message (see "Fixed"). When each dimension of the
  image was more than half the detector's, the image used to be integrated as
  the detector's top-left corner; for a uniform detector with the default
  orientation 3 (e.g. the catalogue Pilatus 1M) the result was identical to
  that of an integrator without a detector, so such calls may have looked
  correct. For a cropped image, attach a custom detector without a fixed
  shape, or no detector. Rust only.
- `examples/export_1d_weights.rs` exports single-pixel 1D weights for
  comparison with independent references.

### Performance

- Add reusable `Dynamic1dWorker` execution with supplied variances and changing
  masks, sharing immutable geometry and correction data across frame workers.
- Add `FixedCsr1dInputPlan` for strict uint16/uint32 integration without a
  full-frame float64 conversion, with serial frame-worker and parallel-bin
  single-image entry points.
- Add an opt-in resident 1D CSC/CSR selector using a broad detector-size/bin estimate:
  CSR generally, with a CSC exception above about 8 MP for high bin counts
  without Poisson errors. No runtime timing probes are used;
  streamed/parallel block execution is unchanged.
  Splitting weights are unchanged; ordinary CSR and CSC can differ in rounding.
- Cache a bin-oriented execution table for sufficiently large resident fixed-mask
  1D frames. Preserve CSC coefficients, chunk/lane reductions and Poisson errors
  bitwise while avoiding histogram scatter/merge work. Streamed block and outer
  Rayon batch workers retain CSC; the optional resident table is capped at 128 MiB.
- Add strict `FixedCsr1dPlan` snapshots for compatibility LUTs, caching fixed-mask
  normalization while retaining the original signal and error reduction order.
- Add `prepare_fixed2d_on_grid` / `FixedGrid2dPlan` for normalized cached cakes,
  validating static geometry once and avoiding unused raw-sum/count output arrays.
  Frame dimensions and unmasked finite samples remain checked on each call.
- Vectorize and parallelize the full-resolution Massif Gaussian background
  filter while preserving each pixel's floating-point summation order.
- Add Rust `PreparedMassif` to reuse immutable samples and component labels
  across seed searches. Existing `find_peaks` remains available and preserves
  peak ordering, deduplication and subpixel positions. The shared filter also
  accelerates image-wide calibration candidate extraction.
- The legacy 1D Histogram method keeps each bin's sum and count together:
  through Python on a Pilatus 1M frame, single-threaded on an Apple M4 Pro,
  3.33 → 2.91 ms per frame (0.12.0: 3.11 ms). Results are unchanged bit for bit.
- `pyfai_full` table builds skip the modulo call when wrapping corner azimuths
  that are already in range, which removes most of the cost of the corner-sign
  vote with a detector attached: about 7–8 ms less single-threaded on a
  Pilatus 1M frame (1D build 151 → 144 ms). Results are unchanged bit for bit.

## [0.12.0] - 2026-11-09

### Added

- `IntegrationMethod::PyfaiFull` / Python `method="pyfai_full"`: approximate
  pyFAI-compatible full pixel splitting for ordinary 1D and 2D integration,
  typed input, and prepared fixed/dynamic plans. The sector-aware 1D LUT
  builder is public for native adapters with independent ranges and variances.
- Physical detector descriptors in Rust, including frozen named-detector
  geometry/masks, orientations, module gaps and curved/separable pixel maps,
  distortion splines, and Fit2D conversion. These are the numerical APIs used
  by the Rust Dioptas desktop; see [detector geometry](https://github.com/CPrescher/Dioptrin/blob/v0.12.0/docs/detectors.md).
- Explicit independent pixel variances for shared-grid 2D and multi-detector
  integration, accumulated with squared pixel weights and the existing
  per-detector normalization.

### Fixed

- Preserve physical pixel-corner geometry throughout integration and calibration
  refinement; share compatible detector coordinates without discarding module
  boundaries or independent corners.
- Include detector-data and pixel-splitting attribution notices in wheels and
  source archives.
- Keep archived benchmark dependency paths relative so source distributions
  remain portable outside the original development checkout.

### Compatibility and validation

- Existing polygon/subpixel defaults remain unchanged. The new full-splitting
  mode is approximate compatibility: singular beam pixels use nonnegative
  detector-plane sampling, and negative numerical raster fractions are rejected.
  Tiny-coverage boundary bins can differ from pyFAI. See the
  [full-splitting validation](https://github.com/CPrescher/Dioptrin/blob/v0.12.0/docs/validation/pyfai-full.md).
- Validate 1D/2D output against independent pyFAI engines in all four supported
  library radial units, plus fixed/dynamic plans, typed frames and detector
  coordinate/mask reference fixtures. The Rust desktop retains its native
  processing and project workflows; it does not need Python at runtime.

## [0.11.0] - 2026-11-08

### Added

- Add shared-grid multi-detector 2D integration in Rust and Python. Each
  detector retains its own geometry, mask and corrections; raw signal,
  normalization and Poisson variance are merged before final division.
- Default the new pixel-splitting path to fractional polygon intersections,
  with periodic seam handling, fractional range clipping and detector-plane
  sampling for pixels containing or touching the beam center. Counting and
  subpixel sampling are also supported; pyFAI histogram compatibility is opt-in.
- Expose per-detector monitor normalization, relative or absolute solid-angle
  scaling, explicit azimuthal sectors and cached integration tables. See the
  [multi-detector guide](https://github.com/CPrescher/Dioptrin/blob/v0.11.0/docs/MULTI_DETECTOR.md) and
  [first-frame/Eiger validation](https://github.com/CPrescher/Dioptrin/blob/v0.11.0/docs/validation/multi2d-startup.md).

### Performance and scope

- On the tested Eiger 9M frame with a 2000 × 360 grid, native fractional
  polygons took 0.729 s for the first frame and a median 24.01 ms for cached
  frames on an Apple M4 Pro using eight threads. Input decoding and application
  display are excluded; the validation report records the complete scope.
- This release supplies the Rust and Python integration APIs for applications
  such as Dioptas. It does not change Dioptas itself. Existing ordinary
  single-detector 2D and all 1D integration retain their prior conventions.

## [0.10.0] - 2026-11-07

### Added

- Expose immutable float64 `Fixed1dPlan` snapshots and lightweight workers for
  fixed-mask CSC integration directly from decoded pixel blocks. Plans share
  geometry, coefficients and normalization across workers.
- Support multicore integration within one frame through validated contiguous
  ranges and deterministic merging of private float64 histograms. Original
  uint16/uint32 counts can be consumed without a full-frame float64 copy.
- Validate complete block coverage, plan identity and frame lifecycle before
  publishing results. Failed or incomplete frames cannot leak into reuse.

- Publish native Windows ARM64 wheels for CPython 3.13–3.15, alongside the
  existing Linux, macOS and Windows x64 wheels.

### Fixed

- Publish cold-start geometry and LUT invalidation together so concurrent batch
  workers cannot clear another worker's newly prepared lookup table.

### Performance and scope

- These APIs let dioptrin-io decode fixed-mask Eiger frames once for both display
  and integration. In the matched warm-cache Eiger 9M Qt benchmark, median
  open-to-pattern latency decreased from 180–182 ms to 143–145 ms. Dynamic masks
  and batch/map processing retain their existing paths.
- All 625 source frames matched independent pixel hashes; float64 intensities
  and Poisson uncertainties agreed within the documented tolerance. See the
  [combined-path benchmark](https://github.com/CPrescher/Dioptrin/blob/v0.10.0/docs/PERFORMANCE.md) for scope and conditions.

## [0.9.0] - 2026-11-06

### Added

- Default fast 1D fixed-mask integration to CSC with cached bin normalization
  and a cached active-pixel selection. Dynamic masks use the same geometry-only
  CSC coefficients with per-frame normalization. All coefficients and sums
  remain float64; fast summation order can change final rounding.
- Select `sparse_layout="csc" | "csr"` in Python constructors, or use
  `set_sparse_layout()`; Rust exposes `SparseLayout`. CSC is the default for
  fast 1D. Strict arithmetic, azimuthal spread and 2D retain compatible paths.
- Use deterministic fixed-mask partitioning across direct and batch APIs,
  preserving output equality across worker counts.
- Keep both geometry layouts and valid caches when only the mask policy or
  requested sparse layout changes. Actual mask/correction/geometry edits
  invalidate only the dependent state.


- Prepare immutable `Dynamic1dPlan` snapshots once and share their geometry,
  float64 CSR/CSC weights and correction arrays across frame workers. Masks are
  explicit per-call inputs; normalization and accumulators remain independent.
- Python `prepare_dynamic1d(shape, n_points)` returns sendable plans with
  lightweight `worker(num_threads=1)` handles. Worker calls borrow contiguous
  float64 images and uint8 masks without copies and release the GIL. Source
  integrator edits do not mutate existing plans.

### Fixed

- Retain valid geometry-only CSC, mask-specific CSR and cached denominators
  when switching mask modes; mask/correction/geometry setters still invalidate
  their dependent state.

- Skip polarization correction recalculation and normalization invalidation when
  `set_polarization_factor()` receives the current value. Changing or clearing
  the factor still refreshes corrections for subsequent integrations.

## [0.8.0] - 2026-11-06

### Added

- Add Rust `MaskMode` and Python `mask_mode="fixed" | "dynamic"`, plus
  `set_mask_mode()`, retaining float64 weights, corrections and accumulators.
- Fixed mode compacts the mask into cached CSR and reuses normalization.
  Editing or clearing the mask refreshes execution state without rebuilding
  detector geometry or original pixel weights.
- Dynamic mode uses geometry-only CSC with four independent histograms and
  fused per-frame normalization/Poisson variance. Strict accumulation and
  oversized sparse tables use fused CSR. Single-image calls can parallelize
  pixels; batches retain frame parallelism.
- Validate mask restoration, policy/correction changes, 1D/2D raw sums,
  concurrent initialization, typed TIFF inputs and Python API behavior.

Fast accumulation may differ slightly from earlier results because mask
compaction and CSC change summation order. Strict accumulation is preserved.

### Performance

Matched measurements against 0.7.0 on an Apple M4 Pro (48 GiB), using the same
Eiger9M frames, geometry, masks, 2,000 bins, full pixel splitting, solid-angle
correction and polarization 0.965. Weights, corrections and accumulators remain
float64. The table reports median throughput across five measurement blocks
with eight workers; fixed-mask measurements use the batch API, while dynamic
measurements replace the mask between single-frame integrations and include
mask creation/update and normalization costs. File reading and decompression
are excluded from these integration measurements.

| Mode | 0.7.0 fps | 0.8.0 fps | Speedup |
| --- | ---: | ---: | ---: |
| Fixed mask, intensity | 227.35 | 379.32 | 1.67× |
| Fixed mask, Poisson errors | 220.11 | 288.50 | 1.31× |
| Dynamic mask, intensity | 106.96 | 236.20 | 2.21× |
| Dynamic mask, Poisson errors | 107.02 | 201.67 | 1.88× |

Fast-mode intensity differences from 0.7.0 have relative L2 errors below
7e-16 in this comparison. See the [matched mask-mode results](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-11-05-mask-modes-validation/conclusions.json)
and [measurement summary](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-11-05-mask-modes-validation/summary.csv).

## [0.7.0] - 2026-11-05

### Changed

- Release the Rust library and Python package as open-source software under MIT.
- Remove license activation, signature verification, expiration checks, and
  license-signing tools from every build and distribution.
- Build unrestricted Python wheels and publish source distributions alongside them.
- Preserve legacy Python license exception names and `validate_license()` as
  compatibility APIs; validation is now a no-op.

Integration algorithms, corrections, and numerical results are unchanged.

## [0.6.1] - 2026-10-24

### Changed

- Reuse typed TIFF decoding buffers within and between file batches, and integrate
  standard 1D/2D CSR file inputs directly from their native sample type. Pixel values
  convert to float64 before the existing arithmetic; weights, accumulation order
  and each strict/fast mode’s results are preserved.
- Retain Python file-batch workers and their buffers for the current worker/thread
  configuration. Add `Integrator.clear_file_workers()` to release cached workers
  and buffers without cancelling already admitted streams.
- Feed 2D array streams concurrently so a full bounded input queue cannot block
  consumer creation.

### Performance

The implementation in 0.6.1 was measured against a fresh 0.6.0 baseline build
on an Apple M4 Pro (48 GiB), through Python 3.13.5, at 1/2/4/8 workers.
Each task reopens and decodes its TIFF before integration. The table below uses
eight workers, 2,000 bins, Poisson errors, solid-angle correction and polarization
0.965. Supersampling uses n=15 for Pilatus1M and n=5 for Eiger9M; pixel-split has
no subpixel count. Both builds use the same compiler, generic CPU target, LTO and
`fast_math=True`. Three fresh-process trials per point; the normal macOS file
cache is warm. These are file-processing rates, not cold-SSD bandwidth.

| Detector / method | File fps 0.6.0 → 0.6.1 | Speedup | Calibration s before → after | First 32-file batch s before → after | Peak RSS GiB before → after |
| --- | --- | --- | --- | --- | --- |
| Pilatus1M / pixel-split | 1331.4 → 1820.4 | 1.37× | 0.107 → 0.108 | 0.024 → 0.019 | 0.616 → 0.599 |
| Pilatus1M / supersampled | 1361.9 → 1885.9 | 1.38× | 0.373 → 0.371 | 0.024 → 0.018 | 0.409 → 0.356 |
| Eiger9M / pixel-split | 192.9 → 248.0 | 1.29× | 0.603 → 0.613 | 0.171 → 0.136 | 2.901 → 2.742 |
| Eiger9M / supersampled | 201.7 → 259.6 | 1.29× | 0.702 → 0.701 | 0.170 → 0.127 | 2.168 → 1.640 |

Persistent Eiger workers retain about 310 MiB of additional live decoding buffers at
eight workers. `clear_file_workers()` releases those buffers. Peak RSS and live
heap are reported separately; pixel-split setup still dominates its total peak.
All 6,144 primary frame results match the archived 0.6.0 references byte-for-byte.
See the [file-pipeline optimization benchmark](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-10-24-file-pipeline-optimization.md)
for all worker counts, the array control, setup time and validation.

Calibration setup is essentially unchanged; the matched float64 NumPy-array
control differs by less than 1% at eight workers. These gains apply to file-based
processing. Fast CSR arithmetic remains enabled by default; float64 weights,
strict/fast reduction order and Poisson formulas are unchanged. The standalone
Python regression suite checks 13,744 results across ten sample types, both byte
orders, uncompressed/Deflate/LZW TIFFs, strips/tiles, strict/fast arithmetic,
alternating images and streaming lifetimes.

The existing offline license format and activation remain compatible. Distribution
remains wheel-only for CPython 3.11–3.15 on Windows x86-64, macOS Intel/Apple
silicon and Linux x86-64/ARM64.

## [0.6.0] - 2026-10-24

### Added

- Configurable `fast_math` for the Python integrator and `with_fast_math` in Rust.
  Reassociate CSR accumulation into four partial float64 sums, using hardware
  fused multiply-add where available. Cover normalized and raw 1D/2D sums,
  Poisson errors and batches. Calibration geometry and CSR weights remain
  unchanged; histogram and azimuthal-spread calculations
  keep their existing algorithms. The option can change rounding. Mode changes reuse calibration caches and
  are rejected while an asynchronous batch holds the Python integrator.

### Changed

- Enable fast CSR accumulation by default for new Python and Rust integrators
  in 0.6.0. Both Python constructors default to `fast_math=True`;
  use `fast_math=False`, `set_fast_math(False)` or Rust's
  `with_fast_math(false)` to select strict accumulation and retain its previous
  rounding. Existing explicit mode selections and calibration formulas are
  unchanged.

- Reduce standard CSR integration memory traffic without changing each arithmetic
  mode's results: evaluate Poisson variance in the integration loop, cache
  per-bin normalization with mask/correction invalidation, and store internal
  pixel indices as checked u32 values with a usize fallback. Keep float64 weights,
  reduction order and the public Rust LUT API. Apply to normalized/raw 1D/2D
  integrations and Python batches; retain histogram and azimuthal-spread formulas.

- Build supersampled 1D and 2D CSR tables from per-pixel bin counts instead of
  collecting and sorting every subpixel event. Reuse a scalar geometry transform
  and per-row counting scratch, and allocate final CSR buffers at their exact
  size. Preserve the original float64 sampling arithmetic, pixel ordering,
  normalization, masks and Poisson propagation; regression tests compare CSR
  arrays bit-for-bit against the previous builders across units and ranges.
- Borrow contiguous float64 NumPy image buffers in synchronous Python `batch1d`
  calls instead of copying every image into an owned frame. Input borrow guards
  remain alive until all native workers finish, including optional Poisson and
  azimuthal-spread calculations. Asynchronous streaming retains owned buffers.
  Initialize common geometry/LUTs before parallel array work, and process mixed
  image shapes sequentially to avoid concurrent changes to the shared geometry.
- Rank grid-search auto-calibration candidates by Gaussian-weighted ring
  residuals instead of distinct-ring and inlier counts.
- Expand the Python grid-search defaults: 18 distance steps, 15 rotation
  steps over ±45°, 13 PONI steps over ±400 pixels, and 50 refinement
  candidates. The default calibration method remains `radial`; Rust
  `GridSearchOptions` defaults are unchanged.

<!-- measured-060 -->
### Measured Python integration performance

Published 0.5.5 versus the licensed 0.6.0 release build on an Apple M4 Pro
(48 GiB), with 8 workers, Poisson errors, 2,000 bins and warmed float64 images.
The fast-math argument is omitted to measure each version's default behavior.
Supersampling uses n=15 for Pilatus1M and n=5 for Eiger9M.

| Detector / method | 0.5.5 fps | 0.6.0 fps | Speed factor | Setup old → new (s) | Peak RSS old → new (GiB) |
| --- | --- | --- | --- | --- | --- |
| Pilatus1M / supersampled | 815.4 | 1746.2 | 2.14× | 2.223 → 0.370 | 14.263 → 0.549 |
| Pilatus1M / pixel-split | 799.6 | 1693.8 | 2.12× | 0.093 → 0.099 | 1.250 → 0.796 |
| Eiger9M / supersampled | 114.8 | 284.2 | 2.48× | 2.805 → 0.679 | 18.249 → 2.528 |
| Eiger9M / pixel-split | 110.8 | 264.3 | 2.38× | 0.548 → 0.561 | 8.050 → 3.598 |

The measurements include all cumulative integration changes since 0.5.5.
These are integration-only rates; disk decoding and acquisition are excluded.
Two qualified fresh-process trials per point cover 1/2/4/8 workers and both
error modes. Setup includes the first Poisson result; peak RSS includes input
pages and allocator retention. Fast mode can change rounding; explicit
`fast_math=False` retains strict accumulation. Explicit strict-mode results
were byte-identical to 0.5.5 on the tested patterns. The default-mode maximum
relative L2 differences were 7.1e-15 for intensity and 1.3e-15 for Poisson sigma.
All 96 release-wheel alternating-image checks passed exactly within each mode.

| Detector | Library / method | Poisson fps | Workers ready (s) | First result (s) | Peak RSS (GiB) | Live heap (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| Pilatus1M | Dioptrin 0.6.0 supersampled | 1746.2 | 0.370 | 0.370 | 0.549 | 105.5 |
| Pilatus1M | Dioptrin 0.6.0 pixel-split | 1693.8 | 0.099 | 0.099 | 0.796 | 108.2 |
| Pilatus1M | azint generic | 574.0 | 0.768 | 0.768 | 0.592 | 75.2 |
| Pilatus1M | azint CPU/LTO/fast-math | 891.6 | 0.757 | 0.757 | 0.592 | 75.2 |
| Pilatus1M | pyFAI full CSR | 635.8 | 1.319 | 0.570 | 1.440 | 1002.1 |
| Pilatus1M | pyFAI bbox CSR | 641.4 | 0.902 | 0.534 | 0.987 | 525.0 |
| Eiger9M | Dioptrin 0.6.0 supersampled | 284.2 | 0.679 | 0.679 | 2.528 | 697.3 |
| Eiger9M | Dioptrin 0.6.0 pixel-split | 264.3 | 0.561 | 0.561 | 3.598 | 722.4 |
| Eiger9M | azint generic | 102.6 | 1.860 | 1.860 | 3.255 | 431.3 |
| Eiger9M | azint CPU/LTO/fast-math | 146.3 | 1.924 | 1.924 | 3.253 | 431.3 |
| Eiger9M | pyFAI full CSR | 89.7 | 7.305 | 1.835 | 10.063 | 8138.6 |
| Eiger9M | pyFAI bbox CSR | 89.6 | 3.561 | 1.438 | 5.158 | 3265.6 |

The competitor rows use the same data and corrections. azint's generic and
CPU/LTO/fast-math builds are separate. pyFAI 2025.12.1 uses its full CSR and
default bbox CSR methods with a separate engine per Python worker; setup includes
all initialized engines. Algorithms and internal precision differ, so these
rates are not claims of identical numerical tasks at the kernel level.
See the [release benchmark](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-10-24-0.6.0.md) and
[azint/pyFAI comparison](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-10-24-0.6.0-comparison.md) for
settings, all worker counts, memory details and numerical differences.

<!-- end-measured-060 -->

<!-- latest-pyfai-release-060 -->
### Latest stable pyFAI follow-up

The comparison was extended to pyFAI 2026.9.0. Both 2025.12.1 and 2026.9.0 were rerun with identical dependencies and 1/2/4/8 Python workers; the original results are retained above.

| Detector | Library / method | Poisson fps | Workers ready (s) | First result (s) | Peak RSS (GiB) | Live heap (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| Pilatus1M | Dioptrin 0.6.0 supersampled | 1746.2 | 0.370 | 0.370 | 0.549 | 105.5 |
| Pilatus1M | Dioptrin 0.6.0 pixel-split | 1693.8 | 0.099 | 0.099 | 0.796 | 108.2 |
| Pilatus1M | azint generic | 574.0 | 0.768 | 0.768 | 0.592 | 75.2 |
| Pilatus1M | azint CPU/LTO/fast-math | 891.6 | 0.757 | 0.757 | 0.592 | 75.2 |
| Pilatus1M | pyFAI 2026.9.0 full CSR | 647.2 | 1.309 | 0.572 | 1.440 | 1002.2 |
| Pilatus1M | pyFAI 2026.9.0 bbox CSR | 644.1 | 0.905 | 0.530 | 0.990 | 525.1 |
| Eiger9M | Dioptrin 0.6.0 supersampled | 284.2 | 0.679 | 0.679 | 2.528 | 697.3 |
| Eiger9M | Dioptrin 0.6.0 pixel-split | 264.3 | 0.561 | 0.561 | 3.598 | 722.4 |
| Eiger9M | azint generic | 102.6 | 1.860 | 1.860 | 3.255 | 431.3 |
| Eiger9M | azint CPU/LTO/fast-math | 146.3 | 1.924 | 1.924 | 3.253 | 431.3 |
| Eiger9M | pyFAI 2026.9.0 full CSR | 90.6 | 7.163 | 1.730 | 10.054 | 8138.7 |
| Eiger9M | pyFAI 2026.9.0 bbox CSR | 91.7 | 3.638 | 1.475 | 5.158 | 3265.7 |

See the [latest pyFAI version control](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-10-24-pyfai-2026.9.0.md) for all worker counts, setup/memory details and numerical checks. Dioptrin and azint measurements come from the earlier qualified sweep.

At eight Python workers with Poisson errors, the new/old throughput factors were 1.008–1.019 (about 1–2% faster). Setup and memory were similar. Other worker counts showed small changes in both directions. Two trials per point do not establish a general performance gain; this test shows no large speed jump between these versions on this Mac/CPU backend.

Axes, intensities and Poisson sigmas were byte-identical between the two pyFAI versions on the tested patterns. For both pyFAI versions, setup starts before pyFAI.load: top-level imports and input warm-up are excluded, but lazy imports reached during construction remain included. Constructor and post-constructor components are recorded separately in the latest-version CSV.
<!-- end-latest-pyfai-release-060 -->

<!-- file-pipeline-release-060 -->
### File-backed Python benchmark

The end-to-end TIFF comparison reopens and decodes every image inside its worker before integration. The published 0.6.0 native file-batch API is compared with azint and pyFAI 2026.9.0, using 1/2/4/8 workers and Poisson errors.

| Detector | Library / method | File fps | Array-control fps | File/array rate | Logical TIFF MiB/s |
| --- | --- | --- | --- | --- | --- |
| Pilatus1M | Dioptrin 0.6.0 pixel-split | 1307.5 | 1636.5 | 79.9% | 5108 |
| Pilatus1M | Dioptrin 0.6.0 supersampled | 1343.1 | 1722.2 | 78.0% | 5248 |
| Pilatus1M | azint generic | 488.0 | 614.1 | 79.5% | 1907 |
| Pilatus1M | azint CPU/LTO/fast-math | 690.3 | 972.1 | 71.0% | 2697 |
| Pilatus1M | pyFAI 2026.9.0 full CSR | 495.0 | 644.2 | 76.8% | 1934 |
| Pilatus1M | pyFAI 2026.9.0 bbox CSR | 504.6 | 630.2 | 80.1% | 1971 |
| Eiger9M | Dioptrin 0.6.0 pixel-split | 183.8 | 262.5 | 70.0% | 7110 |
| Eiger9M | Dioptrin 0.6.0 supersampled | 197.3 | 281.5 | 70.1% | 7631 |
| Eiger9M | azint generic | 99.1 | 124.7 | 79.5% | 3833 |
| Eiger9M | azint CPU/LTO/fast-math | 127.2 | 174.3 | 73.0% | 4921 |
| Eiger9M | pyFAI 2026.9.0 full CSR | 77.9 | 89.7 | 86.9% | 3013 |
| Eiger9M | pyFAI 2026.9.0 bbox CSR | 76.1 | 92.3 | 82.4% | 2944 |

The normal macOS file cache is warm; these are logical file-processing rates, not cold-SSD bandwidth. Each batch has 32 tasks. The matched array control uses the same calibrated pixels, with backend-appropriate dtypes. All 6,144 file/array frame results were byte-identical to their own backend references. Algorithms and internal precision still differ across libraries. Memory and setup, per-worker stage diagnostics and proposed future improvements are in the [file-pipeline benchmark](https://github.com/CPrescher/Dioptrin/blob/main/docs/benchmarks/2026-10-24-file-pipeline.md). The integration engine is unchanged; these measurements describe an additional workflow, not another code-speed upgrade.
<!-- end-file-pipeline-release-060 -->

## [0.5.5] - 2026-09-29

### Changed

- Retain signed license evidence inside native Python integrators and revalidate
  it before 1D, 2D, batch, streaming-batch, and multi-detector operations.
  Construction and operation admission use independent Ed25519 verifiers.
  Offline activation, existing license format, and numerical kernels are unchanged.
- Check expiry for new operations. Existing objects retain their original license;
  construct a new integrator after renewal. Already admitted batches may finish
  after expiry. Verification adds per-call overhead, amortized across batch frames.

### Added

- Cross-platform tests with ephemeral licenses and deliberately bypassed native
  checks, plus numerical-reference and isolated release-wheel checks.

### Security limitations

- This is additional defense in depth, not tamper-proof license enforcement.
  A separate blind assessment still bypassed the hardened macOS ARM64 candidate
  with two native patches and performed genuine numerical integration. See
  `docs/offline-licensing.md` for the scope, evidence, and operational behavior.

## [0.5.3] - 2026-09-21

### Added

- Optional `azimuthal_std=True` for `Integrator.integrate1d`, `batch1d`, and
  `batch1d_iter`, returning weighted population spread separately from Poisson
  errors. Includes Rust frame/slice and batch APIs, mask/correction handling,
  stable accumulation, and NaN spread for empty bins. Existing defaults are
  unchanged; the statistic and its difference from pyFAI variance propagation
  are documented in the README.

- **Experimental `AutoMethod::Auto` (`method="auto"`).** Runs `Radial`,
  `Ransac`, and `Grid` concurrently and selects the lowest refined RMS.
  Available explicitly for further development; `method="radial"` remains
  the default in both Rust and Python while automatic selection is improved.

- **Grid-search auto-calibration bootstrap (`AutoMethod::Grid`,
  `method="grid"`).** Bypasses ellipse fitting entirely: samples a coarse
  `(distance, rot1, rot2, poni1, poni2)` grid, scores each cell by how
  many peaks land within a 2θ tolerance of any calibrant ring, ranked
  primarily by *number of distinct rings populated* (so wrong-anchor
  cells that coincidentally pile peaks onto a few sub-multiple-2θ rings
  lose to the true cell that distributes peaks across many rings), then
  L-M-refines the top-K candidates and returns the one with the lowest
  RMS. The refine-and-pick step disambiguates between ring-index
  aliases that single-cell scoring cannot.

  End-to-end recovery on the heavy-tilt `ceo2_side.tif` fixture
  (Pilatus300kw strip detector at 31.6° tilt, 630 raw peaks, no clicks,
  no geometry hint): distance recovered to 0.07 %, rotations within
  5 mrad of pyFAI truth, RMS 0.097 mrad of 2θ (picking-noise floor) —
  the same answer L-M reaches when started from the truth PONI.
  `test_auto_calibrate_ceo2_side.py` is now a strict-gate test rather
  than a diagnostic. The grid method is opt-in (`method="grid"`); the
  default remains `radial` for near-perpendicular detectors where the
  cheaper ellipse-fit bootstrap already works.

- **Tilt-aware geometry inversion (`src/calibration/inversion.rs`).** Given
  a fitted Debye-Scherrer ellipse on the detector plane plus the
  calibrant's known `2θ_k`, closed-form algebra recovers `(D, PONI,
  rot1, rot2)` exactly for a flat tilted detector — no L-M required.
  Math: `axis_ratio = 1/√(cos²τ - sin²τ·tan²(2θ))` solves for the tilt
  magnitude τ, `semi_minor·px / (a·tan(2θ))` gives D, and the
  PONI-ring-center offset along the ellipse major axis closes the
  system. Aggregating across multiple rings (median per parameter)
  is robust to one bad ellipse fit. Verified on synthetic perpendicular
  and 31.6°-tilt cases to 10⁻⁹ precision (4 unit tests). Used as the
  bootstrap step in `auto_calibrate` for visibly tilted geometries (see
  next entry).
- **Hybrid bootstrap in `auto_calibrate`.** Replaces the rotation-zero
  bootstrap with a regime switch:
    - median ellipse axis_ratio < 1.05 → perpendicular bootstrap
      (PONI = ring-center, rot = 0, D = mean(b·px / tan(2θ))). More
      robust than tilt inversion when the geometry is near-perpendicular,
      because the `(1-C)/(1+tan²(2θ))` term that sets τ amplifies tiny
      ellipse-fit noise into spurious tilts.
    - median axis_ratio ≥ 1.05 → tilt-aware inversion (above).
  This restores the near-perpendicular L-M convergence on real LaB6 (poni
  recovered to ~0.1 %) while giving a physically meaningful starting
  point for tilted cases where ellipse fitting still produces usable
  ellipses. For strip detectors (e.g. `ceo2_side.tif`) where short arcs
  defeat ellipse fitting entirely, use `AutoMethod::Grid` (see above).
- **Radial-binning ring extraction (`dioptrin.extract_rings_radial`).**
  Given an approximate **ring center**, compute peak radii, build a
  smoothed radial histogram, pick local maxima as ring radii, and fit each
  ring's peaks with a single least-squares ellipse. Avoids the
  5-of-5-same-ring Monte Carlo coincidence that crippled RANSAC on dense
  multi-ring patterns — every peak from each ring goes directly into that
  ring's clean fit. Only valid when rings are approximately circular
  (low detector tilt); for heavily tilted detectors a tilt-aware
  ellipse-center estimator is needed instead.
- **Ring-center estimation (`dioptrin.estimate_ring_center`).** 3-point
  circle RANSAC + radial-band consistency scoring: sample triples of
  peaks, fit the circle they define, score the candidate center by how
  concentrated peaks are in radial bins from it. The function name is
  deliberately **not** "beam_center" or "PONI" — those three geometric
  points coincide only for a perpendicular detector and are distinct
  in general. This module's docstring covers the distinction in detail;
  the short version: for near-perpendicular detectors, ring-center ≈ PONI
  ≈ beam center to within a few pixels, and the estimator is a useful
  starting guess for the bootstrap PONI. For heavily tilted detectors
  the three differ by `~D · tan(tilt)` and a separate ellipse-based
  inversion is required.
  Works robustly on full-coverage multi-ring patterns, partial arcs
  (centroid estimators fail here), and off-detector beams — but only
  when the near-circular assumption holds. 5 Rust unit tests cover
  each case.
- **`AutoMethod::Radial`** is now the default for `auto_calibrate`,
  replacing the old 5-point RANSAC-ellipse approach. End-to-end real-data
  recovery on LaB6_40keV_MarCCD with no clicks and no geometry hint:
  refined distance 0.196656 m (truth 0.196687 m, Δ = 31 µm =
  0.016 %), rot1 4.27 mrad (truth 4.3 mrad, essentially exact),
  RMS residual 0.4 mrad in 8 L-M iterations. The old method is still
  selectable via `method="ransac"`.
  Known limitations:
  - The bootstrap initialises PONI to the estimated ring center. This
    is correct only for near-perpendicular detectors (where ring center
    ≈ PONI ≈ beam center). For tilted detectors PONI is at the foot
    of the perpendicular from sample to detector and is displaced from
    the ring center by `~D · tan(tilt)`; the radial method has no way
    to recover that displacement on its own.
  - The radial-binning assumption (rings are concentric circles around
    one center) breaks for tilted detectors because each ring becomes
    an ellipse with its own geometric center.
  Both issues are addressed by a future tilt-aware "geometry inversion"
  bootstrap that algebraically solves for `(D, PONI, rot1, rot2)` from
  one or more fitted ellipses + the calibrant's known `2θ_k`. The
  shipped `ceo2_side.tif` test fixture (Pilatus300kw strip detector at
  31° tilt) is a known failure case for the current bootstrap.
- **End-to-end auto calibration (`dioptrin.auto_calibrate`).** Wires the
  RANSAC ellipse clustering, ring-identity matching, and L-M refinement
  into a single call: given a list of picked peaks plus a calibrant,
  wavelength, and pixel size, returns a refined PONI with no manual
  seed clicks and no geometry hint. Internal pipeline: RANSAC →
  shared-center filter (drops spurious cross-ring ghost ellipses) →
  identify rings by tan(2θ) ratio matching → bootstrap PONI from
  matched-ellipse centers + s_i / tan(2θ_k) → L-M refine. A Rust unit
  test (`auto_calibrate_recovers_synthetic_geometry`) recovers a known
  synthetic LaB6 geometry to numerical precision.
  Known limitation: on dense calibrants like LaB6 (91 d-spacings),
  when the innermost ring is weakly detected, the ratio-only anchor
  disambiguation can lock onto a self-consistent chain at the wrong
  scaled distance. A follow-up will add an optional distance hint to
  resolve the ambiguity. The slice-2 and slice-3a/3b pipelines are
  unaffected.
- **Image-wide peak extraction (`dioptrin.find_peaks_image`).** A no-seed
  variant of Massif: subtract a Gaussian background, threshold by
  `mean + k·std` of the residual, return every 3×3 local maximum with
  sub-pixel parabolic refinement. Intended as the peak source for the
  auto-calibration pipeline.
- **RANSAC ellipse deduplication.** `ransac_ellipses` now merges fits
  with similar centers and similar semi-major axes (configurable
  thresholds), unioning their inlier sets and refitting. Defaults
  `5 px` center distance and `2 %` size, which removes the manual
  Python-side dedup that the slice-3b test used to do.
- **Ring identity matching (`dioptrin.identify_rings`).** Given a list of
  detected ellipses (sorted ascending by semi-major axis) and a known
  calibrant, identify which calibrant ring each ellipse corresponds to —
  with no input geometry hint (no distance, no beam-center estimate).
  Algorithm: the ratio `r_k / r_0 = tan(2θ_k) / tan(2θ_0)` is independent
  of detector distance, so a hypothesis-test loop over every
  `(anchor_ellipse, anchor_ring)` pair finds the assignment whose
  implied-distance scatter is tightest. Handles missing rings, spurious
  ellipses, smallest-ellipse-not-being-ring-0, and dense calibrants
  (LaB6 has 91 d-spacings) where the ratio chain alone is ambiguous —
  the distance-consistency check breaks the tie. Validated by 5 Rust
  unit tests on synthetic LaB6 (full sequence, missing rings, anchor
  offset, spurious extra ellipse, random sizes) and a real-data smoke
  test on LaB6_40keV_MarCCD.tif that recovers 10/11 ring identities
  matching Dioptas's iterative-fit labels (the one mismatch is a
  Dioptrin-rejected spurious ellipse).
- **RANSAC ellipse clustering (`dioptrin.ransac_ellipses`).** Given a list of
  2-D peak coordinates, fit a family of ellipses with outlier rejection.
  Implementation: direct linear conic fit (F=-1 normalization) with the
  ellipse discriminant check `4AC-B² > 0`, Sampson distance for inlier
  testing, greedy RANSAC that detects one ellipse, removes its inliers, and
  repeats. Optional `max_axis_ratio` filter rejects degenerate elongated
  fits when the user knows the detector is near-perpendicular (e.g.
  setting `1.2` for typical lab/synchrotron geometries). Returns sorted
  ellipses (semi-major ascending → ring order under a powder pattern) plus
  per-input-point assignment (`None` = outlier). Public Rust API in
  `dioptrin::calibration::{ellipse, ransac_ellipses}`. Synthetic Rust
  tests verify single-ellipse recovery to 10⁻⁶, multi-ellipse clustering
  with planted outliers, and pure-noise rejection. Real-data smoke test
  on Dioptas's LaB6 peaks confirms at least the brightest rings are
  cleanly captured at the true beam center.
- **Massif peak picking (`dioptrin.find_peaks_massif`).** A port of pyFAI's
  `Massif.find_peaks` algorithm: separable Gaussian background filter,
  binarisation against the local background, 4-connected component
  labeling, uphill walk to integer-pixel local max, and sub-pixel
  refinement via parabolic fit on the 3×3 neighbourhood. The Gaussian σ
  defaults to pyFAI's adaptive `max(5, max(dim1, dim2) / 50)`. Validated
  against Dioptas's pyFAI-backed Massif on the real LaB6_40keV_MarCCD
  frame: every Dioptas-picked control point has a Dioptrin-picked
  counterpart within sub-pixel distance (100% within 1 px on seed 1,
  96% within 1 px on seed 2). Combined with the refine-parity test this
  validates Dioptrin as a self-contained pyFAI replacement for
  calibration.
- **Geometry calibration (`dioptrin.calibration`).** A Levenberg–Marquardt
  refinement that takes picked control points and a known calibrant and
  refines the detector PONI. Highlights:
  - Hand-rolled L-M with finite-difference Jacobian (no `nalgebra` /
    `argmin` dependency).
  - Optional Huber IRLS loss for robustness against spotty / single-crystal
    contamination on rings.
  - Auto ring-assignment by predicted-2θ-nearest-d-spacing each iteration.
  - Free / fixed parameter mask (`distance`, `poni1`, `poni2`, `rot1`,
    `rot2`, `rot3`, `wavelength`).
  - Off-detector primary beam works correctly (PONI may lie outside the
    detector window).
  - Rotations wrap to `(-π, π]` after each accepted step to prevent drift
    along under-constrained periodic directions.
  - pyFAI/Dioptas pixel-center convention (`(d + 0.5) * pixel`) so picked
    points from those tools feed in directly.
- pyFAI-compatible `.D` calibrant parser; `LaB6_SRM660b.D` and `CeO2.D`
  shipped in `test_data/calibrants/`.
- Python API: `dioptrin.Calibrant`, `dioptrin.CalibrationPoints`,
  `dioptrin.CalibrationResult`, `dioptrin.refine(...)`.
- `LaB6_40keV_MarCCD.tif` + `.poni` test fixture from Dioptas's test corpus.
- `python_examples/test_calibration_lab6.py` — pyFAI-forward-model round-trip
  integration test (recovers a known LaB6 PONI to numerical precision).
- `python_examples/test_calibration_dioptas_parity.py` — cross-validates
  against Dioptas + pyFAI on the same 271 real Massif-picked points across
  10 LaB6 rings; refined PONIs agree at the picking-noise floor (distance
  to 0.4 ppm, PONI to 18 µm, rotations to 93 µrad).

## [0.5.2] - 2026-05-12

### Fixed

- Avoid an 80 MB `normalization` clone on every `integrate1d` / `integrate2d`
  call. `GeometryState.normalization` is now wrapped in `Arc<Vec<f64>>`, so
  the per-call cost is an atomic refcount bump. On a 12-worker batch1d run
  over Eiger 9M frames at 4400 bins, throughput goes from ~163 to ~303
  files/s, and serial→12-worker scaling improves from ~3× to ~4.8×. The
  single-worker numpy-array path more than doubles (53 → 127 files/s).

### Changed

- Batch worker loops (`process_files`, `process_files_streaming`,
  `process_queue_streaming`) reuse a single `Frame` per worker and decode
  TIFFs into its existing `data` buffer via the new
  `Frame::reload_from_tiff` instead of allocating a fresh `Frame` per file.
  Peak RSS during a 12-worker batch1d run drops from ~4.7 GB to ~3.85 GB.
  Throughput is unchanged.

## [0.5.1] - 2026-03-22

### Fixed

- `python_examples/batch_benchmark_comparison.py` no longer hard-requires
  `psutil`; memory tracking is skipped with a notice when the package is
  not installed.

## [0.5.0] - 2026-03-21

### Added

- `MultiIntegrator` for merging results from multiple detectors (similar in
  spirit to pyFAI's `MultiGeometry`). Sums raw signal / normalization /
  variance across detectors on a shared radial bin grid and divides once
  at the end. Available from Python as `dioptrin.MultiIntegrator`.
- `python_examples/batch_benchmark_comparison.py` for side-by-side
  throughput and peak-memory comparisons against pyFAI.

## [0.4.4] - 2026-03-03

### Added

- `dioptrin.validate_license()` Python function for offline license-key
  validation outside of `Integrator` construction.

### Changed

- License format no longer carries a tier field; license keys signed in
  the new format are required.

### Fixed

- License exception classes are always registered on the Python module,
  even when the binary is built without the `licensed` feature, so user
  code can `except dioptrin.LicenseExpiredError:` unconditionally.

## [0.4.3] - 2026-03-03

### Added

- `LicenseNotFoundError` and `LicenseExpiredError` Python exception
  subclasses for finer-grained handling of license errors.

## [0.4.2] - 2026-03-02

### Fixed

- maturin wheel build for `linux/x86_64` (`-i` flag passed correctly).

## [0.4.1] - 2026-03-02

Maintenance release: cargo fmt over license / keygen modules. No
user-visible changes.

## [0.4.0] - 2026-03-02

### Added

- Offline license verification using ed25519 signatures for Python
  wheels. License keys are checked at `Integrator()` construction; the
  public key is embedded in the binary.
- `benchmark` optional dependency group in `pyproject.toml`.

### Changed

- **Breaking:** `normalization` is now a required parameter in the LUT
  integration functions (`lut::integrate1d`, `lut::integrate2d`, etc.).
- LUT integration now follows pyFAI's weighted-average convention:
  `I = Σ(w · signal) / Σ(w · norm)`.
- Minimum Python version bumped to 3.11.

### Fixed

- `batch1d_iter` / `batch2d_iter` now preserve the input order of frames.
- README examples updated to match the current API.

### Security

- numpy upgraded to 2.4.2 (2.4.0 was yanked upstream).
- pillow upgraded to 12.1.1 (CVE-2026-25990).

## [0.3.0] - 2026-02-28

Earliest tagged release tracked in this changelog. See `git log v0.3.0`
for the contents at this point.

[Unreleased]: https://github.com/CPrescher/Dioptrin/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/CPrescher/Dioptrin/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/CPrescher/Dioptrin/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/CPrescher/Dioptrin/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/CPrescher/Dioptrin/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/CPrescher/Dioptrin/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/CPrescher/Dioptrin/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/CPrescher/Dioptrin/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/CPrescher/Dioptrin/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/CPrescher/Dioptrin/compare/v0.5.5...v0.6.0
[0.5.5]: https://github.com/CPrescher/Dioptrin/compare/v0.5.4...v0.5.5
[0.5.3]: https://github.com/CPrescher/Dioptrin/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/CPrescher/Dioptrin/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/CPrescher/Dioptrin/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/CPrescher/Dioptrin/compare/v0.4.4...v0.5.0
[0.4.4]: https://github.com/CPrescher/Dioptrin/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/CPrescher/Dioptrin/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/CPrescher/Dioptrin/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/CPrescher/Dioptrin/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/CPrescher/Dioptrin/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/CPrescher/Dioptrin/releases/tag/v0.3.0
