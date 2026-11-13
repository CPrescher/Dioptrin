use super::{IntegrationMethod, Integrator, RadialUnit};
use crate::frame::Frame;
use crate::lut::IntermediateResult2d;
use crate::math::{self, histogram};
use std::f64::consts::TAU;

/// Splitting convention for the shared-grid 2D path. Other methods and legacy
/// single-detector/1D integration are unaffected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PixelSplitMode {
    /// Exact polygon intersections divided by original pixel polygon area.
    #[default]
    Fractional,
    /// Approximate line-area rasterization and clipped-pixel renormalization,
    /// for comparison with pyFAI's full histogram engine. Singular beam pixels
    /// use sampling; numerical negative weights are discarded and renormalized.
    PyfaiHistogram,
}

/// Explicit shared 2D bin edges. Radial values use the integrator's unit;
/// azimuthal values are radians. Bins are half-open, with no edge clamping.
/// Angles are periodic over 2π; sectors exclude angles outside their edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegrationGrid2d {
    pub(crate) pixel_split_mode: PixelSplitMode,
    pub(crate) n_radial: usize,
    pub(crate) n_azimuthal: usize,
    radial: [u64; 2],
    azimuthal: [u64; 2],
    /// Count a radial value exactly at the upper edge in the last bin and
    /// clamp rounding overflow there, like the default 2D counting table.
    clamp_radial_end: bool,
    /// Put a value that lies inside the edges but whose bin index rounds to
    /// `n` into the last bin, on both axes, like the default 2D tables.
    clamp_index_rounding: bool,
}

impl IntegrationGrid2d {
    pub fn new(
        n_radial: usize,
        n_azimuthal: usize,
        radial_range: (f64, f64),
        azimuthal_range: (f64, f64),
    ) -> Result<Self, String> {
        if n_radial == 0
            || n_azimuthal == 0
            || n_radial.checked_mul(n_azimuthal).is_none()
            || n_radial.saturating_mul(n_azimuthal) > isize::MAX as usize / 8
        {
            return Err(
                "2D bin counts must be positive and their product must fit in memory indexing"
                    .into(),
            );
        }
        for (name, (lo, hi), count) in [
            ("radial", radial_range, n_radial),
            ("azimuthal", azimuthal_range, n_azimuthal),
        ] {
            let step = (hi - lo) / count as f64;
            if !lo.is_finite()
                || !hi.is_finite()
                || !step.is_finite()
                || step <= 0.0
                || lo + step == lo
                || hi - step == hi
            {
                return Err(format!(
                    "{name} range must have finite, increasing, resolvable bin edges"
                ));
            }
        }
        if radial_range.0 < 0.0 {
            return Err("radial range must be nonnegative".into());
        }
        if azimuthal_range.1 - azimuthal_range.0 > TAU + 1e-12 {
            return Err("azimuthal span must be at most 2π (one full turn)".into());
        }
        // Correct distant starts before computing weights or axis labels. Keep
        // all previously accepted ranges, including pyFAI's documented seams.
        let azimuthal_range = if !(-TAU..=TAU).contains(&azimuthal_range.0) {
            let mut start = azimuthal_range.0.rem_euclid(TAU);
            if start >= std::f64::consts::PI {
                start -= TAU;
            }
            (start, start + (azimuthal_range.1 - azimuthal_range.0))
        } else {
            azimuthal_range
        };
        Ok(Self {
            pixel_split_mode: PixelSplitMode::Fractional,
            n_radial,
            n_azimuthal,
            radial: [radial_range.0.to_bits(), radial_range.1.to_bits()],
            azimuthal: [azimuthal_range.0.to_bits(), azimuthal_range.1.to_bits()],
            clamp_radial_end: false,
            clamp_index_rounding: false,
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n_azimuthal, self.n_radial)
    }
    /// Select an alternative pixel-splitting convention without changing edges.
    pub fn with_pixel_split_mode(mut self, mode: PixelSplitMode) -> Self {
        self.pixel_split_mode = mode;
        self
    }
    /// Include radial values equal to the upper edge in the last bin (the
    /// default 2D counting convention, whose upper edge is the data maximum).
    pub(crate) fn with_clamped_radial_end(mut self) -> Self {
        self.clamp_radial_end = true;
        self
    }
    /// Keep values inside the edges whose bin index rounds to `n` (or, in a
    /// full circle, whose wrapped offset rounds to 2π) in the last bin instead
    /// of dropping them.
    pub(crate) fn with_clamped_index_rounding(mut self) -> Self {
        self.clamp_index_rounding = true;
        self
    }
    pub fn radial_range(&self) -> (f64, f64) {
        (
            f64::from_bits(self.radial[0]),
            f64::from_bits(self.radial[1]),
        )
    }
    pub fn azimuthal_range(&self) -> (f64, f64) {
        (
            f64::from_bits(self.azimuthal[0]),
            f64::from_bits(self.azimuthal[1]),
        )
    }
    pub fn radial_centers(&self) -> Vec<f64> {
        centers(self.radial_range(), self.n_radial)
    }
    pub fn azimuthal_centers(&self) -> Vec<f64> {
        centers(self.azimuthal_range(), self.n_azimuthal)
    }
    pub(crate) fn radial_step(&self) -> f64 {
        let (lo, hi) = self.radial_range();
        (hi - lo) / self.n_radial as f64
    }
    pub(crate) fn azimuthal_step(&self) -> f64 {
        let (lo, hi) = self.azimuthal_range();
        (hi - lo) / self.n_azimuthal as f64
    }
    pub(crate) fn bin(&self, radial: f64, azimuthal: f64) -> Option<usize> {
        let (r0, r1) = self.radial_range();
        let (a0, a1) = self.azimuthal_range();
        if !radial.is_finite()
            || !azimuthal.is_finite()
            || radial < r0
            || radial > r1
            || (radial == r1 && !self.clamp_radial_end)
        {
            return None;
        }
        let offset = if self.clamp_index_rounding {
            // Wrap only angles outside the edges, so a value just below `a1`
            // is not rounded onto it. In a full circle a wrapped value can
            // round up to `a1`; it lies just below `a0` and belongs to the
            // last bin.
            let wrapped = if azimuthal >= a0 && azimuthal < a1 {
                azimuthal
            } else {
                a0 + (azimuthal - a0).rem_euclid(TAU)
            };
            if wrapped >= a1 && a1 - a0 < TAU {
                return None;
            }
            wrapped - a0
        } else {
            let azimuthal = a0 + (azimuthal - a0).rem_euclid(TAU);
            if azimuthal >= a1 {
                return None;
            }
            azimuthal - a0
        };
        let mut r = ((radial - r0) / self.radial_step()) as usize;
        let mut a = (offset / self.azimuthal_step()) as usize;
        if self.clamp_radial_end || self.clamp_index_rounding {
            r = r.min(self.n_radial - 1);
        }
        if self.clamp_index_rounding {
            a = a.min(self.n_azimuthal - 1);
        }
        (r < self.n_radial && a < self.n_azimuthal).then_some(a * self.n_radial + r)
    }
}

fn centers((lo, hi): (f64, f64), count: usize) -> Vec<f64> {
    let step = (hi - lo) / count as f64;
    (0..count).map(|i| lo + (i as f64 + 0.5) * step).collect()
}

/// Bin count and outer edges of contiguous azimuthal rows described by their
/// centres: the first row is centred on `center_range.0`, every row covers its
/// centre ± `half_width`, and the centres advance by `2 * half_width` until
/// just before `center_range.1`. All angles are radians. The end is exclusive:
/// it is where the next centre would be, not a centre itself, so the rows end
/// one half-width before it.
///
/// Returns `(n_azimuthal, (min, max))` for
/// [`Integrator::integrate2d_on_azimuthal_range`]. For example `(0, 2π)` with a
/// 2.5° bracket gives 72 rows centred on 0°, 5°, …, 355° with edges
/// `(-2.5°, 357.5°)`. The span must be at most 2π and a whole number of rows,
/// otherwise an error is returned rather than adjusting the bracket. Both
/// checks allow 1e-6 relative, so single-precision angles are accepted; the
/// row width actually used is the span divided by the row count.
///
/// A first centre in [-π, 2π] is kept as given, which covers both the
/// -180°…180° and the 0°…360° convention. Any other first centre is corrected
/// to the equivalent angle in [-π, π), independent of the bracket size.
pub fn azimuthal_bracket_bins(
    center_range: (f64, f64),
    half_width: f64,
) -> Result<(usize, (f64, f64)), String> {
    let (first, end) = center_range;
    let span = end - first;
    if !first.is_finite() || !end.is_finite() || !span.is_finite() || span <= 0.0 {
        return Err("azimuthal centre range must be finite and increasing".into());
    }
    if !half_width.is_finite() || half_width <= 0.0 {
        return Err("azimuthal bracket must be finite and positive".into());
    }
    if span > TAU * (1.0 + 1e-6) {
        return Err("azimuthal centres must span at most 2π (one full turn)".into());
    }
    // A full turn written in single precision, or far from zero, is a full turn.
    let span = span.min(TAU);
    let rows = span / (2.0 * half_width);
    let n = rows.round();
    // Relative for rounding in the bracket, capped so that very fine brackets
    // cannot be off by a visible fraction of a row.
    if !n.is_finite() || n < 1.0 || (rows - n).abs() > (1e-6 * n).min(1e-3) {
        return Err(format!(
            "azimuthal bracket of ±{half_width} rad does not divide the span of {span} rad \
             into a whole number of rows ({rows} rows)"
        ));
    }
    // Use the exact row width implied by the count, so the edges stay consistent.
    let half = span / (2.0 * n);
    // The range path corrects first *edges* outside ±2π, which would depend on
    // the bracket. Correct the centre here instead: with `half <= π`, a centre
    // in [-π, 2π] always has its edge inside ±2π, so the grid leaves it alone.
    let first = if (-std::f64::consts::PI..=TAU).contains(&first) {
        first
    } else {
        let wrapped = first.rem_euclid(TAU);
        if wrapped >= std::f64::consts::PI {
            wrapped - TAU
        } else {
            wrapped
        }
    };
    Ok((n as usize, (first - half, first - half + span)))
}

/// 2D result on explicit `(min, max)` azimuthal edges, from
/// [`Integrator::integrate2d_on_azimuthal_range`].
#[derive(Clone, Debug, PartialEq)]
pub struct AzimuthalRange2dResult {
    /// Row-major intensity of shape `(n_azimuthal, n_radial)`.
    pub intensity: Vec<f64>,
    /// Per-bin standard errors, when requested.
    pub errors: Option<Vec<f64>>,
    /// Radial bin centres in the integrator's unit.
    pub radial: Vec<f64>,
    /// Azimuthal bin centres in radians: `min + (i + 0.5) * (max - min) / n_azimuthal`.
    pub azimuthal: Vec<f64>,
}

/// Inputs that determine [`Integrator`]'s default 2D radial edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RadialEdgesKey {
    dim1: usize,
    dim2: usize,
    n_radial: usize,
    unit: RadialUnit,
    method: usize,
    radial_range: Option<[u64; 2]>,
}

impl Integrator {
    /// Integrate onto explicit shared edges and return raw sums. Splitting uses
    /// the grid's mode (default: fractional polygon areas, with sampling for
    /// singular beam pixels; optional: pyFAI histogram approximation). Variance,
    /// when requested, is Poisson max(signal, 0), accumulated with squared weights.
    /// Relative solid angle and this detector's polarization are in sum_norm.
    pub fn integrate2d_raw_on_grid_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        grid: &IntegrationGrid2d,
        errors: bool,
    ) -> Result<IntermediateResult2d, String> {
        self.integrate2d_raw_on_grid_impl(data, None, dim1, dim2, grid, errors)
    }

    /// Integrate independent, externally propagated pixel variances using squared
    /// pixel-splitting weights. Calibration/geometry uncertainties are not added.
    /// Variances must be finite, nonnegative and match the image, including masked pixels.
    pub fn integrate2d_raw_on_grid_with_variances_from_slice(
        &self,
        data: &[f64],
        variance: &[f64],
        dim1: usize,
        dim2: usize,
        grid: &IntegrationGrid2d,
    ) -> Result<IntermediateResult2d, String> {
        self.integrate2d_raw_on_grid_impl(data, Some(variance), dim1, dim2, grid, true)
    }

    fn integrate2d_raw_on_grid_impl(
        &self,
        data: &[f64],
        variance: Option<&[f64]>,
        dim1: usize,
        dim2: usize,
        grid: &IntegrationGrid2d,
        errors: bool,
    ) -> Result<IntermediateResult2d, String> {
        let grid = if matches!(self.method, IntegrationMethod::PyfaiFull) {
            grid.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram)
        } else {
            *grid
        };
        self.validate_2d_input(data, dim1, dim2)?;
        if let Some(v) = variance {
            validate_variances(data, v)?;
        }
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        Self::validate_grid_coordinates(&view.state, &normalization)?;
        let mask = self.mask.as_ref().map(|m| m.data.as_slice());
        let lut = view.ensure_lut2d(self.shared_grid_key(grid));
        Ok(if let Some(variance) = variance {
            lut.raw2d_with_variances(data, variance, mask, &normalization)
        } else if errors {
            self.lut_integrate2d_raw_with_variance(data, &lut, mask, &normalization)
        } else {
            self.lut_integrate2d_raw(data, &lut, mask, &normalization)
        })
    }

    pub(super) fn validate_grid_coordinates(
        state: &super::GeometryState,
        normalization: &[f64],
    ) -> Result<(), String> {
        {
            if state
                .radial
                .iter()
                .chain(&state.azimuth)
                .any(|v| !v.is_finite())
                || normalization.iter().any(|v| !v.is_finite() || *v < 0.0)
            {
                return Err(
                    "geometry or corrections produce nonfinite coordinates/invalid normalization"
                        .into(),
                );
            }
        }
        Ok(())
    }

    pub(super) fn grid_lut_key(&self, grid: &IntegrationGrid2d) -> super::cache::TwoDKey {
        let grid = if matches!(self.method, IntegrationMethod::PyfaiFull) {
            grid.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram)
        } else {
            *grid
        };
        self.shared_grid_key(grid)
    }

    /// Cache key of the shared-grid table for this integrator's method and unit.
    fn shared_grid_key(&self, grid: IntegrationGrid2d) -> super::cache::TwoDKey {
        super::cache::TwoDKey::SharedGrid {
            grid,
            unit: self.radial_unit,
            // 0 = counting, 1 = polygon splitting, >=2 = supersampling + 1
            sampling: match self.method {
                IntegrationMethod::Counting => 0,
                IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => 1,
                IntegrationMethod::Supersampled { n_ss } => n_ss + 1,
                IntegrationMethod::Histogram => unreachable!(),
            },
        }
    }

    /// Integrate a frame to 2D on a true azimuthal range. See
    /// [`Integrator::integrate2d_on_azimuthal_range_from_slice`].
    pub fn integrate2d_on_azimuthal_range(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_range: (f64, f64),
        errors: bool,
    ) -> Result<AzimuthalRange2dResult, String> {
        self.integrate2d_on_azimuthal_range_from_slice(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_radial,
            n_azimuthal,
            azimuthal_range,
            errors,
        )
    }

    /// Integrate to 2D on the azimuthal edges `(min, max)` in radians, split
    /// into `n_azimuthal` equal half-open bins `[min + i*step, min + (i+1)*step)`.
    ///
    /// Unlike [`Integrator::integrate2d_with_azimuthal_range`] (first-bin centre
    /// and step, wrapping periodically) this is a true range: a sector such as
    /// `(0, PI / 2)` receives only the pixels (or pixel fractions) inside it.
    /// Angles are periodic, so a sector may cross the ±π cut, e.g.
    /// `(2.5, 2.5 + PI / 2)`. Edges must be finite and resolvable, with a
    /// positive span of at most 2π. Any starting angle is accepted; starts
    /// outside ±2π are corrected to [-π, π), including the returned axis.
    /// For example, (-3π, -π) is corrected to the full turn (-π, π).
    ///
    /// The radial edges are the ones [`Integrator::integrate2d`] uses for the
    /// same method, unit and `n_radial`, so the radial axis matches the default
    /// cake. Binning uses the shared-grid tables (as `MultiIntegrator` does);
    /// mask, mask mode, solid-angle and polarization corrections and fast math
    /// apply exactly as on the default path. On the default path's own edges,
    /// counting gives identical values and supersampling identical values
    /// except next to the beam centre (the pixel holding it is sampled densely);
    /// pixel splitting uses exact fractional polygon areas and differs slightly
    /// from the legacy 2D pixel-split table (median 5e-5 relative per bin on the
    /// CeO2 Pilatus 1M test image).
    /// Non-finite pixels behave as on the default path: they are not rejected
    /// and make only the bins they contribute to NaN or infinite.
    /// `IntegrationMethod::Histogram` is rejected.
    ///
    /// With `IntegrationMethod::PyfaiFull`, individual bins depend on how the
    /// range is written: a range and the same range shifted by 2π can differ,
    /// as can the interior rows of a sector and the matching rows of a full
    /// circle, because pyFAI's full splitting unwraps pixels with a heuristic
    /// that depends on where the azimuth seam lies. pyFAI itself behaves the
    /// same (`docs/validation/pyfai-full-seam.md`). Pixel splitting agrees to
    /// about 1e-6 in the same comparison and is the better choice for sectors.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate2d_on_azimuthal_range_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_range: (f64, f64),
        errors: bool,
    ) -> Result<AzimuthalRange2dResult, String> {
        // Validate the azimuthal edges before any geometry work.
        IntegrationGrid2d::new(n_radial, n_azimuthal, (0.0, 1.0), azimuthal_range)?;
        // Like the default 2D path, non-finite pixels are not rejected: they
        // propagate into the bins they contribute to.
        self.validate_2d_setup(dim1, dim2, data.len())?;
        let view = self.for_shape(dim1, dim2);
        let radial_range = self.default_radial_edges_2d(&view, n_radial);
        // Values inside the edges must land in a bin even when their index
        // rounds to `n`, as on the default path.
        let mut grid =
            IntegrationGrid2d::new(n_radial, n_azimuthal, radial_range, azimuthal_range)?
                .with_clamped_index_rounding();
        match self.method {
            IntegrationMethod::PyfaiFull => {
                grid = grid.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram);
            }
            // The default counting edges end at the largest pixel value, which
            // `min_max_step`'s 10 ulp margin does not lift for values above ~32.
            IntegrationMethod::Counting => grid = grid.with_clamped_radial_end(),
            _ => (),
        }
        let normalization = view.normalization();
        let mask = self.mask.as_ref().map(|m| m.data.as_slice());
        let lut = view.ensure_lut2d(self.shared_grid_key(grid));
        let (intensity, errors) = if errors {
            let (intensity, errors) =
                self.lut_integrate2d_with_errors(data, &lut, mask, &normalization);
            (intensity, Some(errors))
        } else {
            (self.lut_integrate2d(data, &lut, mask, &normalization), None)
        };
        Ok(AzimuthalRange2dResult {
            intensity,
            errors,
            radial: grid.radial_centers(),
            azimuthal: grid.azimuthal_centers(),
        })
    }

    /// Integrate a frame to 2D on contiguous rows given by their centres. See
    /// [`Integrator::integrate2d_on_azimuthal_brackets_from_slice`].
    pub fn integrate2d_on_azimuthal_brackets(
        &self,
        frame: &Frame,
        n_radial: usize,
        center_range: (f64, f64),
        half_width: f64,
        errors: bool,
    ) -> Result<AzimuthalRange2dResult, String> {
        self.integrate2d_on_azimuthal_brackets_from_slice(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_radial,
            center_range,
            half_width,
            errors,
        )
    }

    /// Integrate to 2D on contiguous azimuthal rows described by their centres,
    /// as used in texture analysis: the first row is centred on
    /// `center_range.0` and every row covers its centre ± `half_width`
    /// (radians). `(0, 2π)` with a 2.5° bracket gives 72 rows centred on
    /// 0°, 5°, …, 355°, so directions such as 0° and 90° each sit in the middle
    /// of a row. The number of rows is `result.azimuthal.len()`.
    ///
    /// This is [`Integrator::integrate2d_on_azimuthal_range_from_slice`] on the
    /// edges from [`azimuthal_bracket_bins`]; everything documented there
    /// applies. The span must be a whole number of rows.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate2d_on_azimuthal_brackets_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        center_range: (f64, f64),
        half_width: f64,
        errors: bool,
    ) -> Result<AzimuthalRange2dResult, String> {
        let (n_azimuthal, azimuthal_range) = azimuthal_bracket_bins(center_range, half_width)?;
        self.integrate2d_on_azimuthal_range_from_slice(
            data,
            dim1,
            dim2,
            n_radial,
            n_azimuthal,
            azimuthal_range,
            errors,
        )
    }

    /// Radial edges the default 2D path uses for this method, unit and size.
    /// Cached: computing them needs a pass over all pixels (or pixel corners).
    fn default_radial_edges_2d(
        &self,
        view: &super::cache::ShapeView<'_>,
        n_radial: usize,
    ) -> (f64, f64) {
        let (dim1, dim2) = (view.state.dim1, view.state.dim2);
        let key = RadialEdgesKey {
            dim1,
            dim2,
            n_radial,
            unit: self.radial_unit,
            method: match self.method {
                IntegrationMethod::Counting => 0,
                IntegrationMethod::PixelSplit => 1,
                IntegrationMethod::PyfaiFull => 2,
                IntegrationMethod::Supersampled { n_ss } => n_ss + 3,
                IntegrationMethod::Histogram => unreachable!(),
            },
            radial_range: self
                .radial_range
                .map(|(lo, hi)| [lo.to_bits(), hi.to_bits()]),
        };
        let cache = &self.radial_edges_2d;
        if let Some((cached, edges)) = *cache.read().expect("radial edge cache lock poisoned")
            && cached == key
        {
            return edges;
        }
        let state = &view.state;
        let mut geom = self.geometry.clone();
        geom.dim1 = dim1;
        geom.dim2 = dim2;
        let unit = self.radial_unit;
        let edges = match self.method {
            IntegrationMethod::Counting => {
                let (lo, hi, _) = histogram::min_max_step(&state.radial, n_radial);
                (lo, hi)
            }
            IntegrationMethod::PixelSplit => {
                let (tth_corners, _) = geom.angular_corners();
                let corners = unit.convert_from_tth(&tth_corners, geom.poni.wavelength);
                (
                    math::min(&corners),
                    math::max(&corners) + 10.0 * f64::EPSILON,
                )
            }
            IntegrationMethod::PyfaiFull => self.radial_range.unwrap_or_else(|| {
                let [lo, hi] = crate::lut::pyfai_radial_range(&geom, unit);
                (lo, hi)
            }),
            IntegrationMethod::Supersampled { n_ss } => {
                let (lo, hi, _) =
                    histogram::ss_tth_min_max_step(&geom, &state.radial, n_radial, n_ss, unit);
                (lo, hi)
            }
            IntegrationMethod::Histogram => unreachable!(),
        };
        *cache.write().expect("radial edge cache lock poisoned") = Some((key, edges));
        edges
    }

    pub(crate) fn validate_2d_input(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
    ) -> Result<(), String> {
        self.validate_2d_setup(dim1, dim2, data.len())?;
        // Nonfinite pixels would poison raw sums. Masked pixels are allowed.
        if data
            .iter()
            .enumerate()
            .any(|(i, x)| !x.is_finite() && self.mask.as_ref().is_none_or(|m| m.data[i] == 0))
        {
            return Err("unmasked image pixels must be finite".into());
        }
        Ok(())
    }

    /// Shape, mask, geometry and method checks shared by the 2D grid paths;
    /// the pixel values themselves are not inspected.
    pub(super) fn validate_2d_setup(
        &self,
        dim1: usize,
        dim2: usize,
        data_len: usize,
    ) -> Result<(), String> {
        if dim1 == 0
            || dim2 == 0
            || dim1.checked_mul(dim2) != Some(data_len)
            || dim1
                .checked_add(1)
                .and_then(|d| dim2.checked_add(1).and_then(|e| d.checked_mul(e)))
                .is_none()
        {
            return Err("image dimensions must be positive and match data length".into());
        }
        if let Some(mask) = &self.mask
            && (mask.dim1 != dim1 || mask.dim2 != dim2 || mask.data.len() != data_len)
        {
            return Err("mask dimensions must match its detector image".into());
        }
        let p = &self.geometry.poni;
        if [p.pixel1, p.pixel2, p.distance]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
            || [p.poni1, p.poni2, p.rot1, p.rot2, p.rot3]
                .iter()
                .any(|x| !x.is_finite())
            || (self.radial_unit.requires_wavelength()
                && (!p.wavelength.is_finite() || p.wavelength <= 0.0))
        {
            return Err("geometry requires finite parameters, positive pixels/distance, and a positive wavelength for q units".into());
        }
        if self
            .polarization_factor
            .is_some_and(|p| !p.is_finite() || !(-1.0..=1.0).contains(&p))
        {
            return Err("polarization_factor must be finite and in [-1, 1]".into());
        }
        match self.method {
            IntegrationMethod::Histogram => {
                return Err("shared 2D integration does not support Histogram".into());
            }
            IntegrationMethod::Supersampled { n_ss }
                if n_ss == 0
                    || n_ss.checked_add(1).is_none()
                    || n_ss.checked_mul(n_ss).is_none()
                    || dim2.checked_mul(n_ss).is_none() =>
            {
                return Err("n_ss must be positive and fit sampling indices".into());
            }
            _ => (),
        }
        Ok(())
    }
}

pub(crate) fn validate_variances(data: &[f64], variance: &[f64]) -> Result<(), String> {
    if variance.len() != data.len() || variance.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("pixel variances must match the image and be finite and nonnegative".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "azimuthal_range_tests.rs"]
mod azimuthal_range_tests;
