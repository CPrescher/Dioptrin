use super::{IntegrationMethod, Integrator};
use crate::lut::IntermediateResult2d;
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
        if azimuthal_range.0 < -TAU
            || azimuthal_range.0 > TAU
            || azimuthal_range.1 - azimuthal_range.0 > TAU + 1e-12
        {
            return Err("azimuthal start must be in [-2π, 2π] and span must be at most 2π".into());
        }
        Ok(Self {
            pixel_split_mode: PixelSplitMode::Fractional,
            n_radial,
            n_azimuthal,
            radial: [radial_range.0.to_bits(), radial_range.1.to_bits()],
            azimuthal: [azimuthal_range.0.to_bits(), azimuthal_range.1.to_bits()],
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
        let azimuthal = a0 + (azimuthal - a0).rem_euclid(TAU);
        if !radial.is_finite()
            || !azimuthal.is_finite()
            || radial < r0
            || radial >= r1
            || azimuthal >= a1
        {
            return None;
        }
        let r = ((radial - r0) / self.radial_step()) as usize;
        let a = ((azimuthal - a0) / self.azimuthal_step()) as usize;
        (r < self.n_radial && a < self.n_azimuthal).then_some(a * self.n_radial + r)
    }
}

fn centers((lo, hi): (f64, f64), count: usize) -> Vec<f64> {
    let step = (hi - lo) / count as f64;
    (0..count).map(|i| lo + (i as f64 + 0.5) * step).collect()
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
        self.update_geometry_arrays(dim1, dim2);
        let normalization = self.get_normalization();
        {
            let state = self.state.read().map_err(|e| e.to_string())?;
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
        let mask = self.mask.as_ref().map(|m| m.data.as_slice());
        let key = super::cache::TwoDKey::SharedGrid {
            grid,
            unit: self.radial_unit,
            // 0 = counting, 1 = polygon splitting, >=2 = supersampling + 1
            sampling: match self.method {
                IntegrationMethod::Counting => 0,
                IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => 1,
                IntegrationMethod::Supersampled { n_ss } => n_ss + 1,
                IntegrationMethod::Histogram => unreachable!(),
            },
        };
        let lut = self.ensure_lut2d(key);
        Ok(if let Some(variance) = variance {
            lut.raw2d_with_variances(data, variance, mask, &normalization)
        } else if errors {
            self.lut_integrate2d_raw_with_variance(data, &lut, mask, &normalization)
        } else {
            self.lut_integrate2d_raw(data, &lut, mask, &normalization)
        })
    }

    pub(crate) fn validate_2d_input(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
    ) -> Result<(), String> {
        if dim1 == 0
            || dim2 == 0
            || dim1.checked_mul(dim2) != Some(data.len())
            || dim1
                .checked_add(1)
                .and_then(|d| dim2.checked_add(1).and_then(|e| d.checked_mul(e)))
                .is_none()
        {
            return Err("image dimensions must be positive and match data length".into());
        }
        if let Some(mask) = &self.mask
            && (mask.dim1 != dim1 || mask.dim2 != dim2 || mask.data.len() != data.len())
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
}

pub(crate) fn validate_variances(data: &[f64], variance: &[f64]) -> Result<(), String> {
    if variance.len() != data.len() || variance.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("pixel variances must match the image and be finite and nonnegative".into());
    }
    Ok(())
}
