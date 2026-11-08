//! Multi-detector integration support.
//!
//! The [`MultiIntegrator`] merges integration results from multiple detectors into
//! a single diffraction pattern, similar to pyFAI's `MultiGeometry`. Each detector
//! contributes raw signal, normalization, and variance to shared radial bins. These
//! are summed across detectors, then divided once at the end.
//!
//! # Example
//!
//! ```
//! use dioptrin::geometry::Geometry;
//! use dioptrin::integration::{Integrator, IntegrationMethod};
//! use dioptrin::frame::Frame;
//! use dioptrin::multi::MultiIntegrator;
//! use dioptrin::poni::Poni;
//!
//! let mut poni = Poni::new();
//! poni.pixel1 = 1.0;
//! poni.pixel2 = 1.0;
//! poni.distance = 100.0;
//!
//! let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
//!     .with_method(IntegrationMethod::Counting)
//!     .with_radial_range(0.0, 0.2);
//! let int2 = Integrator::new(Geometry::new(10, 10, poni))
//!     .with_method(IntegrationMethod::Counting)
//!     .with_radial_range(0.0, 0.2);
//!
//! let multi = MultiIntegrator::new(vec![int1, int2]);
//!
//! let frame1 = Frame { name: "det1".into(), data: vec![1.0; 100], dim1: 10, dim2: 10 };
//! let frame2 = Frame { name: "det2".into(), data: vec![2.0; 100], dim1: 10, dim2: 10 };
//! let (bins, intensity) = multi.integrate1d(&[&frame1, &frame2], 20);
//! assert_eq!(bins.len(), 20);
//! ```

use std::sync::Arc;

use crate::frame::Frame;
use crate::integration::IntegrationGrid2d;

#[cfg(test)]
mod tests;
use crate::integration::{IntegrationMethod, Integrator};

/// Denominator convention for the new shared-grid 2D path only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MultiNormalization {
    /// Per-pixel relative solid angle × polarization (ordinary Dioptrin scale).
    #[default]
    Relative,
    /// Also multiply each detector's denominator by pixel1*pixel2/distance².
    /// Matches pyFAI MultiGeometry(correctSolidAngle=True) absolute scaling.
    Absolute,
}

/// Options for shared-grid 2D integration. Scaling factors are positive monitor
/// values multiplying each detector's normalization, not its signal/variance.
#[derive(Clone, Debug)]
pub struct Multi2dOptions {
    pub azimuthal_range: (f64, f64),
    pub errors: bool,
    pub normalization: MultiNormalization,
    pub normalization_factors: Option<Vec<f64>>,
}

impl Default for Multi2dOptions {
    fn default() -> Self {
        Self {
            azimuthal_range: (-std::f64::consts::PI, std::f64::consts::PI),
            errors: false,
            normalization: MultiNormalization::Relative,
            normalization_factors: None,
        }
    }
}

/// Combined cake, flattened as azimuthal*n_radial + radial. Errors are Poisson
/// propagated std. Empty bins have zero intensity and zero error. Raw sums are
/// included for coverage inspection and further statistically correct merging.
pub struct Multi2dResult {
    pub radial: Vec<f64>,
    pub azimuthal: Vec<f64>,
    pub intensity: Vec<f64>,
    pub errors: Option<Vec<f64>>,
    pub raw: crate::lut::IntermediateResult2d,
}

/// Multi-detector integrator that merges integration results from multiple detectors.
///
/// All integrators must share the same radial unit and radial range to ensure
/// aligned bin grids. The merge sums raw signal, normalization, and variance
/// across detectors before dividing.
pub struct MultiIntegrator {
    integrators: Vec<Arc<Integrator>>,
}

fn validate_integrators(integrators: &[Arc<Integrator>]) -> Result<(), String> {
    if integrators.is_empty() {
        return Err("MultiIntegrator requires at least one integrator".into());
    }

    let first = &integrators[0];
    let unit = first.radial_unit();
    let range = first
        .radial_range()
        .ok_or("All integrators must have a radial_range set for MultiIntegrator")?;

    for (i, integrator) in integrators.iter().enumerate() {
        if matches!(integrator.method(), IntegrationMethod::Histogram) {
            return Err(format!(
                "MultiIntegrator does not support Histogram method (integrator {})",
                i
            ));
        }
        if integrator.radial_unit() != unit {
            return Err(format!(
                "All integrators must have the same radial_unit (integrator {} differs)",
                i
            ));
        }
        let r = integrator.radial_range().ok_or(format!(
            "All integrators must have a radial_range set (integrator {} missing)",
            i
        ))?;
        if (r.0 - range.0).abs() > f64::EPSILON || (r.1 - range.1).abs() > f64::EPSILON {
            return Err(format!(
                "All integrators must have the same radial_range (integrator {} differs)",
                i
            ));
        }
    }
    Ok(())
}

impl MultiIntegrator {
    /// Fallible owned constructor; unlike `new`, invalid inputs return an error.
    pub fn try_new(integrators: Vec<Integrator>) -> Result<Self, String> {
        Self::from_arcs(integrators.into_iter().map(Arc::new).collect())
    }

    /// Integrate one independent frame per detector on a common grid. Radial
    /// edges come from the integrators' shared radial_range; azimuthal edges in
    /// options are radians. No detector mosaic or normalized-cake average is used.
    pub fn integrate2d(
        &self,
        frames: &[&Frame],
        n_radial: usize,
        n_azimuthal: usize,
        options: &Multi2dOptions,
    ) -> Result<Multi2dResult, String> {
        let slices: Vec<_> = frames
            .iter()
            .map(|f| (f.data.as_slice(), f.dim1, f.dim2))
            .collect();
        self.integrate2d_from_slices(&slices, n_radial, n_azimuthal, options)
    }

    /// Borrow one row-major f64 image per detector. All inputs are checked before
    /// integration. Counting, PixelSplit and Supersampled are supported.
    pub fn integrate2d_from_slices(
        &self,
        slices: &[(&[f64], usize, usize)],
        n_radial: usize,
        n_azimuthal: usize,
        options: &Multi2dOptions,
    ) -> Result<Multi2dResult, String> {
        validate_integrators(&self.integrators)?;
        if slices.len() != self.integrators.len() {
            return Err(format!(
                "Expected {} images (one per detector), got {}",
                self.integrators.len(),
                slices.len()
            ));
        }
        let grid = IntegrationGrid2d::new(
            n_radial,
            n_azimuthal,
            self.integrators[0].radial_range().unwrap(),
            options.azimuthal_range,
        )?;
        if self
            .integrators
            .iter()
            .any(|i| i.radial_range() != Some(grid.radial_range()))
        {
            return Err(
                "All integrators must have identical finite radial_range edges for 2D integration"
                    .into(),
            );
        }
        if let Some(factors) = &options.normalization_factors
            && (factors.len() != slices.len()
                || factors.iter().any(|f| !f.is_finite() || *f <= 0.0))
        {
            return Err(
                "normalization_factors must contain one finite positive monitor value per detector"
                    .into(),
            );
        }
        let mut factors = Vec::with_capacity(slices.len());
        for (i, (integrator, &(data, dim1, dim2))) in
            self.integrators.iter().zip(slices).enumerate()
        {
            integrator
                .validate_2d_input(data, dim1, dim2)
                .map_err(|e| format!("detector {i}: {e}"))?;
            let monitor = options.normalization_factors.as_ref().map_or(1.0, |f| f[i]);
            let p = &integrator.geometry().poni;
            let sac = match options.normalization {
                MultiNormalization::Relative => 1.0,
                MultiNormalization::Absolute => (p.pixel1 / p.distance) * (p.pixel2 / p.distance),
            };
            let factor = monitor * sac;
            if !factor.is_finite() || factor <= 0.0 {
                return Err(format!(
                    "detector {i}: normalization factor underflows or overflows"
                ));
            }
            factors.push(factor);
        }
        let rows = n_radial * n_azimuthal;
        let mut total = crate::lut::IntermediateResult2d {
            sum_signal: vec![0.0; rows],
            sum_norm: vec![0.0; rows],
            sum_variance: options.errors.then(|| vec![0.0; rows]),
            count: vec![0.0; rows],
            n_radial,
            n_azimuthal,
        };
        for ((integrator, &(data, dim1, dim2)), factor) in
            self.integrators.iter().zip(slices).zip(factors)
        {
            let raw = integrator.integrate2d_raw_on_grid_from_slice(
                data,
                dim1,
                dim2,
                &grid,
                options.errors,
            )?;
            for j in 0..rows {
                total.sum_signal[j] += raw.sum_signal[j];
                total.sum_norm[j] += raw.sum_norm[j] * factor;
                total.count[j] += raw.count[j];
                if let (Some(tv), Some(v)) = (&mut total.sum_variance, &raw.sum_variance) {
                    tv[j] += v[j];
                }
            }
        }
        let intensity = total
            .sum_signal
            .iter()
            .zip(&total.sum_norm)
            .map(|(&s, &n)| if n == 0.0 { 0.0 } else { s / n })
            .collect();
        let errors = total.sum_variance.as_ref().map(|v| {
            v.iter()
                .zip(&total.sum_norm)
                .map(|(&v, &n)| if n == 0.0 { 0.0 } else { v.max(0.0).sqrt() / n })
                .collect()
        });
        Ok(Multi2dResult {
            radial: grid.radial_centers(),
            azimuthal: grid.azimuthal_centers(),
            intensity,
            errors,
            raw: total,
        })
    }

    /// Create a new MultiIntegrator from a list of owned Integrators.
    ///
    /// # Panics
    /// - If the list is empty
    /// - If any integrator uses the Histogram method
    /// - If any integrator does not have a radial_range set
    /// - If the radial_unit or radial_range differs across integrators
    pub fn new(integrators: Vec<Integrator>) -> Self {
        let arcs: Vec<Arc<Integrator>> = integrators.into_iter().map(Arc::new).collect();
        validate_integrators(&arcs).unwrap();
        Self { integrators: arcs }
    }

    /// Create a new MultiIntegrator from Arc-wrapped Integrators.
    /// Returns Err with a message if validation fails.
    pub fn from_arcs(integrators: Vec<Arc<Integrator>>) -> Result<Self, String> {
        validate_integrators(&integrators)?;
        Ok(Self { integrators })
    }

    /// Integrate one frame per detector and merge into a single 1D pattern.
    pub fn integrate1d(&self, frames: &[&Frame], n_points: usize) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(
            frames.len(),
            self.integrators.len(),
            "Number of frames must match number of integrators"
        );

        let mut total_signal = vec![0.0; n_points];
        let mut total_norm = vec![0.0; n_points];
        let mut bins = Vec::new();

        for (integrator, frame) in self.integrators.iter().zip(frames.iter()) {
            let raw = integrator.integrate1d_raw(frame, n_points);
            if bins.is_empty() {
                bins = raw.bins;
            }
            for j in 0..n_points {
                total_signal[j] += raw.sum_signal[j];
                total_norm[j] += raw.sum_norm[j];
            }
        }

        let intensity: Vec<f64> = total_signal
            .iter()
            .zip(total_norm.iter())
            .map(|(&s, &n)| if n == 0.0 { 0.0 } else { s / n })
            .collect();

        (bins, intensity)
    }

    /// Integrate one frame per detector and merge into a single 1D pattern with errors.
    pub fn integrate1d_with_errors(
        &self,
        frames: &[&Frame],
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        assert_eq!(
            frames.len(),
            self.integrators.len(),
            "Number of frames must match number of integrators"
        );

        let mut total_signal = vec![0.0; n_points];
        let mut total_norm = vec![0.0; n_points];
        let mut total_var = vec![0.0; n_points];
        let mut bins = Vec::new();

        for (integrator, frame) in self.integrators.iter().zip(frames.iter()) {
            let raw = integrator.integrate1d_raw_with_variance(frame, n_points);
            if bins.is_empty() {
                bins = raw.bins;
            }
            let var = raw.sum_variance.unwrap();
            for j in 0..n_points {
                total_signal[j] += raw.sum_signal[j];
                total_norm[j] += raw.sum_norm[j];
                total_var[j] += var[j];
            }
        }

        let mut intensity = Vec::with_capacity(n_points);
        let mut sigma = Vec::with_capacity(n_points);
        for j in 0..n_points {
            if total_norm[j] == 0.0 {
                intensity.push(0.0);
                sigma.push(0.0);
            } else {
                intensity.push(total_signal[j] / total_norm[j]);
                sigma.push(
                    (total_var[j] / (total_norm[j] * total_norm[j]))
                        .max(0.0)
                        .sqrt(),
                );
            }
        }

        (bins, intensity, sigma)
    }

    /// Integrate one slice per detector and merge into a single 1D pattern.
    pub fn integrate1d_from_slices(
        &self,
        slices: &[(&[f64], usize, usize)],
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(
            slices.len(),
            self.integrators.len(),
            "Number of slices must match number of integrators"
        );

        let mut total_signal = vec![0.0; n_points];
        let mut total_norm = vec![0.0; n_points];
        let mut bins = Vec::new();

        for (integrator, &(data, dim1, dim2)) in self.integrators.iter().zip(slices.iter()) {
            let raw = integrator.integrate1d_raw_from_slice(data, dim1, dim2, n_points);
            if bins.is_empty() {
                bins = raw.bins;
            }
            for j in 0..n_points {
                total_signal[j] += raw.sum_signal[j];
                total_norm[j] += raw.sum_norm[j];
            }
        }

        let intensity: Vec<f64> = total_signal
            .iter()
            .zip(total_norm.iter())
            .map(|(&s, &n)| if n == 0.0 { 0.0 } else { s / n })
            .collect();

        (bins, intensity)
    }

    /// Integrate one slice per detector and merge into a single 1D pattern with errors.
    pub fn integrate1d_with_errors_from_slices(
        &self,
        slices: &[(&[f64], usize, usize)],
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        assert_eq!(
            slices.len(),
            self.integrators.len(),
            "Number of slices must match number of integrators"
        );

        let mut total_signal = vec![0.0; n_points];
        let mut total_norm = vec![0.0; n_points];
        let mut total_var = vec![0.0; n_points];
        let mut bins = Vec::new();

        for (integrator, &(data, dim1, dim2)) in self.integrators.iter().zip(slices.iter()) {
            let raw =
                integrator.integrate1d_raw_with_variance_from_slice(data, dim1, dim2, n_points);
            if bins.is_empty() {
                bins = raw.bins;
            }
            let var = raw.sum_variance.unwrap();
            for j in 0..n_points {
                total_signal[j] += raw.sum_signal[j];
                total_norm[j] += raw.sum_norm[j];
                total_var[j] += var[j];
            }
        }

        let mut intensity = Vec::with_capacity(n_points);
        let mut sigma = Vec::with_capacity(n_points);
        for j in 0..n_points {
            if total_norm[j] == 0.0 {
                intensity.push(0.0);
                sigma.push(0.0);
            } else {
                intensity.push(total_signal[j] / total_norm[j]);
                sigma.push(
                    (total_var[j] / (total_norm[j] * total_norm[j]))
                        .max(0.0)
                        .sqrt(),
                );
            }
        }

        (bins, intensity, sigma)
    }

    /// Get a reference to the integrators.
    pub fn integrators(&self) -> &[Arc<Integrator>] {
        &self.integrators
    }
}
