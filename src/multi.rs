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
use crate::integration::{IntegrationMethod, Integrator};

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
