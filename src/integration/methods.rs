use crate::frame::Frame;
use crate::math::histogram;

use super::cache::{ChiRangeKey, OneDKey, TwoDKey};
use super::{IntegrationMethod, Integrator};

impl Integrator {
    /// Integrate a frame to 1D using the configured integration method.
    ///
    /// Examples
    /// ```
    /// use dioptrin::integration::{Integrator, IntegrationMethod};
    /// use dioptrin::{frame::Frame, geometry::Geometry, poni::Poni};
    ///
    /// let mut poni = Poni::new();
    /// poni.pixel1 = 1.0;
    /// poni.pixel2 = 1.0;
    /// poni.distance = 1.0;
    ///
    /// let frame = Frame {
    ///     name: String::from("demo"),
    ///     data: vec![1.0, 2.0, 3.0, 4.0],
    ///     dim1: 2,
    ///     dim2: 2,
    /// };
    ///
    /// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    /// let mut integrator = Integrator::new(geometry)
    ///     .with_method(IntegrationMethod::Counting);
    /// let (_x, y_count) = integrator.integrate1d(&frame, 4);
    /// integrator.set_method(IntegrationMethod::PixelSplit);
    /// let (_x, y_ps) = integrator.integrate1d(&frame, 4);
    /// integrator.set_method(IntegrationMethod::Supersampled { n_ss: 2 });
    /// let (_x, y_ss) = integrator.integrate1d(&frame, 4);
    /// assert_eq!(y_count.len(), 4);
    /// assert_eq!(y_ps.len(), 4);
    /// assert_eq!(y_ss.len(), 4);
    /// ```
    pub fn integrate1d(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 1D and return per-bin errors (standard deviations).
    pub fn integrate1d_with_errors(
        &mut self,
        frame: &Frame,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => histogram::histogram_with_errors(
                &self.tth, &intensity, &variance, n_points, mask_data,
            ),
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
        };
        result
    }

    pub fn integrate1d_from_slice(
        &mut self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let intensity = self.correct_intensity(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            &mask.data
        });
        match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
        }
    }

    pub fn integrate1d_with_errors_from_slice(
        &mut self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            &mask.data
        });
        match self.method {
            IntegrationMethod::Histogram => histogram::histogram_with_errors(
                &self.tth, &intensity, &variance, n_points, mask_data,
            ),
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
        }
    }

    /// Integrate a frame to 2D using the configured integration method.
    ///
    /// Example
    /// ```
    /// use dioptrin::integration::{Integrator, IntegrationMethod};
    /// use dioptrin::{frame::Frame, geometry::Geometry, poni::Poni};
    ///
    /// let mut poni = Poni::new();
    /// poni.pixel1 = 1.0;
    /// poni.pixel2 = 1.0;
    /// poni.distance = 1.0;
    ///
    /// let frame = Frame {
    ///     name: String::from("demo"),
    ///     data: vec![1.0, 2.0, 3.0, 4.0],
    ///     dim1: 2,
    ///     dim2: 2,
    /// };
    ///
    /// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    /// let mut integrator = Integrator::new(geometry)
    ///     .with_method(IntegrationMethod::PixelSplit);
    /// let y = integrator.integrate2d(&frame, 2, 2);
    /// assert_eq!(y.len(), 4);
    /// ```
    pub fn integrate2d(&mut self, frame: &Frame, n_tth: usize, n_chi: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_tth, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting { n_tth, n_chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_chi, n_ss });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 2D using an explicit chi start and step.
    ///
    /// For a full-circle chi range, use `chi_step = 2*pi / n_chi` and a
    /// `chi_start` that defines the center of the first bin (e.g. `0.0` for 0..2*pi).
    /// Chi bins wrap periodically, so angles outside the total range fold into
    /// `0..n_chi`.
    /// Note: `IntegrationMethod::Histogram` ignores chi binning.
    pub fn integrate2d_with_chi_range(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_chi: usize,
        chi_start: f64,
        chi_step: f64,
    ) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let chi = ChiRangeKey::new(chi_start, chi_step);
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_tth, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingChi { n_tth, n_chi, chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitChi { n_tth, n_chi, chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledChi { n_tth, n_chi, n_ss, chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 2D and return per-bin errors (standard deviations).
    pub fn integrate2d_with_errors(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_chi: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &self.tth, &intensity, &variance, n_tth, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting { n_tth, n_chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_chi, n_ss });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 2D with errors using an explicit chi binning definition.
    ///
    /// `chi_start` is the center of the first chi bin; the effective start edge is
    /// `chi_start - chi_step / 2`. Chi bins wrap periodically, so angles outside
    /// the total range fold into `0..n_chi`.
    /// Note: `IntegrationMethod::Histogram` ignores chi binning.
    pub fn integrate2d_with_chi_range_and_errors(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_chi: usize,
        chi_start: f64,
        chi_step: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let chi = ChiRangeKey::new(chi_start, chi_step);
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &self.tth, &intensity, &variance, n_tth, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingChi { n_tth, n_chi, chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitChi { n_tth, n_chi, chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledChi { n_tth, n_chi, n_ss, chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
        };
        result
    }
}
