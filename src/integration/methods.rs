use crate::frame::Frame;
use crate::math::histogram;

use super::cache::{AzimuthalRangeKey, OneDKey, TwoDKey};
use super::{IntegrationMethod, Integrator};

impl Integrator {
    /// Integrate a frame to 1D using the configured integration method.
    /// Thread-safe: can be called from multiple threads on a shared `Arc<Integrator>`.
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
    /// let integrator = Integrator::new(geometry)
    ///     .with_method(IntegrationMethod::Counting);
    /// let (_x, y_count) = integrator.integrate1d(&frame, 4);
    /// assert_eq!(y_count.len(), 4);
    /// ```
    pub fn integrate1d(&self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram(&radial, &intensity, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points, unit });
                crate::lut::integrate1d(&intensity, &lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points, unit });
                crate::lut::integrate1d(&intensity, &lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                });
                crate::lut::integrate1d(&intensity, &lut1d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 1D and return per-bin errors (standard deviations).
    pub fn integrate1d_with_errors(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram_with_errors(
                    &radial, &intensity, &variance, n_points, mask_data,
                )
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points, unit });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points, unit });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &lut1d, mask_data)
            }
        };
        result
    }

    pub fn integrate1d_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let intensity = self.correct_intensity(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram(&radial, &intensity, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points, unit });
                crate::lut::integrate1d(&intensity, &lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points, unit });
                crate::lut::integrate1d(&intensity, &lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                });
                crate::lut::integrate1d(&intensity, &lut1d, mask_data)
            }
        }
    }

    pub fn integrate1d_with_errors_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram_with_errors(
                    &radial, &intensity, &variance, n_points, mask_data,
                )
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points, unit });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points, unit });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &lut1d, mask_data)
            }
        }
    }

    /// Integrate a frame to 2D using the configured integration method.
    /// Thread-safe: can be called from multiple threads on a shared `Arc<Integrator>`.
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
    /// let integrator = Integrator::new(geometry)
    ///     .with_method(IntegrationMethod::PixelSplit);
    /// let y = integrator.integrate2d(&frame, 2, 2);
    /// assert_eq!(y.len(), 4);
    /// ```
    pub fn integrate2d(&self, frame: &Frame, n_radial: usize, n_azimuthal: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram(&radial, &intensity, n_radial, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
        };
        result
    }

    pub fn integrate2d_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> Vec<f64> {
        self.update_geometry_arrays(dim1, dim2);
        let intensity = self.correct_intensity(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram(&radial, &intensity, n_radial, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
        }
    }

    /// Integrate a frame to 2D using an explicit azimuthal start and step.
    /// Thread-safe: can be called from multiple threads on a shared `Arc<Integrator>`.
    ///
    /// For a full-circle azimuthal range, use `azimuthal_step = 2*pi / n_azimuthal` and a
    /// `azimuthal_start` that defines the center of the first bin (e.g. `0.0` for 0..2*pi).
    /// Azimuthal bins wrap periodically, so angles outside the total range fold into
    /// `0..n_azimuthal`.
    /// Note: `IntegrationMethod::Histogram` ignores azimuthal binning.
    pub fn integrate2d_with_azimuthal_range(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram(&radial, &intensity, n_radial, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
        };
        result
    }

    pub fn integrate2d_with_azimuthal_range_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> Vec<f64> {
        self.update_geometry_arrays(dim1, dim2);
        let intensity = self.correct_intensity(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                histogram::histogram(&radial, &intensity, n_radial, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d(&intensity, &lut2d, mask_data)
            }
        }
    }

    /// Integrate a frame to 2D and return per-bin errors (standard deviations).
    pub fn integrate2d_with_errors(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &radial, &intensity, &variance, n_radial, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
        };
        result
    }

    pub fn integrate2d_with_errors_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &radial, &intensity, &variance, n_radial, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
        }
    }

    /// Integrate a frame to 2D with errors using an explicit azimuthal binning definition.
    ///
    /// `azimuthal_start` is the center of the first azimuthal bin; the effective start edge is
    /// `azimuthal_start - azimuthal_step / 2`. Azimuthal bins wrap periodically, so angles outside
    /// the total range fold into `0..n_azimuthal`.
    /// Note: `IntegrationMethod::Histogram` ignores azimuthal binning.
    pub fn integrate2d_with_azimuthal_range_and_errors(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &radial, &intensity, &variance, n_radial, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
        };
        result
    }

    pub fn integrate2d_with_azimuthal_range_and_errors_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                let radial = self
                    .state
                    .read()
                    .expect("geometry state lock poisoned")
                    .radial
                    .clone();
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &radial, &intensity, &variance, n_radial, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &lut2d, mask_data)
            }
        }
    }
}
