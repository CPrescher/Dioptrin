use crate::frame::Frame;
use crate::math::histogram;

use super::cache::{AzimuthalRangeKey, OneDKey, RadialRangeKey, TwoDKey};
use super::{IntegrationMethod, Integrator};

impl Integrator {
    pub(super) fn radial_range_key(&self) -> Option<RadialRangeKey> {
        self.radial_range
            .map(|(min, max)| RadialRangeKey::new(min, max))
    }

    /// Integrate with the weighted population spread of corrected pixel intensities.
    /// `errors` independently enables the existing Poisson uncertainty estimate.
    /// See [`crate::lut::integrate1d_with_azimuthal_std`] for the precise statistic.
    pub fn integrate1d_with_azimuthal_std(
        &self,
        frame: &Frame,
        n_points: usize,
        errors: bool,
    ) -> crate::lut::AzimuthalIntegrationResult {
        self.integrate1d_with_azimuthal_std_from_slice(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_points,
            errors,
        )
    }

    /// Zero-copy input counterpart of [`Self::integrate1d_with_azimuthal_std`].
    pub fn integrate1d_with_azimuthal_std_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
        errors: bool,
    ) -> crate::lut::AzimuthalIntegrationResult {
        assert!(n_points > 0, "n_points must be positive");
        assert_eq!(data.len(), dim1 * dim2, "Image dimensions must match data");
        self.update_geometry_arrays(dim1, dim2);
        let normalization = self.get_normalization();
        let mask = self.mask.as_ref().map(|m| {
            m.assert_compatible(dim1, dim2);
            m.data.as_slice()
        });
        let variance = errors.then(|| data.iter().map(|&s| s.max(0.0)).collect::<Vec<_>>());
        if matches!(self.method, IntegrationMethod::Histogram) {
            // Preserve Histogram's legacy mean of pre-corrected intensities and
            // its automatic radial range, rather than changing its weighting.
            let radial = self
                .state
                .read()
                .expect("geometry state lock poisoned")
                .radial
                .clone();
            let lut = crate::lut::build_lut1d_csr(&radial, n_points);
            let corrected: Vec<_> = data
                .iter()
                .zip(normalization.iter())
                .map(|(&s, &n)| if n > 0.0 { s / n } else { f64::NAN })
                .collect();
            let corrected_variance = variance.map(|v| {
                v.iter()
                    .zip(normalization.iter())
                    .map(|(&v, &n)| v / (n * n))
                    .collect::<Vec<_>>()
            });
            return crate::lut::integrate1d_with_azimuthal_std(
                &corrected,
                &lut,
                mask,
                &vec![1.0; data.len()],
                corrected_variance.as_deref(),
            );
        }
        let unit = self.radial_unit;
        let radial_range = self.radial_range_key();
        let key = match self.method {
            IntegrationMethod::Counting => OneDKey::Counting {
                n_points,
                unit,
                radial_range,
            },
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => OneDKey::PixelSplit {
                pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                n_points,
                unit,
                radial_range,
            },
            IntegrationMethod::Supersampled { n_ss } => OneDKey::Supersampled {
                n_points,
                n_ss,
                unit,
                radial_range,
            },
            IntegrationMethod::Histogram => unreachable!(),
        };
        let lut = self.ensure_lut1d(key);
        crate::lut::azimuthal::integrate_with_columns(
            data,
            &lut.row_ptr,
            |k| lut.col_idx.get(k),
            lut.values.as_deref(),
            lut.bins(),
            mask,
            &normalization,
            variance.as_deref(),
        )
    }

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
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
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
                // Histogram uses pre-corrected intensity for backward compat
                let corrected: Vec<f64> = frame
                    .data
                    .iter()
                    .zip(normalization.iter())
                    .map(|(&s, &n)| if n > 0.0 { s / n } else { 0.0 })
                    .collect();
                histogram::histogram(&radial, &corrected, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d(&frame.data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d(&frame.data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d(&frame.data, &lut1d, mask_data, &normalization[..])
            }
        }
    }

    /// Integrate a frame to 1D and return per-bin errors (standard deviations).
    pub fn integrate1d_with_errors(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let normalization = self.get_normalization();
        // Variance under Poisson assumption: var(pixel) = signal (raw counts)
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
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
                let corrected: Vec<f64> = frame
                    .data
                    .iter()
                    .zip(normalization.iter())
                    .map(|(&s, &n)| if n > 0.0 { s / n } else { 0.0 })
                    .collect();
                let variance: Vec<f64> = frame.data.iter().map(|&x| x.max(0.0)).collect();
                let corrected_var: Vec<f64> = variance
                    .iter()
                    .zip(normalization.iter())
                    .map(|(&v, &n)| if n > 0.0 { v / (n * n) } else { 0.0 })
                    .collect();
                histogram::histogram_with_errors(
                    &radial,
                    &corrected,
                    &corrected_var,
                    n_points,
                    mask_data,
                )
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_with_errors(&frame.data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_with_errors(&frame.data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_with_errors(&frame.data, &lut1d, mask_data, &normalization[..])
            }
        }
    }

    pub fn integrate1d_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let normalization = self.get_normalization();
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
                let corrected: Vec<f64> = data
                    .iter()
                    .zip(normalization.iter())
                    .map(|(&s, &n)| if n > 0.0 { s / n } else { 0.0 })
                    .collect();
                histogram::histogram(&radial, &corrected, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d(data, &lut1d, mask_data, &normalization[..])
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
        let normalization = self.get_normalization();
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
                let corrected: Vec<f64> = data
                    .iter()
                    .zip(normalization.iter())
                    .map(|(&s, &n)| if n > 0.0 { s / n } else { 0.0 })
                    .collect();
                let variance: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();
                let corrected_var: Vec<f64> = variance
                    .iter()
                    .zip(normalization.iter())
                    .map(|(&v, &n)| if n > 0.0 { v / (n * n) } else { 0.0 })
                    .collect();
                histogram::histogram_with_errors(
                    &radial,
                    &corrected,
                    &corrected_var,
                    n_points,
                    mask_data,
                )
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_with_errors(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_with_errors(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_with_errors(data, &lut1d, mask_data, &normalization[..])
            }
        }
    }

    /// Raw 1D integration returning intermediate sums (for multi-detector merging).
    /// Panics if method is Histogram.
    pub fn integrate1d_raw(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate1d_raw is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw(&frame.data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw(&frame.data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw(&frame.data, &lut1d, mask_data, &normalization[..])
            }
        }
    }

    /// Raw 1D integration with variance returning intermediate sums (for multi-detector merging).
    /// Panics if method is Histogram.
    pub fn integrate1d_raw_with_variance(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate1d_raw_with_variance is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw_with_variance(
                    &frame.data,
                    &lut1d,
                    mask_data,
                    &normalization[..],
                )
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw_with_variance(
                    &frame.data,
                    &lut1d,
                    mask_data,
                    &normalization[..],
                )
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw_with_variance(
                    &frame.data,
                    &lut1d,
                    mask_data,
                    &normalization[..],
                )
            }
        }
    }

    /// Raw 1D integration from slice returning intermediate sums (for multi-detector merging).
    /// Panics if method is Histogram.
    pub fn integrate1d_raw_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.update_geometry_arrays(dim1, dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate1d_raw_from_slice is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw(data, &lut1d, mask_data, &normalization[..])
            }
        }
    }

    /// Raw 1D integration with variance from slice (for multi-detector merging).
    /// Panics if method is Histogram.
    pub fn integrate1d_raw_with_variance_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.update_geometry_arrays(dim1, dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!(
                    "integrate1d_raw_with_variance_from_slice is not supported for Histogram method"
                )
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting {
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw_with_variance(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw_with_variance(data, &lut1d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range: self.radial_range_key(),
                });
                self.lut_integrate1d_raw_with_variance(data, &lut1d, mask_data, &normalization[..])
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
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    pub fn integrate2d(&self, frame: &Frame, n_radial: usize, n_azimuthal: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate2d is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                self.lut_integrate2d(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(n_radial, n_azimuthal, None));
                self.lut_integrate2d(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                self.lut_integrate2d(&frame.data, &lut2d, mask_data, &normalization[..])
            }
        }
    }

    /// Like [`Self::integrate2d`], for a row-major `dim1 x dim2` slice.
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    pub fn integrate2d_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> Vec<f64> {
        self.update_geometry_arrays(dim1, dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate2d_from_slice is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                self.lut_integrate2d(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(n_radial, n_azimuthal, None));
                self.lut_integrate2d(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                self.lut_integrate2d(data, &lut2d, mask_data, &normalization[..])
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
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    pub fn integrate2d_with_azimuthal_range(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate2d_with_azimuthal_range is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(
                    n_radial,
                    n_azimuthal,
                    Some(azimuthal),
                ));
                self.lut_integrate2d(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d(&frame.data, &lut2d, mask_data, &normalization[..])
            }
        }
    }

    /// Like [`Self::integrate2d_with_azimuthal_range`], for a row-major `dim1 x dim2` slice.
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    #[allow(clippy::too_many_arguments)]
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
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!(
                    "integrate2d_with_azimuthal_range_from_slice is not supported for Histogram method"
                )
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(
                    n_radial,
                    n_azimuthal,
                    Some(azimuthal),
                ));
                self.lut_integrate2d(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d(data, &lut2d, mask_data, &normalization[..])
            }
        }
    }

    /// Integrate a frame to 2D and return per-bin errors (standard deviations).
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    pub fn integrate2d_with_errors(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate2d_with_errors is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                self.lut_integrate2d_with_errors(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(n_radial, n_azimuthal, None));
                self.lut_integrate2d_with_errors(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                self.lut_integrate2d_with_errors(&frame.data, &lut2d, mask_data, &normalization[..])
            }
        }
    }

    /// Like [`Self::integrate2d_with_errors`], for a row-major `dim1 x dim2` slice.
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    pub fn integrate2d_with_errors_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(dim1, dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!("integrate2d_with_errors_from_slice is not supported for Histogram method")
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit,
                });
                self.lut_integrate2d_with_errors(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(n_radial, n_azimuthal, None));
                self.lut_integrate2d_with_errors(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                });
                self.lut_integrate2d_with_errors(data, &lut2d, mask_data, &normalization[..])
            }
        }
    }

    /// Integrate a frame to 2D with errors using an explicit azimuthal binning definition.
    ///
    /// `azimuthal_start` is the center of the first azimuthal bin; the effective start edge is
    /// `azimuthal_start - azimuthal_step / 2`. Azimuthal bins wrap periodically, so angles outside
    /// the total range fold into `0..n_azimuthal`.
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    pub fn integrate2d_with_azimuthal_range_and_errors(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!(
                    "integrate2d_with_azimuthal_range_and_errors is not supported for Histogram method"
                )
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d_with_errors(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(
                    n_radial,
                    n_azimuthal,
                    Some(azimuthal),
                ));
                self.lut_integrate2d_with_errors(&frame.data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d_with_errors(&frame.data, &lut2d, mask_data, &normalization[..])
            }
        }
    }

    /// Like [`Self::integrate2d_with_azimuthal_range_and_errors`], for a row-major
    /// `dim1 x dim2` slice.
    ///
    /// # Panics
    ///
    /// Panics if the method is `IntegrationMethod::Histogram`, which has no 2D binning.
    #[allow(clippy::too_many_arguments)]
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
        let normalization = self.get_normalization();
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        });
        let azimuthal = AzimuthalRangeKey::new(azimuthal_start, azimuthal_step);
        let unit = self.radial_unit;
        match self.method {
            IntegrationMethod::Histogram => {
                panic!(
                    "integrate2d_with_azimuthal_range_and_errors_from_slice is not supported for Histogram method"
                )
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d_with_errors(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                let lut2d = self.ensure_lut2d(self.pixel_split2d_key(
                    n_radial,
                    n_azimuthal,
                    Some(azimuthal),
                ));
                self.lut_integrate2d_with_errors(data, &lut2d, mask_data, &normalization[..])
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                });
                self.lut_integrate2d_with_errors(data, &lut2d, mask_data, &normalization[..])
            }
        }
    }
}
