use crate::frame::Frame;
use crate::math::histogram;

use super::cache::{AzimuthalRangeKey, RadialRangeKey, TwoDKey};
use super::{IntegrationMethod, Integrator};

impl Integrator {
    pub(super) fn radial_range_key(&self) -> Option<RadialRangeKey> {
        self.radial_range
            .map(|(min, max)| RadialRangeKey::new(min, max))
    }

    #[inline]
    pub(super) fn checked_mask(&self, dim1: usize, dim2: usize) -> Option<&[u8]> {
        self.mask.as_ref().map(|mask| {
            mask.assert_compatible(dim1, dim2);
            mask.data.as_slice()
        })
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
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask = self.checked_mask(dim1, dim2);
        let variance = errors.then(|| data.iter().map(|&s| s.max(0.0)).collect::<Vec<_>>());
        let lut = view.ensure_lut1d(self.one_d_key(n_points));
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
    #[inline]
    pub fn integrate1d(&self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.integrate1d_from_slice(&frame.data, frame.dim1, frame.dim2, n_points)
    }

    /// Integrate a frame to 1D and return per-bin errors (standard deviations).
    #[inline]
    pub fn integrate1d_with_errors(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.integrate1d_with_errors_from_slice(&frame.data, frame.dim1, frame.dim2, n_points)
    }

    /// Returns whether `try_integrate1d_batch4` will take the 4-frame tiled
    /// path for `(dim1, dim2, n_points, errors)`.
    #[allow(dead_code)]
    pub(crate) fn can_integrate1d_batch4(
        &self,
        dim1: usize,
        dim2: usize,
        n_points: usize,
        errors: bool,
    ) -> bool {
        let view = self.for_shape(dim1, dim2);
        let lut = view.ensure_lut1d(self.one_d_key(n_points));
        lut.can_batch4(self.fast_math, errors)
    }

    /// 4-frame SIMD-tiled fixed-mask 1D integration on both fast CSC and CSR
    /// (fast intensity or strict) paths. Returns `None` for dynamic masks or
    /// CSR `fast_math && errors`, so callers fall back to per-frame integration.
    #[allow(dead_code, clippy::type_complexity)]
    pub(crate) fn try_integrate1d_batch4<P: crate::frame::Pixel>(
        &self,
        signals: [&[P]; 4],
        dim1: usize,
        dim2: usize,
        n_points: usize,
        errors: bool,
    ) -> Option<[(Vec<f64>, Vec<f64>, Option<Vec<f64>>); 4]> {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let lut = view.ensure_lut1d(self.one_d_key(n_points));
        let results = match (self.fast_math, errors) {
            (true, false) => {
                lut.integrate_batch4::<true, false, P>(signals, mask_data, &normalization)
            }
            (true, true) => {
                lut.integrate_batch4::<true, true, P>(signals, mask_data, &normalization)
            }
            (false, false) => {
                lut.integrate_batch4::<false, false, P>(signals, mask_data, &normalization)
            }
            (false, true) => {
                lut.integrate_batch4::<false, true, P>(signals, mask_data, &normalization)
            }
        }?;
        Some(results.map(|(r, y, e)| (r, y, errors.then_some(e))))
    }

    #[allow(dead_code)]
    pub(crate) fn integrate1d_pixels_from_slice<P: crate::frame::Pixel>(
        &self,
        data: &[P],
        dim1: usize,
        dim2: usize,
        n_points: usize,
        errors: bool,
    ) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
        assert_eq!(
            data.len(),
            dim1 * dim2,
            "data length ({}) does not match shape ({dim1}, {dim2})",
            data.len()
        );
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let lut = view.ensure_lut1d(self.one_d_key(n_points));
        let (r, y, e) = match (self.fast_math, errors) {
            (true, false) => {
                lut.integrate_pixels::<true, false, P>(data, mask_data, &normalization)
            }
            (true, true) => lut.integrate_pixels::<true, true, P>(data, mask_data, &normalization),
            (false, false) => {
                lut.integrate_pixels::<false, false, P>(data, mask_data, &normalization)
            }
            (false, true) => {
                lut.integrate_pixels::<false, true, P>(data, mask_data, &normalization)
            }
        };
        (r, y, errors.then_some(e))
    }

    pub fn integrate1d_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let lut1d = view.ensure_lut1d(self.one_d_key(n_points));
        self.lut_integrate1d(data, &lut1d, mask_data, &normalization[..])
    }

    pub fn integrate1d_with_errors_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let lut1d = view.ensure_lut1d(self.one_d_key(n_points));
        self.lut_integrate1d_with_errors(data, &lut1d, mask_data, &normalization[..])
    }

    /// Legacy unweighted 1D histogram integration of pre-corrected pixel intensities.
    ///
    /// Always bins over the automatic radial range of pixel centres (`radial_range`
    /// is ignored) and does not use a cached LUT.
    #[inline]
    pub fn integrate1d_histogram(&self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.integrate1d_histogram_from_slice(&frame.data, frame.dim1, frame.dim2, n_points)
    }

    /// Legacy unweighted 1D histogram integration from a flat slice of pre-corrected
    /// pixel intensities.
    pub fn integrate1d_histogram_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let radial = &view.state.radial;
        let corrected: Vec<f64> = data
            .iter()
            .zip(normalization.iter())
            .map(|(&s, &n)| if n > 0.0 { s / n } else { 0.0 })
            .collect();
        histogram::histogram(radial, &corrected, n_points, mask_data)
    }

    /// Legacy unweighted 1D histogram integration with Poisson error estimates.
    #[inline]
    pub fn integrate1d_histogram_with_errors(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.integrate1d_histogram_with_errors_from_slice(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_points,
        )
    }

    /// Legacy unweighted 1D histogram integration with Poisson error estimates from a slice.
    pub fn integrate1d_histogram_with_errors_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let radial = &view.state.radial;
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
        histogram::histogram_with_errors(radial, &corrected, &corrected_var, n_points, mask_data)
    }

    fn integrate1d_raw_impl(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
        with_variance: bool,
    ) -> crate::lut::IntermediateResult1d {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let lut1d = view.ensure_lut1d(self.one_d_key(n_points));
        if with_variance {
            self.lut_integrate1d_raw_with_variance(data, &lut1d, mask_data, &normalization[..])
        } else {
            self.lut_integrate1d_raw(data, &lut1d, mask_data, &normalization[..])
        }
    }

    /// Raw 1D integration returning intermediate sums (for multi-detector merging).
    #[inline]
    pub fn integrate1d_raw(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.integrate1d_raw_impl(&frame.data, frame.dim1, frame.dim2, n_points, false)
    }

    /// Raw 1D integration with variance returning intermediate sums (for multi-detector merging).
    #[inline]
    pub fn integrate1d_raw_with_variance(
        &self,
        frame: &Frame,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.integrate1d_raw_impl(&frame.data, frame.dim1, frame.dim2, n_points, true)
    }

    /// Raw 1D integration from slice returning intermediate sums (for multi-detector merging).
    #[inline]
    pub fn integrate1d_raw_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.integrate1d_raw_impl(data, dim1, dim2, n_points, false)
    }

    /// Raw 1D integration with variance from slice (for multi-detector merging).
    #[inline]
    pub fn integrate1d_raw_with_variance_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_points: usize,
    ) -> crate::lut::IntermediateResult1d {
        self.integrate1d_raw_impl(data, dim1, dim2, n_points, true)
    }

    /// Cache key of the 2D table for `integrate2d*` and
    /// `integrate2d_with_azimuthal_range*` (`azimuthal` = first-bin centre and
    /// step). Without an explicit radial range these are the default tables;
    /// with one, `Counting`, `PixelSplit` and `Supersampled` use the shared
    /// grid of [`Self::explicit_radial_grid_2d`] (`PyfaiFull` takes the range
    /// in its own key).
    pub(super) fn try_lut2d_key(
        &self,
        view: &super::cache::ShapeView<'_>,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: Option<AzimuthalRangeKey>,
    ) -> Result<TwoDKey, String> {
        let unit = self.radial_unit;
        Ok(match (self.method, self.radial_range, azimuthal) {
            (IntegrationMethod::PyfaiFull, _, _) | (IntegrationMethod::PixelSplit, None, _) => {
                self.pixel_split2d_key(n_radial, n_azimuthal, azimuthal)
            }
            (_, Some(range), _) => self.shared_grid_key(self.explicit_radial_grid_2d(
                view,
                n_radial,
                n_azimuthal,
                azimuthal,
                range,
            )?),
            (IntegrationMethod::Counting, None, None) => TwoDKey::Counting {
                n_radial,
                n_azimuthal,
                unit,
            },
            (IntegrationMethod::Counting, None, Some(azimuthal)) => TwoDKey::CountingAzimuthal {
                n_radial,
                n_azimuthal,
                azimuthal,
                unit,
            },
            (IntegrationMethod::Supersampled { n_ss }, None, None) => TwoDKey::Supersampled {
                n_radial,
                n_azimuthal,
                n_ss,
                unit,
            },
            (IntegrationMethod::Supersampled { n_ss }, None, Some(azimuthal)) => {
                TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                }
            }
        })
    }

    /// [`Self::try_lut2d_key`] for the panicking entry point `caller`.
    pub(super) fn lut2d_key(
        &self,
        view: &super::cache::ShapeView<'_>,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: Option<AzimuthalRangeKey>,
        caller: &str,
    ) -> TwoDKey {
        self.try_lut2d_key(view, n_radial, n_azimuthal, azimuthal)
            .unwrap_or_else(|e| panic!("{caller}: {e}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn integrate2d_impl(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: Option<AzimuthalRangeKey>,
        caller: &str,
    ) -> Vec<f64> {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let lut2d =
            view.ensure_lut2d(self.lut2d_key(&view, n_radial, n_azimuthal, azimuthal, caller));
        self.lut_integrate2d(data, &lut2d, mask_data, &normalization[..])
    }

    #[allow(clippy::too_many_arguments)]
    fn integrate2d_with_errors_impl(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: Option<AzimuthalRangeKey>,
        caller: &str,
    ) -> (Vec<f64>, Vec<f64>) {
        let view = self.for_shape(dim1, dim2);
        let normalization = view.normalization();
        let mask_data = self.checked_mask(dim1, dim2);
        let lut2d =
            view.ensure_lut2d(self.lut2d_key(&view, n_radial, n_azimuthal, azimuthal, caller));
        self.lut_integrate2d_with_errors(data, &lut2d, mask_data, &normalization[..])
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
    #[inline]
    pub fn integrate2d(&self, frame: &Frame, n_radial: usize, n_azimuthal: usize) -> Vec<f64> {
        self.integrate2d_impl(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_radial,
            n_azimuthal,
            None,
            "integrate2d",
        )
    }

    /// Like [`Self::integrate2d`], for a row-major `dim1 x dim2` slice.
    #[inline]
    pub fn integrate2d_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> Vec<f64> {
        self.integrate2d_impl(
            data,
            dim1,
            dim2,
            n_radial,
            n_azimuthal,
            None,
            "integrate2d_from_slice",
        )
    }

    /// Integrate a frame to 2D using an explicit azimuthal start and step.
    /// Thread-safe: can be called from multiple threads on a shared `Arc<Integrator>`.
    ///
    /// For a full-circle azimuthal range, use `azimuthal_step = 2*pi / n_azimuthal` and a
    /// `azimuthal_start` that defines the center of the first bin (e.g. `0.0` for 0..2*pi).
    /// Without an explicit `radial_range` on the integrator, the azimuthal bins of
    /// `Counting`, `PixelSplit` and `Supersampled` wrap periodically, so angles outside the
    /// total range fold into `0..n_azimuthal`.
    ///
    /// `IntegrationMethod::PyfaiFull`, and every method once an explicit `radial_range` is set,
    /// do not wrap: they treat the bins as the sector
    /// `[azimuthal_start - azimuthal_step/2, azimuthal_start + (n_azimuthal - 0.5)*azimuthal_step)`
    /// and drop angles outside it instead of folding them into the bins. A sector whose lower
    /// edge lies outside `[-2π, 2π]` is shifted by whole turns into that window.
    /// With `IntegrationMethod::PyfaiFull` or an explicit `radial_range`, panics if
    /// `n_azimuthal * azimuthal_step` exceeds 2π, if `azimuthal_step` is not finite and
    /// positive, or if `azimuthal_start` is not finite.
    #[inline]
    pub fn integrate2d_with_azimuthal_range(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> Vec<f64> {
        self.integrate2d_impl(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_radial,
            n_azimuthal,
            Some(AzimuthalRangeKey::new(azimuthal_start, azimuthal_step)),
            "integrate2d_with_azimuthal_range",
        )
    }

    /// Like [`Self::integrate2d_with_azimuthal_range`], for a row-major `dim1 x dim2` slice.
    ///
    /// `IntegrationMethod::PyfaiFull`, and every method once an explicit `radial_range` is set,
    /// do not wrap: they treat the bins as the sector
    /// `[azimuthal_start - azimuthal_step/2, azimuthal_start + (n_azimuthal - 0.5)*azimuthal_step)`
    /// and drop angles outside it instead of folding them into the bins. A sector whose lower
    /// edge lies outside `[-2π, 2π]` is shifted by whole turns into that window.
    /// With `IntegrationMethod::PyfaiFull` or an explicit `radial_range`, panics if
    /// `n_azimuthal * azimuthal_step` exceeds 2π, if `azimuthal_step` is not finite and
    /// positive, or if `azimuthal_start` is not finite.
    #[inline]
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
        self.integrate2d_impl(
            data,
            dim1,
            dim2,
            n_radial,
            n_azimuthal,
            Some(AzimuthalRangeKey::new(azimuthal_start, azimuthal_step)),
            "integrate2d_with_azimuthal_range_from_slice",
        )
    }

    /// Integrate a frame to 2D and return per-bin errors (standard deviations).
    #[inline]
    pub fn integrate2d_with_errors(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.integrate2d_with_errors_impl(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_radial,
            n_azimuthal,
            None,
            "integrate2d_with_errors",
        )
    }

    /// Like [`Self::integrate2d_with_errors`], for a row-major `dim1 x dim2` slice.
    #[inline]
    pub fn integrate2d_with_errors_from_slice(
        &self,
        data: &[f64],
        dim1: usize,
        dim2: usize,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.integrate2d_with_errors_impl(
            data,
            dim1,
            dim2,
            n_radial,
            n_azimuthal,
            None,
            "integrate2d_with_errors_from_slice",
        )
    }

    /// Integrate a frame to 2D with errors using an explicit azimuthal binning definition.
    ///
    /// `azimuthal_start` is the center of the first azimuthal bin; the effective start edge is
    /// `azimuthal_start - azimuthal_step / 2`. Without an explicit `radial_range` on the
    /// integrator, the azimuthal bins of `Counting`, `PixelSplit` and `Supersampled` wrap
    /// periodically, so angles outside the total range fold into `0..n_azimuthal`.
    ///
    /// `IntegrationMethod::PyfaiFull`, and every method once an explicit `radial_range` is set,
    /// do not wrap: they treat the bins as the sector
    /// `[azimuthal_start - azimuthal_step/2, azimuthal_start + (n_azimuthal - 0.5)*azimuthal_step)`
    /// and drop angles outside it instead of folding them into the bins. A sector whose lower
    /// edge lies outside `[-2π, 2π]` is shifted by whole turns into that window.
    /// With `IntegrationMethod::PyfaiFull` or an explicit `radial_range`, panics if
    /// `n_azimuthal * azimuthal_step` exceeds 2π, if `azimuthal_step` is not finite and
    /// positive, or if `azimuthal_start` is not finite.
    #[inline]
    pub fn integrate2d_with_azimuthal_range_and_errors(
        &self,
        frame: &Frame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_start: f64,
        azimuthal_step: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        self.integrate2d_with_errors_impl(
            &frame.data,
            frame.dim1,
            frame.dim2,
            n_radial,
            n_azimuthal,
            Some(AzimuthalRangeKey::new(azimuthal_start, azimuthal_step)),
            "integrate2d_with_azimuthal_range_and_errors",
        )
    }

    /// Like [`Self::integrate2d_with_azimuthal_range_and_errors`], for a row-major
    /// `dim1 x dim2` slice.
    ///
    /// `IntegrationMethod::PyfaiFull`, and every method once an explicit `radial_range` is set,
    /// do not wrap: they treat the bins as the sector
    /// `[azimuthal_start - azimuthal_step/2, azimuthal_start + (n_azimuthal - 0.5)*azimuthal_step)`
    /// and drop angles outside it instead of folding them into the bins. A sector whose lower
    /// edge lies outside `[-2π, 2π]` is shifted by whole turns into that window.
    /// With `IntegrationMethod::PyfaiFull` or an explicit `radial_range`, panics if
    /// `n_azimuthal * azimuthal_step` exceeds 2π, if `azimuthal_step` is not finite and
    /// positive, or if `azimuthal_start` is not finite.
    #[inline]
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
        self.integrate2d_with_errors_impl(
            data,
            dim1,
            dim2,
            n_radial,
            n_azimuthal,
            Some(AzimuthalRangeKey::new(azimuthal_start, azimuthal_step)),
            "integrate2d_with_azimuthal_range_and_errors_from_slice",
        )
    }
}
