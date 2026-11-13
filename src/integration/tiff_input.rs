//! Typed TIFF adapters. Arithmetic is shared with the existing f64 CSR path.
use super::cache::{AzimuthalRangeKey, OneDKey, TwoDKey};
use super::{IntegrationMethod, Integrator};
use crate::frame::TiffFrame;
use tiff::decoder::DecodingResult;

macro_rules! pixels {
    ($frame:expr, $data:ident, $body:expr) => {{
        match &$frame.pixels {
            DecodingResult::U8($data) => $body,
            DecodingResult::U16($data) => $body,
            DecodingResult::U32($data) => $body,
            DecodingResult::U64($data) => $body,
            DecodingResult::I8($data) => $body,
            DecodingResult::I16($data) => $body,
            DecodingResult::I32($data) => $body,
            DecodingResult::I64($data) => $body,
            DecodingResult::F32($data) => $body,
            DecodingResult::F64($data) => $body,
        }
    }};
}

impl Integrator {
    pub(crate) fn integrate_tiff1d<const ERRORS: bool>(
        &self,
        frame: &TiffFrame,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        if frame.legacy().is_some() || matches!(self.method, IntegrationMethod::Histogram) {
            let f = frame.to_frame();
            return if ERRORS {
                self.integrate1d_with_errors(&f, n_points)
            } else {
                let (x, y) = self.integrate1d(&f, n_points);
                (x, y, Vec::new())
            };
        }
        let (h, w) = (frame.dim1, frame.dim2);
        let view = self.for_shape(h, w);
        let norm = view.normalization();
        let mask = self.mask.as_ref().map(|m| {
            m.assert_compatible(h, w);
            m.data.as_slice()
        });
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
        let lut = view.ensure_lut1d(key);
        pixels!(
            frame,
            data,
            if self.fast_math {
                lut.integrate_pixels::<true, ERRORS, _>(data, mask, &norm)
            } else {
                lut.integrate_pixels::<false, ERRORS, _>(data, mask, &norm)
            }
        )
    }

    pub(crate) fn integrate_tiff2d<const ERRORS: bool>(
        &self,
        frame: &TiffFrame,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: Option<(f64, f64)>,
    ) -> (Vec<f64>, Vec<f64>) {
        if frame.legacy().is_some() || matches!(self.method, IntegrationMethod::Histogram) {
            let f = frame.to_frame();
            return match (ERRORS, azimuthal) {
                (true, None) => self.integrate2d_with_errors(&f, n_radial, n_azimuthal),
                (false, None) => (self.integrate2d(&f, n_radial, n_azimuthal), Vec::new()),
                (true, Some((a, b))) => self.integrate2d_with_azimuthal_range_and_errors(
                    &f,
                    n_radial,
                    n_azimuthal,
                    a,
                    b,
                ),
                (false, Some((a, b))) => (
                    self.integrate2d_with_azimuthal_range(&f, n_radial, n_azimuthal, a, b),
                    Vec::new(),
                ),
            };
        }
        let (h, w) = (frame.dim1, frame.dim2);
        let view = self.for_shape(h, w);
        let norm = view.normalization();
        let mask = self.mask.as_ref().map(|m| {
            m.assert_compatible(h, w);
            m.data.as_slice()
        });
        let unit = self.radial_unit;
        let key = match (self.method, azimuthal) {
            (IntegrationMethod::Counting, None) => TwoDKey::Counting {
                n_radial,
                n_azimuthal,
                unit,
            },
            (IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull, None) => {
                self.pixel_split2d_key(n_radial, n_azimuthal, None)
            }
            (IntegrationMethod::Supersampled { n_ss }, None) => TwoDKey::Supersampled {
                n_radial,
                n_azimuthal,
                n_ss,
                unit,
            },
            (IntegrationMethod::Counting, Some((a, b))) => TwoDKey::CountingAzimuthal {
                n_radial,
                n_azimuthal,
                azimuthal: AzimuthalRangeKey::new(a, b),
                unit,
            },
            (IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull, Some((a, b))) => {
                self.pixel_split2d_key(n_radial, n_azimuthal, Some(AzimuthalRangeKey::new(a, b)))
            }
            (IntegrationMethod::Supersampled { n_ss }, Some((a, b))) => {
                TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal: AzimuthalRangeKey::new(a, b),
                    unit,
                }
            }
            _ => unreachable!(),
        };
        let lut = view.ensure_lut2d(key);
        pixels!(
            frame,
            data,
            if self.fast_math {
                lut.integrate_pixels::<true, ERRORS, _>(data, mask, &norm)
            } else {
                lut.integrate_pixels::<false, ERRORS, _>(data, mask, &norm)
            }
        )
    }
}
