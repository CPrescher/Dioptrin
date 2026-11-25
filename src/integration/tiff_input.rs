//! Typed TIFF adapters. Arithmetic is shared with the existing f64 CSR path.
use super::Integrator;
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
        if frame.legacy().is_some() {
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
        let mask = self.checked_mask(h, w);
        let lut = view.ensure_lut1d(self.one_d_key(n_points));
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
    ) -> (Vec<f64>, Vec<f64>) {
        if frame.legacy().is_some() {
            let f = frame.to_frame();
            return if ERRORS {
                self.integrate2d_with_errors(&f, n_radial, n_azimuthal)
            } else {
                (self.integrate2d(&f, n_radial, n_azimuthal), Vec::new())
            };
        }
        let (h, w) = (frame.dim1, frame.dim2);
        let view = self.for_shape(h, w);
        let norm = view.normalization();
        let mask = self.checked_mask(h, w);
        let key = self.lut2d_key(&view, n_radial, n_azimuthal, "integrate2d");
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
