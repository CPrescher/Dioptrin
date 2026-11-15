#![allow(dead_code)]
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::{PhotometricInterpretation, Tag};
use tiff::{ColorType, TiffError, TiffUnsupportedError};

mod tiff_input;
#[doc(hidden)]
pub use self::tiff_input::TiffFrame;

/// Conversion performed before the existing f64 arithmetic in CSR kernels.
pub(crate) trait Pixel: Copy + Send + Sync {
    fn to_f64(self) -> f64;
}
macro_rules! pixels {
    ($($ty:ty),*) => {$(impl Pixel for $ty {
        #[inline(always)]
        fn to_f64(self) -> f64 { self as f64 }
    })*};
}
pixels!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

pub struct Frame {
    pub name: String,
    pub data: Vec<f64>,
    pub dim1: usize,
    pub dim2: usize,
}

impl Frame {
    pub fn from_tiff(file_path: String) -> Result<Frame, Box<dyn std::error::Error>> {
        let mut frame = Frame {
            name: String::new(),
            data: Vec::new(),
            dim1: 0,
            dim2: 0,
        };
        frame.reload_from_tiff(file_path)?;
        Ok(frame)
    }

    /// Decode a TIFF into this frame in place, reusing the existing `data`
    /// buffer when possible.
    ///
    /// `data.clear()` keeps the heap allocation, and `reserve(size)` only
    /// requests more capacity when the existing buffer is too small. For
    /// batch workloads with same-shape frames this avoids a fresh 80 MB
    /// allocation (and the page-faulting / zeroing that comes with it) on
    /// every call.
    pub fn reload_from_tiff(
        &mut self,
        file_path: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new(&file_path);
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut decoder = open_tiff(reader)?;
        require_single_channel(&mut decoder)?;

        let dims = decoder.dimensions()?;
        // tiff library returns (width, height) but we want (height, width)
        let dim1 = dims.1 as usize;
        let dim2 = dims.0 as usize;
        let size = dim1
            .checked_mul(dim2)
            .ok_or(tiff::TiffError::IntSizeError)?;

        let image = decoder.read_image()?;
        check_decoded_len(&image, dim1, dim2)?;

        self.data.clear();
        self.data.reserve(size);

        match image {
            DecodingResult::U8(v) => convert_and_flip_into(&v, dim2, dim1, &mut self.data),
            DecodingResult::U16(v) => convert_and_flip_into(&v, dim2, dim1, &mut self.data),
            DecodingResult::U32(v) => convert_and_flip_into(&v, dim2, dim1, &mut self.data),
            DecodingResult::U64(v) => convert_and_flip_into_u64(&v, dim2, dim1, &mut self.data),
            DecodingResult::I8(v) => convert_and_flip_into(&v, dim2, dim1, &mut self.data),
            DecodingResult::I16(v) => convert_and_flip_into(&v, dim2, dim1, &mut self.data),
            DecodingResult::I32(v) => convert_and_flip_into(&v, dim2, dim1, &mut self.data),
            DecodingResult::I64(v) => convert_and_flip_into_i64(&v, dim2, dim1, &mut self.data),
            DecodingResult::F32(v) => convert_and_flip_into(&v, dim2, dim1, &mut self.data),
            DecodingResult::F64(v) => flip_ud_into_f64(&v, dim2, dim1, &mut self.data),
        }

        self.dim1 = dim1;
        self.dim2 = dim2;
        self.name = file_path;
        Ok(())
    }
}

const SINGLE_CHANNEL_ONLY: &str = "only single-channel (grayscale) TIFFs are supported";

/// Open a TIFF decoder. A file whose sample count the decoder cannot handle
/// at all (2 or 5+ samples per pixel) fails here, so its error gets the same
/// "single-channel only" explanation as the colour files rejected by
/// [`require_single_channel`].
pub(crate) fn open_tiff<R: Read + Seek>(
    reader: R,
) -> Result<Decoder<R>, Box<dyn std::error::Error>> {
    Decoder::new(reader).map_err(|err| match err {
        TiffError::UnsupportedError(TiffUnsupportedError::UnsupportedSampleDepth(samples)) => {
            format!("unsupported TIFF: {samples} samples per pixel; {SINGLE_CHANNEL_ONLY}").into()
        }
        err => err.into(),
    })
}

/// Reject any TIFF image that does not hold exactly one sample per pixel.
///
/// Diffraction detectors write single-channel images. A colour TIFF (RGB,
/// RGBA, CMYK, palette, planar colour) is almost certainly the wrong file,
/// and the decoder returns its samples interleaved, so it is rejected rather
/// than reduced to one channel. The error names the photometric
/// interpretation, the samples per pixel and the image shape (rows, columns).
pub(crate) fn require_single_channel<R: Read + Seek>(
    decoder: &mut Decoder<R>,
) -> Result<(), Box<dyn std::error::Error>> {
    // `ColorType::Gray` means BlackIsZero/WhiteIsZero with one BitsPerSample
    // entry. The tiff crate (0.8) also reports it for SamplesPerPixel 3 or 4
    // with a single BitsPerSample value (accepted by libtiff and tifffile) and
    // then decodes only the first rows x cols interleaved samples, so the
    // SamplesPerPixel tag must say 1 as well.
    let samples_tag = decoder
        .find_tag_unsigned::<u16>(Tag::SamplesPerPixel)
        .ok()
        .flatten();
    if let Ok(ColorType::Gray(_)) = decoder.colortype()
        && samples_tag.unwrap_or(1) == 1
    {
        return Ok(());
    }
    let photometric = match decoder.find_tag_unsigned::<u16>(Tag::PhotometricInterpretation) {
        Ok(Some(value)) => match PhotometricInterpretation::from_u16(value) {
            Some(p) => format!("{p:?}"),
            None => format!("unknown ({value})"),
        },
        _ => "missing".to_owned(),
    };
    let bits = decoder
        .find_tag_unsigned_vec::<u16>(Tag::BitsPerSample)
        .ok()
        .flatten()
        .unwrap_or_default();
    // Without the tag, the number of BitsPerSample entries is the best guess.
    let samples = samples_tag.unwrap_or(bits.len().max(1) as u16);
    let (width, height) = decoder.dimensions()?;
    let plural = if samples == 1 { "" } else { "s" };
    Err(format!(
        "unsupported TIFF: photometric interpretation {photometric}, {samples} sample{plural} \
         per pixel (bits per sample {bits:?}), image shape ({height}, {width}); \
         {SINGLE_CHANNEL_ONLY}"
    )
    .into())
}

/// Number of samples in a decoded TIFF buffer.
fn decoded_len(image: &DecodingResult) -> usize {
    match image {
        DecodingResult::U8(v) => v.len(),
        DecodingResult::U16(v) => v.len(),
        DecodingResult::U32(v) => v.len(),
        DecodingResult::U64(v) => v.len(),
        DecodingResult::I8(v) => v.len(),
        DecodingResult::I16(v) => v.len(),
        DecodingResult::I32(v) => v.len(),
        DecodingResult::I64(v) => v.len(),
        DecodingResult::F32(v) => v.len(),
        DecodingResult::F64(v) => v.len(),
    }
}

/// Invariant behind [`require_single_channel`]: the decoded buffer holds
/// exactly one sample per pixel. Catches any kind the colour check misses
/// with an error instead of reading a wrong slice or panicking.
pub(crate) fn check_decoded_len(
    image: &DecodingResult,
    dim1: usize,
    dim2: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let len = decoded_len(image);
    if Some(len) == dim1.checked_mul(dim2) {
        return Ok(());
    }
    Err(format!(
        "unsupported TIFF: decoded {len} samples for an image of shape ({dim1}, {dim2}); \
         {SINGLE_CHANNEL_ONLY}"
    )
    .into())
}

/// Convert to f64 and flip up-down in a single pass into a caller-supplied buffer.
#[inline]
fn convert_and_flip_into<T: Copy + Into<f64>>(
    array: &[T],
    row_len: usize,
    n_rows: usize,
    out: &mut Vec<f64>,
) {
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend(array[start..end].iter().map(|&x| x.into()));
    }
}

/// u64 doesn't impl `Into<f64>` due to precision loss — use `as f64`.
#[inline]
fn convert_and_flip_into_u64(array: &[u64], row_len: usize, n_rows: usize, out: &mut Vec<f64>) {
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend(array[start..end].iter().map(|&x| x as f64));
    }
}

/// i64 doesn't impl `Into<f64>` due to precision loss — use `as f64`.
#[inline]
fn convert_and_flip_into_i64(array: &[i64], row_len: usize, n_rows: usize, out: &mut Vec<f64>) {
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend(array[start..end].iter().map(|&x| x as f64));
    }
}

/// Flip up-down for f64 data (no conversion needed).
#[inline]
fn flip_ud_into_f64(array: &[f64], row_len: usize, n_rows: usize, out: &mut Vec<f64>) {
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend_from_slice(&array[start..end]);
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::util;

    #[test]
    fn frame_from_tiff() {
        util::download_pilatus_test_file();
        let file_path = String::from("test_data/CeO2_Pilatus1M.tif");
        let frame = Frame::from_tiff(file_path).unwrap();
        assert_eq!(frame.dim1, 1043);
        assert_eq!(frame.dim2, 981);

        util::save_vec("frame.txt", &frame.data);
    }

    #[test]
    fn frame_reload_reuses_buffer() {
        util::download_pilatus_test_file();
        let path = String::from("test_data/CeO2_Pilatus1M.tif");

        let mut frame = Frame::from_tiff(path.clone()).unwrap();
        let cap_before = frame.data.capacity();
        let ptr_before = frame.data.as_ptr();

        frame.reload_from_tiff(path).unwrap();
        // Same shape file → buffer should be reused (capacity & pointer unchanged).
        assert_eq!(frame.data.capacity(), cap_before);
        assert_eq!(frame.data.as_ptr(), ptr_before);
    }
}
