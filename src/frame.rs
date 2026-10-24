#![allow(dead_code)]
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

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
        let mut decoder = tiff::decoder::Decoder::new(reader)?;

        let dims = decoder.dimensions()?;
        // tiff library returns (width, height) but we want (height, width)
        let dim1 = dims.1 as usize;
        let dim2 = dims.0 as usize;
        let size = dim1 * dim2;

        self.data.clear();
        self.data.reserve(size);

        match decoder.read_image()? {
            tiff::decoder::DecodingResult::U8(v) => {
                convert_and_flip_into(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::U16(v) => {
                convert_and_flip_into(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::U32(v) => {
                convert_and_flip_into(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::U64(v) => {
                convert_and_flip_into_u64(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::I8(v) => {
                convert_and_flip_into(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::I16(v) => {
                convert_and_flip_into(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::I32(v) => {
                convert_and_flip_into(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::I64(v) => {
                convert_and_flip_into_i64(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::F32(v) => {
                convert_and_flip_into(&v, dim2, dim1, &mut self.data)
            }
            tiff::decoder::DecodingResult::F64(v) => {
                flip_ud_into_f64(&v, dim2, dim1, &mut self.data)
            }
        }

        self.dim1 = dim1;
        self.dim2 = dim2;
        self.name = file_path;
        Ok(())
    }
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

/// u64 doesn't impl Into<f64> due to precision loss — use `as f64`.
#[inline]
fn convert_and_flip_into_u64(array: &[u64], row_len: usize, n_rows: usize, out: &mut Vec<f64>) {
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend(array[start..end].iter().map(|&x| x as f64));
    }
}

/// i64 doesn't impl Into<f64> due to precision loss — use `as f64`.
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
