#![allow(dead_code)]
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct Frame {
    pub name: String,
    pub data: Vec<f64>,
    pub dim1: usize,
    pub dim2: usize,
}

impl Frame {
    pub fn from_tiff(file_path: String) -> Result<Frame, Box<dyn std::error::Error>> {
        let path = Path::new(&file_path);

        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut decoder = tiff::decoder::Decoder::new(reader)?;

        let dims = decoder.dimensions()?;
        // tiff library returns (width, height) but we want (height, width)
        let dim1 = dims.1 as usize;
        let dim2 = dims.0 as usize;
        let size = dim1 * dim2;

        // Convert and flip in one pass to avoid double allocation
        let data = match decoder.read_image()? {
            tiff::decoder::DecodingResult::U8(v) => convert_and_flip(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::U16(v) => convert_and_flip(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::U32(v) => convert_and_flip(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::U64(v) => convert_and_flip_u64(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::I8(v) => convert_and_flip(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::I16(v) => convert_and_flip(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::I32(v) => convert_and_flip(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::I64(v) => convert_and_flip_i64(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::F32(v) => convert_and_flip(&v, dim2, dim1, size),
            tiff::decoder::DecodingResult::F64(v) => flip_ud_f64(&v, dim2, dim1, size),
        };

        let frame = Frame {
            name: file_path,
            data,
            dim1,
            dim2,
        };
        Ok(frame)
    }
}

/// Convert to f64 and flip up-down in a single pass (for types with Into<f64>)
#[inline]
fn convert_and_flip<T: Copy + Into<f64>>(
    array: &[T],
    row_len: usize,
    n_rows: usize,
    size: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(size);
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend(array[start..end].iter().map(|&x| x.into()));
    }
    out
}

/// Convert u64 to f64 and flip up-down (u64 doesn't impl Into<f64> due to precision loss)
#[inline]
fn convert_and_flip_u64(array: &[u64], row_len: usize, n_rows: usize, size: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(size);
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend(array[start..end].iter().map(|&x| x as f64));
    }
    out
}

/// Convert i64 to f64 and flip up-down (i64 doesn't impl Into<f64> due to precision loss)
#[inline]
fn convert_and_flip_i64(array: &[i64], row_len: usize, n_rows: usize, size: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(size);
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend(array[start..end].iter().map(|&x| x as f64));
    }
    out
}

/// Flip up-down for f64 data (no conversion needed)
#[inline]
fn flip_ud_f64(array: &[f64], row_len: usize, n_rows: usize, size: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(size);
    for row in (0..n_rows).rev() {
        let start = row * row_len;
        let end = start + row_len;
        out.extend_from_slice(&array[start..end]);
    }
    out
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
}
