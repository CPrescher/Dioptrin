#![allow(dead_code)]
use std::fs;
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

        let f = fs::File::open(path)?;
        let mut decoder = tiff::decoder::Decoder::new(f)?;

        let dims = decoder.dimensions()?;
        // tiff library returns (width, height) but we want (height, width)
        let dim1 = dims.1 as usize;
        let dim2 = dims.0 as usize;

        let array: Vec<f64> = match decoder.read_image()? {
            tiff::decoder::DecodingResult::U8(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::U16(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::U32(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::U64(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::I8(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::I16(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::I32(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::I64(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::F32(v) => v.iter().map(|x| *x as f64).collect(),
            tiff::decoder::DecodingResult::F64(v) => v.iter().map(|x| *x as f64).collect(),
        };

        let frame = Frame {
            name: file_path,
            data: flip_ud(&array, dim2, dim1),
            dim1,
            dim2,
        };
        Ok(frame)
    }
}

/// Creates a flipped up-down version of the input array with dimensions dim1 and dim2
fn flip_ud(array: &Vec<f64>, dim1: usize, dim2: usize) -> Vec<f64> {
    let mut out_array = vec![0.0; dim1 * dim2];
    let mut out_i = 0;
    for chunk in array.chunks(dim1).rev() {
        for value in chunk {
            out_array[out_i] = *value;
            out_i += 1;
        }
    }
    out_array
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
