use std::fs;
use std::path::Path;

/// Binary pixel mask where non-zero values indicate masked pixels.
pub struct Mask {
    pub data: Vec<u8>,
    pub dim1: usize,
    pub dim2: usize,
}

impl Mask {
    pub fn from_tiff(file_path: String) -> Result<Mask, Box<dyn std::error::Error>> {
        let path = Path::new(&file_path);

        let f = fs::File::open(path)?;
        let mut decoder = tiff::decoder::Decoder::new(f)?;

        let dims = decoder.dimensions()?;
        // tiff library returns (width, height) but we want (height, width)
        let dim1 = dims.1 as usize;
        let dim2 = dims.0 as usize;

        let array: Vec<u8> = match decoder.read_image()? {
            tiff::decoder::DecodingResult::U8(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::U16(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::U32(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::U64(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::I8(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::I16(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::I32(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::I64(v) => v.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::F32(v) => v.iter().map(|&x| if x != 0.0 { 1 } else { 0 }).collect(),
            tiff::decoder::DecodingResult::F64(v) => v.iter().map(|&x| if x != 0.0 { 1 } else { 0 }).collect(),
        };

        let mask = Mask {
            data: flip_ud(&array, dim2, dim1),
            dim1,
            dim2,
        };
        Ok(mask)
    }

    pub fn from_vec(data: Vec<u8>, dim1: usize, dim2: usize) -> Result<Mask, String> {
        if data.len() != dim1 * dim2 {
            return Err("mask size does not match dimensions".to_string());
        }
        Ok(Mask { data, dim1, dim2 })
    }

    pub fn assert_compatible(&self, dim1: usize, dim2: usize) {
        assert_eq!(
            self.dim1, dim1,
            "mask dim1 mismatch: mask = {}, data = {}",
            self.dim1, dim1
        );
        assert_eq!(
            self.dim2, dim2,
            "mask dim2 mismatch: mask = {}, data = {}",
            self.dim2, dim2
        );
    }
}

/// Creates a flipped up-down version of the input array with dimensions dim1 and dim2
fn flip_ud(array: &Vec<u8>, dim1: usize, dim2: usize) -> Vec<u8> {
    let mut out_array = vec![0u8; dim1 * dim2];
    let mut out_i = 0;
    for chunk in array.chunks(dim1).rev() {
        for value in chunk {
            out_array[out_i] = *value;
            out_i += 1;
        }
    }
    out_array
}
