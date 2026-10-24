//! Worker-owned decoding storage; the public f64 `Frame` API is unchanged.
use super::Frame;
use std::{fs::File, io::BufReader};
use tiff::decoder::{ChunkType, Decoder, DecodingResult, Limits};
use tiff::tags::Tag;

/// Reusable native TIFF input used by file workers and integration tasks.
#[doc(hidden)]
pub struct TiffFrame {
    pub(crate) pixels: DecodingResult,
    pub(crate) dim1: usize,
    pub(crate) dim2: usize,
    name: String,
    fallback: Option<Frame>,
}

impl Default for TiffFrame {
    fn default() -> Self {
        Self {
            pixels: DecodingResult::U8(Vec::new()),
            dim1: 0,
            dim2: 0,
            name: String::new(),
            fallback: None,
        }
    }
}

impl TiffFrame {
    pub fn reload(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Never expose partially decoded pixels after an error.
        self.dim1 = 0;
        self.dim2 = 0;
        self.fallback = None;
        let mut decoder = Decoder::new(BufReader::new(File::open(path)?))?;
        let (width, height) = decoder.dimensions()?;
        let tiff::ColorType::Gray(bits) = decoder.colortype()? else {
            // Preserve the legacy handling of non-detector/multichannel files.
            self.fallback = Some(Frame::from_tiff(path.to_owned())?);
            return Ok(());
        };
        let format = decoder
            .find_tag_unsigned_vec::<u16>(Tag::SampleFormat)?
            .and_then(|v| v.first().copied())
            .unwrap_or(1);
        let size = (width as usize)
            .checked_mul(height as usize)
            .ok_or(tiff::TiffError::IntSizeError)?;
        macro_rules! buffer {
            ($variant:ident, $ty:ty) => {{
                // Match Decoder::read_image's default allocation limit.
                if size > Limits::default().decoding_buffer_size / std::mem::size_of::<$ty>() {
                    return Err(tiff::TiffError::LimitsExceeded.into());
                }
                if !matches!(self.pixels, DecodingResult::$variant(_)) {
                    self.pixels = DecodingResult::$variant(Vec::new());
                }
                if let DecodingResult::$variant(v) = &mut self.pixels {
                    v.resize(size, 0 as $ty);
                }
            }};
        }
        match (format, bits) {
            (1, 0..=8) => buffer!(U8, u8),
            (1, 9..=16) => buffer!(U16, u16),
            (1, 17..=32) => buffer!(U32, u32),
            (1, 33..=64) => buffer!(U64, u64),
            (2, 0..=8) => buffer!(I8, i8),
            (2, 9..=16) => buffer!(I16, i16),
            (2, 17..=32) => buffer!(I32, i32),
            (2, 33..=64) => buffer!(I64, i64),
            (3, 32) => buffer!(F32, f32),
            (3, 64) => buffer!(F64, f64),
            _ => {
                self.fallback = Some(Frame::from_tiff(path.to_owned())?);
                return Ok(());
            }
        }
        if size != 0 {
            let (cw, ch) = decoder.chunk_dimensions();
            let (cw, ch) = (cw.min(width) as usize, ch.min(height) as usize);
            if cw == 0 || ch == 0 {
                return Err("invalid TIFF chunk dimensions".into());
            }
            let across = (width as usize).div_ceil(cw);
            let count = match decoder.get_chunk_type() {
                ChunkType::Strip => decoder.strip_count()?,
                ChunkType::Tile => decoder.tile_count()?,
            };
            // Same chunk placement as Decoder::read_image, including padded edge tiles.
            for chunk in 0..count {
                let k = chunk as usize;
                let offset = (k / across) * width as usize * ch + (k % across) * cw;
                if offset >= size {
                    return Err("invalid TIFF chunk offset".into());
                }
                decoder.read_chunk_to_buffer(
                    self.pixels.as_buffer(offset),
                    chunk,
                    width as usize,
                )?;
            }
        }
        // Flip in the compact typed buffer. CSR sees exactly the legacy row order.
        macro_rules! flip {
            ($($variant:ident),*) => { match &mut self.pixels { $(DecodingResult::$variant(v) => {
                for row in 0..height as usize / 2 {
                    let (a,b) = v.split_at_mut((height as usize - 1 - row) * width as usize);
                    a[row * width as usize..(row+1) * width as usize].swap_with_slice(&mut b[..width as usize]);
                }
            })* }};
        }
        flip!(U8, U16, U32, U64, I8, I16, I32, I64, F32, F64);
        self.dim1 = height as usize;
        self.dim2 = width as usize;
        self.name.clear();
        self.name.push_str(path);
        Ok(())
    }

    pub(crate) fn legacy(&self) -> Option<&Frame> {
        self.fallback.as_ref()
    }

    /// Compatibility path for custom tasks and statistics that consume f64 frames.
    pub fn to_frame(&self) -> Frame {
        if let Some(f) = &self.fallback {
            return Frame {
                name: f.name.clone(),
                data: f.data.clone(),
                dim1: f.dim1,
                dim2: f.dim2,
            };
        }
        macro_rules! convert {
            ($($variant:ident),*) => { match &self.pixels { $(DecodingResult::$variant(v) => v.iter().map(|&x| x as f64).collect(),)* }};
        }
        Frame {
            name: self.name.clone(),
            data: convert!(U8, U16, U32, U64, I8, I16, I32, I64, F32, F64),
            dim1: self.dim1,
            dim2: self.dim2,
        }
    }
}

#[cfg(test)]
#[path = "tiff_tests.rs"]
mod tests;
