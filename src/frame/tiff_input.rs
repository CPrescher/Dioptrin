//! Worker-owned decoding storage; the public f64 `Frame` API is unchanged.
use super::Frame;
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
};
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
        self.reload_from_reader(path, || Ok(BufReader::new(File::open(path)?)))
    }

    pub(crate) fn reload_from_reader<R: Read + Seek>(
        &mut self,
        path: &str,
        reader: impl FnOnce() -> Result<R, Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Never expose partially decoded pixels after an error.
        self.dim1 = 0;
        self.dim2 = 0;
        self.fallback = None;
        let mut decoder = super::open_tiff(reader()?)?;
        // Colour and other multi-sample files fail here, before either path.
        super::require_single_channel(&mut decoder)?;
        let (width, height) = decoder.dimensions()?;
        let tiff::ColorType::Gray(bits) = decoder.colortype()? else {
            // require_single_channel accepts only ColorType::Gray.
            return Err("unsupported TIFF: not a grayscale image".into());
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
            _ => return self.load_fallback(path),
        }
        read_chunks(
            &mut decoder,
            width,
            height,
            size,
            |decoder, offset, chunk| {
                decoder.read_chunk_to_buffer(self.pixels.as_buffer(offset), chunk, width as usize)
            },
        )?;
        // Flip in the compact typed buffer. CSR sees exactly the legacy row order.
        macro_rules! flip {
            ($($variant:ident),*) => { match &mut self.pixels { $(DecodingResult::$variant(v) => {
                flip_rows(v, width as usize, height as usize);
            })* }};
        }
        flip!(U8, U16, U32, U64, I8, I16, I32, I64, F32, F64);
        self.dim1 = height as usize;
        self.dim2 = width as usize;
        self.name.clear();
        self.name.push_str(path);
        Ok(())
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn read_i32_into<R: Read + Seek>(
        reader: R,
        shape: (usize, usize),
        dst: &mut [i32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use tiff::decoder::DecodingBuffer;
        let mut decoder = super::open_tiff(reader)?;
        super::require_single_channel(&mut decoder)?;
        let (width, height) = decoder.dimensions()?;
        let sample_format = decoder.find_tag_unsigned_vec::<u16>(Tag::SampleFormat)?;
        if shape != (height as usize, width as usize)
            || shape.0.checked_mul(shape.1) != Some(dst.len())
            || decoder.colortype()? != tiff::ColorType::Gray(32)
            || sample_format.as_deref() != Some(&[2])
        {
            return Err("direct TIFF loading requires a matching signed 32-bit image".into());
        }
        read_chunks(
            &mut decoder,
            width,
            height,
            dst.len(),
            |decoder, offset, chunk| {
                decoder.read_chunk_to_buffer(
                    DecodingBuffer::I32(&mut dst[offset..]),
                    chunk,
                    width as usize,
                )
            },
        )?;
        flip_rows(dst, shape.1, shape.0);
        Ok(())
    }

    /// Load `path` through the f64 `Frame` reader, for single-channel TIFFs
    /// whose sample format and bit depth the typed path does not handle.
    /// With tiff 0.8 every such file also fails in the `Frame` reader, so
    /// this returns the decoder's error; colour files are rejected before
    /// this. On success `dim1`/`dim2` are set to that frame's shape, so shape
    /// checks see the real image and not 0 x 0.
    fn load_fallback(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let frame = Frame::from_tiff(path.to_owned())?;
        self.dim1 = frame.dim1;
        self.dim2 = frame.dim2;
        self.name.clear();
        self.name.push_str(path);
        self.fallback = Some(frame);
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

fn read_chunks<R: Read + Seek>(
    decoder: &mut Decoder<R>,
    width: u32,
    height: u32,
    size: usize,
    mut read: impl FnMut(&mut Decoder<R>, usize, u32) -> tiff::TiffResult<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    if size == 0 {
        return Ok(());
    }
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
        read(decoder, offset, chunk)?;
    }
    Ok(())
}

fn flip_rows<T>(pixels: &mut [T], width: usize, height: usize) {
    for row in 0..height / 2 {
        let (top, bottom) = pixels.split_at_mut((height - 1 - row) * width);
        top[row * width..(row + 1) * width].swap_with_slice(&mut bottom[..width]);
    }
}

#[cfg(test)]
#[path = "tiff_tests.rs"]
mod tests;
