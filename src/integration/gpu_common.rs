//! Shared host-side frame-dispatch helpers for the CUDA and Metal backends.

/// Borrowed frames of one GPU-supported pixel type (`u16`, `u32`, `i32`, `f32`, `f64`).
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
#[derive(Clone, Copy)]
pub enum Frames<'a, 'b> {
    U16(&'a [&'b [u16]]),
    U32(&'a [&'b [u32]]),
    I32(&'a [&'b [i32]]),
    F32(&'a [&'b [f32]]),
    F64(&'a [&'b [f64]]),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Frames<'_, '_> {
    pub(crate) fn count(self, pixels: usize, stacked: bool) -> Result<usize, String> {
        fn check<T>(slices: &[&[T]], pixels: usize, stacked: bool) -> Result<usize, String> {
            if stacked {
                let len = slices[0].len();
                if !len.is_multiple_of(pixels) {
                    return Err(format!(
                        "stacked frame length {len} is not a multiple of plan pixel count {pixels}"
                    ));
                }
                Ok(len / pixels)
            } else {
                for (idx, frame) in slices.iter().enumerate() {
                    if frame.len() != pixels {
                        return Err(format!(
                            "frame {idx} has {} pixels, expected {pixels}",
                            frame.len()
                        ));
                    }
                }
                Ok(slices.len())
            }
        }
        match self {
            Self::U16(s) => check(s, pixels, stacked),
            Self::U32(s) => check(s, pixels, stacked),
            Self::I32(s) => check(s, pixels, stacked),
            Self::F32(s) => check(s, pixels, stacked),
            Self::F64(s) => check(s, pixels, stacked),
        }
    }

    pub(crate) fn contiguous_bytes(
        self,
        offset_frames: usize,
        count_frames: usize,
        pixels: usize,
    ) -> Option<(*const u8, usize)> {
        fn slice_bytes<T>(
            slice: &[T],
            offset_frames: usize,
            count_frames: usize,
            pixels: usize,
        ) -> (*const u8, usize) {
            let start = offset_frames * pixels;
            let len = count_frames * pixels;
            (
                slice[start..start + len].as_ptr().cast::<u8>(),
                len * size_of::<T>(),
            )
        }
        match self {
            Self::U16(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
            Self::U32(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
            Self::I32(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
            Self::F32(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
            Self::F64(_) => None,
        }
    }
}

pub trait Sealed {
    fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b>
    where
        Self: Sized;
}

impl Sealed for u16 {
    fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
        Frames::U16(frames)
    }
}

impl Sealed for u32 {
    fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
        Frames::U32(frames)
    }
}

impl Sealed for i32 {
    fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
        Frames::I32(frames)
    }
}

impl Sealed for f32 {
    fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
        Frames::F32(frames)
    }
}

impl Sealed for f64 {
    fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
        Frames::F64(frames)
    }
}
