//! Benchmark-only bridge for a dioptrin-io checkout with caller-buffer decoding.
use dioptrin_io::hdf5::{
    Limits,
    raw::{FrameBuffer, RawReader},
};
use std::{
    ffi::{CStr, c_char},
    path::Path,
    ptr, slice,
};

pub struct Reader {
    source: RawReader,
    scratch: FrameBuffer,
}

/// # Safety
/// String pointers must be valid UTF-8 C strings, and `shape` points to three writable usizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eiger_open(
    path: *const c_char,
    dataset: *const c_char,
    shape: *mut usize,
) -> *mut Reader {
    let opened = (|| -> Result<_, Box<dyn std::error::Error>> {
        let path = unsafe { CStr::from_ptr(path) }.to_str()?;
        let dataset = unsafe { CStr::from_ptr(dataset) }.to_str()?;
        let source = RawReader::open_snapshot(Path::new(path), dataset, Limits::default())?;
        unsafe { ptr::copy_nonoverlapping(source.shape().as_ptr(), shape, 3) };
        Ok(Reader {
            source,
            scratch: FrameBuffer::default(),
        })
    })();
    match opened {
        Ok(reader) => Box::into_raw(Box::new(reader)),
        Err(error) => {
            eprintln!("dioptrin-io open: {error}");
            ptr::null_mut()
        }
    }
}

/// # Safety
/// `reader` is a live, exclusively borrowed handle. `output` points to `len`
/// writable, initialized, aligned u32s with no concurrent access.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eiger_read(
    reader: *mut Reader,
    frame: usize,
    output: *mut u32,
    len: usize,
) -> i32 {
    let reader = unsafe { &mut *reader };
    let output = unsafe { slice::from_raw_parts_mut(output, len) };
    match reader
        .source
        .read_frame_u32_into(frame, output, &mut reader.scratch)
    {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("dioptrin-io read {frame}: {error}");
            -1
        }
    }
}

/// # Safety
/// `reader` is a handle returned by `eiger_open`, closed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn eiger_close(reader: *mut Reader) {
    drop(unsafe { Box::from_raw(reader) });
}
