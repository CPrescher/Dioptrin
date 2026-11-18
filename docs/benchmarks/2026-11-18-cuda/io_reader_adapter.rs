use super::Result;
use dioptrin_io::hdf5::{
    Limits,
    raw::{FrameBuffer, RawReader},
};
use std::path::Path;

pub(super) struct LibraryApi;

impl LibraryApi {
    pub(super) fn load(_: &Path) -> Result<Self> {
        Ok(Self)
    }

    pub(super) fn dataset(&self, path: &str, name: &str) -> Result<Dataset> {
        let reader = RawReader::open_snapshot(path, name, Limits::default())?;
        Ok(Dataset {
            dims: reader.shape(),
            reader,
            scratch: FrameBuffer::default(),
        })
    }
}

pub(super) struct Dataset {
    reader: RawReader,
    scratch: FrameBuffer,
    pub(super) dims: [usize; 3],
}

impl Dataset {
    pub(super) fn read(&mut self, first: usize, destination: &mut [u32]) -> Result<()> {
        let pixels = self.dims[1] * self.dims[2];
        if destination.is_empty() || !destination.len().is_multiple_of(pixels) {
            return Err("expected whole frames".into());
        }
        for (i, frame) in destination.chunks_exact_mut(pixels).enumerate() {
            self.reader
                .read_frame_u32_into(first + i, frame, &mut self.scratch)?;
        }
        Ok(())
    }
}
