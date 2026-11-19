use super::Result;
use dioptrin_io::hdf5::{
    Limits,
    raw::{FrameBuffer, RawReader},
};
use rayon::prelude::*;
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

pub(super) struct ParallelDecoder {
    pool: rayon::ThreadPool,
    scratch: Vec<FrameBuffer>,
}

impl ParallelDecoder {
    pub(super) fn new(threads: usize) -> Result<Self> {
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?,
            scratch: (0..threads).map(|_| FrameBuffer::default()).collect(),
        })
    }

    pub(super) fn read(
        &mut self,
        source: &Dataset,
        first: usize,
        output: &mut [u32],
    ) -> Result<()> {
        let pixels = source.dims[1] * source.dims[2];
        if output.is_empty() || !output.len().is_multiple_of(pixels) {
            return Err("expected whole frames".into());
        }
        let frames_per_worker = (output.len() / pixels).div_ceil(self.scratch.len());
        self.pool.install(|| {
            output
                .par_chunks_mut(frames_per_worker * pixels)
                .zip(&mut self.scratch)
                .enumerate()
                .try_for_each(|(shard, (out, scratch))| {
                    for (i, frame) in out.chunks_exact_mut(pixels).enumerate() {
                        source.reader.read_frame_u32_into(
                            first + shard * frames_per_worker + i,
                            frame,
                            scratch,
                        )?;
                    }
                    Ok::<_, dioptrin_io::hdf5::Error>(())
                })
        })?;
        Ok(())
    }
}
