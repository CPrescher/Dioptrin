use super::*;
use crate::frame::TiffFrame;
use cudarc::driver::CudaEvent;
use tiff::decoder::DecodingResult;

const FRAMES: usize = 64;
const TILE: usize = 8;
const PATTERN: usize = BINS as usize * 2;

struct Reader {
    frame: TiffFrame,
    #[cfg(windows)]
    direct: Option<Vec<u8>>,
}

impl Reader {
    fn new() -> Self {
        Self {
            frame: TiffFrame::default(),
            #[cfg(windows)]
            direct: env::var_os("DIOPTRIN_CUDA_DISK_DIRECT_IO").map(|_| Vec::new()),
        }
    }

    fn reload(&mut self, path: &str) -> Result<(&TiffFrame, f64)> {
        #[cfg(windows)]
        if let Some(buffer) = &mut self.direct {
            let mut read_ms = 0.0;
            self.frame.reload_from_reader(path, || {
                let start = Instant::now();
                let read = read_unbuffered(path, buffer);
                read_ms = elapsed_ms(start);
                read
            })?;
            return Ok((&self.frame, read_ms));
        }
        self.frame.reload(path)?;
        Ok((&self.frame, 0.0))
    }
}

#[cfg(windows)]
fn read_unbuffered<'a>(path: &str, storage: &'a mut Vec<u8>) -> Result<std::io::Cursor<&'a [u8]>> {
    use std::{
        fs::OpenOptions,
        io::{Cursor, Read},
        os::windows::fs::OpenOptionsExt,
    };

    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const ALIGNMENT: usize = 64 * 1024;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open(path)?;
    let bytes = usize::try_from(file.metadata()?.len())?;
    if bytes == 0 || bytes > 256 * 1024 * 1024 {
        return Err("unbuffered benchmark expects a TIFF of up to 256 MiB".into());
    }
    let aligned_len = bytes.next_multiple_of(ALIGNMENT);
    storage.resize(aligned_len + ALIGNMENT - 1, 0);
    let offset = storage.as_ptr().align_offset(ALIGNMENT);
    let slice = &mut storage[offset..offset + aligned_len];
    // One aligned read can extend to the next sector beyond EOF. Windows
    // returns the bytes that precede EOF, which form the TIFF for the decoder.
    if file.read(slice)? != bytes {
        return Err(format!("short unbuffered read for {path}").into());
    }
    Ok(Cursor::new(&slice[..bytes]))
}

#[derive(Default, Serialize)]
struct Phases {
    read_decode_worker_ms: f64,
    direct_read_worker_ms: f64,
    widen_worker_ms: f64,
    cpu_integrate_worker_ms: f64,
    gpu_loader_wall_ms: f64,
    gpu_upload_ms: f64,
    gpu_kernel_ms: f64,
    gpu_download_ms: f64,
}

struct Loaded {
    read_decode: f64,
    direct_read: f64,
    stage: f64,
}

type Load<I> = fn(&mut Reader, &str, (usize, usize), &mut [I]) -> Result<Loaded>;

struct Loader<'a, I> {
    shape: (usize, usize),
    pool: &'a ThreadPool,
    load: Load<I>,
}

fn decode_copied<I: From<i32>>(
    reader: &mut Reader,
    path: &str,
    shape: (usize, usize),
    dst: &mut [I],
) -> Result<Loaded> {
    let start = Instant::now();
    let (frame, direct_read) = reader.reload(path)?;
    let read_decode = elapsed_ms(start);
    if (frame.dim1, frame.dim2) != shape {
        return Err(format!("unexpected frame shape for {path}").into());
    }
    let DecodingResult::I32(pixels) = &frame.pixels else {
        return Err(format!("expected the signed 32-bit Eiger TIFF at {path}").into());
    };
    let start = Instant::now();
    for (d, &s) in dst.iter_mut().zip(pixels) {
        *d = I::from(s);
    }
    Ok(Loaded {
        read_decode,
        direct_read,
        stage: elapsed_ms(start),
    })
}

fn decode_native(
    reader: &mut Reader,
    path: &str,
    shape: (usize, usize),
    dst: &mut [i32],
) -> Result<Loaded> {
    let start = Instant::now();
    #[cfg(windows)]
    if let Some(buffer) = &mut reader.direct {
        let source = read_unbuffered(path, buffer)?;
        let direct_read = elapsed_ms(start);
        TiffFrame::read_i32_into(source, shape, dst)?;
        return Ok(Loaded {
            read_decode: elapsed_ms(start),
            direct_read,
            stage: 0.,
        });
    }
    let _ = reader;
    TiffFrame::read_i32_into(std::io::BufReader::new(File::open(path)?), shape, dst)?;
    Ok(Loaded {
        read_decode: elapsed_ms(start),
        direct_read: 0.,
        stage: 0.,
    })
}

fn cpu_files(
    integrator: &Integrator,
    paths: &[String],
    pool: &ThreadPool,
    frames: &mut [Reader],
) -> Result<(Vec<f64>, Phases)> {
    let workers = frames.len();
    let chunks = pool.install(|| {
        frames
            .par_iter_mut()
            .enumerate()
            .map(|(i, reader)| {
                let mut phases = Phases::default();
                let mut values = Vec::new();
                for path in &paths[i * paths.len() / workers..(i + 1) * paths.len() / workers] {
                    let start = Instant::now();
                    let (frame, read_ms) = reader.reload(path).map_err(|err| err.to_string())?;
                    phases.read_decode_worker_ms += elapsed_ms(start);
                    phases.direct_read_worker_ms += read_ms;
                    let start = Instant::now();
                    integrator.check_frame_shape(frame.dim1, frame.dim2)?;
                    let (_, intensity, sigma) =
                        integrator.integrate_tiff1d::<true>(frame, BINS as usize);
                    phases.cpu_integrate_worker_ms += elapsed_ms(start);
                    values.extend(intensity.into_iter().zip(sigma).flat_map(|(i, s)| [i, s]));
                }
                Ok::<_, String>((values, phases))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
    })?;
    let mut output = Vec::with_capacity(paths.len() * PATTERN);
    let mut phases = Phases::default();
    for (values, chunk) in chunks {
        output.extend(values);
        phases.read_decode_worker_ms += chunk.read_decode_worker_ms;
        phases.direct_read_worker_ms += chunk.direct_read_worker_ms;
        phases.cpu_integrate_worker_ms += chunk.cpu_integrate_worker_ms;
    }
    Ok((output, phases))
}

enum Input<I> {
    Ordinary(Vec<I>),
    Pinned(PinnedHostSlice<I>),
}

struct Pending {
    first: usize,
    start: CudaEvent,
    uploaded: CudaEvent,
    integrated: CudaEvent,
    downloaded: CudaEvent,
}

struct Tile<'a, I> {
    batch: Batch<'a, f64, I>,
    input: Input<I>,
    decoded: Vec<Reader>,
    output: PinnedHostSlice<f64>,
    pending: Option<Pending>,
}

impl<'a, I: DeviceRepr + ValidAsZeroBits + From<i32> + Default + Copy + Send> Tile<'a, I> {
    fn new(plan: &'a Plan<'a, f64>, use_pinned: bool) -> Result<Self> {
        let n = plan.pixels as usize * TILE;
        Ok(Self {
            batch: plan.batch_with_input(TILE as u32, true, plan.cuda.context.new_stream()?)?,
            input: if use_pinned {
                Input::Pinned(pinned(&plan.cuda.context, n)?)
            } else {
                Input::Ordinary(vec![I::default(); n])
            },
            decoded: (0..TILE).map(|_| Reader::new()).collect(),
            output: pinned(&plan.cuda.context, TILE * PATTERN)?,
            pending: None,
        })
    }

    fn submit(
        &mut self,
        first: usize,
        paths: &[String],
        loader: &Loader<'_, I>,
        kernel: &Kernel<f64, I>,
        phases: &mut Phases,
    ) -> Result<()> {
        let start = Instant::now();
        let input = match &mut self.input {
            Input::Ordinary(buf) => buf.as_mut_slice(),
            Input::Pinned(buf) => buf.as_mut_slice()?,
        };
        let loads = loader.pool.install(|| {
            self.decoded
                .par_iter_mut()
                .zip(paths.par_iter())
                .zip(input.par_chunks_exact_mut(loader.shape.0 * loader.shape.1))
                .map(|((reader, path), dst)| {
                    (loader.load)(reader, path, loader.shape, dst).map_err(|err| err.to_string())
                })
                .collect::<std::result::Result<Vec<_>, String>>()
        })?;
        phases.gpu_loader_wall_ms += elapsed_ms(start);
        for times in loads {
            phases.read_decode_worker_ms += times.read_decode;
            phases.direct_read_worker_ms += times.direct_read;
            phases.widen_worker_ms += times.stage;
        }
        let stream = Arc::clone(&self.batch.stream);
        let event = || stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT));
        let start = event()?;
        match &self.input {
            Input::Ordinary(buf) => self.batch.upload(0, buf)?,
            Input::Pinned(buf) => self.batch.upload(0, buf)?,
        }
        let uploaded = event()?;
        self.batch.run(kernel)?;
        let integrated = event()?;
        self.batch.download(&mut self.output)?;
        self.pending = Some(Pending {
            first,
            start,
            uploaded,
            integrated,
            downloaded: event()?,
        });
        Ok(())
    }

    fn finish(&mut self, output: &mut [f64], phases: &mut Phases) -> Result<()> {
        if let Some(pending) = self.pending.take() {
            pending.downloaded.synchronize()?;
            phases.gpu_upload_ms += f64::from(pending.start.elapsed_ms(&pending.uploaded)?);
            phases.gpu_kernel_ms += f64::from(pending.uploaded.elapsed_ms(&pending.integrated)?);
            phases.gpu_download_ms +=
                f64::from(pending.integrated.elapsed_ms(&pending.downloaded)?);
            let first = pending.first * PATTERN;
            output[first..first + TILE * PATTERN].copy_from_slice(self.output.as_slice()?);
        }
        Ok(())
    }
}

fn gpu_files<I: DeviceRepr + ValidAsZeroBits + From<i32> + Default + Copy + Send>(
    tiles: &mut [Tile<'_, I>; 2],
    paths: &[String],
    loader: &Loader<'_, I>,
    kernel: &Kernel<f64, I>,
) -> Result<(Vec<f64>, Phases)> {
    let mut output = vec![0.0; paths.len() * PATTERN];
    let mut phases = Phases::default();
    let run: Result<()> = (|| {
        let (path_tiles, tail) = paths.as_chunks::<TILE>();
        assert!(tail.is_empty());
        for (i, paths) in path_tiles.iter().enumerate() {
            let tile = &mut tiles[i % 2];
            tile.finish(&mut output, &mut phases)?;
            tile.submit(i * TILE, paths, loader, kernel, &mut phases)?;
        }
        for tile in tiles.iter_mut() {
            tile.finish(&mut output, &mut phases)?;
        }
        Ok(())
    })();
    if run.is_err() {
        for tile in tiles {
            let _ = tile.batch.stream.synchronize();
        }
    }
    run?;
    Ok((output, phases))
}

#[test]
#[ignore = "fresh-process Eiger file benchmark; use docs/benchmarks/2026-11-18-cuda/disk_batch.py"]
fn eiger_files() -> Result<()> {
    if !cfg!(windows) && env::var_os("DIOPTRIN_CUDA_DISK_DIRECT_IO").is_some() {
        return Err("unbuffered file benchmark requires Windows".into());
    }
    let selection = env::var("DIOPTRIN_CUDA_DISK_BACKEND")?;
    let method_name = env::var("DIOPTRIN_CUDA_DISK_METHOD")?;
    let method = METHODS
        .iter()
        .find_map(|&(method, name)| (name == method_name).then_some(method))
        .ok_or("unknown disk benchmark method")?;
    let dir = PathBuf::from(env::var_os("DIOPTRIN_CUDA_DISK_DIR").ok_or("missing TIFF directory")?);
    let paths: Vec<_> = (0..FRAMES)
        .map(|i| dir.join(format!("frame-{i:02}.tiff")))
        .map(|p| {
            p.into_os_string()
                .into_string()
                .map_err(|_| "invalid TIFF path")
        })
        .collect::<std::result::Result<_, _>>()?;
    let start = Instant::now();
    let case = Case::load("eiger", false)?;
    let fixture_ms = elapsed_ms(start);
    let mut cpu = case.strict(method)?;
    let prepare = Instant::now();
    if selection == "cpu" {
        cpu.set_fast_math(true);
        if matches!(method, IntegrationMethod::Counting) {
            cpu.set_sparse_layout(SparseLayout::Csr);
        }
        let pool = ThreadPoolBuilder::new().num_threads(14).build()?;
        // Build the layout once before starting multiple readers.
        pool.install(|| {
            black_box(cpu.integrate1d_with_errors_from_slice(
                &case.pixels,
                case.shape.0,
                case.shape.1,
                BINS as usize,
            ));
        });
        let mut buffers: Vec<_> = (0..14).map(|_| Reader::new()).collect();
        measure(fixture_ms, prepare, || {
            cpu_files(&cpu, &paths, &pool, &mut buffers)
        })?;
    } else {
        if !matches!(
            selection.as_str(),
            "cuda_pageable" | "cuda_pinned" | "cuda_native" | "cuda_native_direct"
        ) {
            return Err("unknown disk benchmark backend".into());
        }
        let use_pinned = selection != "cuda_pageable";
        let cuda = Cuda::open()?.ok_or("CUDA driver, GPU or NVRTC unavailable")?;
        let plan = cuda.prepare_f64(&cpu, case.shape, BINS)?;
        let layout = if use_pinned || matches!(method, IntegrationMethod::Counting) {
            Layout::Block
        } else {
            Layout::Warp
        };
        let pool = ThreadPoolBuilder::new().num_threads(TILE).build()?;
        if matches!(selection.as_str(), "cuda_native" | "cuda_native_direct") {
            let kernel = plan.kernel_i32(layout, true)?;
            let mut tiles = [Tile::new(&plan, true)?, Tile::new(&plan, true)?];
            let load = if selection == "cuda_native_direct" {
                decode_native
            } else {
                decode_copied
            };
            let loader = Loader {
                shape: case.shape,
                pool: &pool,
                load,
            };
            measure(fixture_ms, prepare, || {
                gpu_files(&mut tiles, &paths, &loader, &kernel)
            })?;
        } else {
            let kernel = plan.kernel(layout, true)?;
            let mut tiles = [Tile::new(&plan, use_pinned)?, Tile::new(&plan, use_pinned)?];
            let loader = Loader {
                shape: case.shape,
                pool: &pool,
                load: decode_copied,
            };
            measure(fixture_ms, prepare, || {
                gpu_files(&mut tiles, &paths, &loader, &kernel)
            })?;
        }
    }
    Ok(())
}

fn measure(
    fixture_ms: f64,
    prepare: Instant,
    mut integrate: impl FnMut() -> Result<(Vec<f64>, Phases)>,
) -> Result<()> {
    let prepare_ms = elapsed_ms(prepare);
    let mut first = None;
    for stage in ["fresh", "prepared"] {
        let start = Instant::now();
        let (patterns, phases) = integrate()?;
        let batch_ms = elapsed_ms(start);
        let values = json!({
            "stage": stage,
            "fixture_ms": fixture_ms,
            "prepare_ms": prepare_ms,
            "batch_ms": batch_ms,
            "phases": phases,
        });
        let mut output = std::io::stdout().lock();
        writeln!(output, "DIOPTRIN_FILE_BATCH {values}")?;
        output.flush()?;
        assert_eq!(patterns.len(), FRAMES * PATTERN);
        assert!(patterns.iter().all(|x| x.is_finite()));
        assert!(
            patterns
                .as_chunks::<PATTERN>()
                .0
                .iter()
                .all(|p| p == &patterns[..PATTERN])
        );
        if let Some(first) = &first {
            assert_eq!(first, &patterns);
        } else {
            first = Some(patterns);
        }
    }
    let patterns = first.ok_or("missing disk benchmark output")?;
    let mut output = std::io::stdout().lock();
    writeln!(
        output,
        "DIOPTRIN_FILE_OUTPUT {}",
        json!(&patterns[..PATTERN])
    )?;
    output.flush()?;
    Ok(())
}
