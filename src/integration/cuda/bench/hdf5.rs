use super::*;
use cudarc::driver::CudaEvent;
use std::{sync::mpsc::sync_channel, thread};

mod reader;
use reader::{Dataset, LibraryApi};

const PATTERN: usize = BINS as usize * 2;

#[derive(serde::Deserialize)]
struct ScanGeometry {
    detector: String,
    pixel1: f64,
    pixel2: f64,
    distance_m: f64,
    center_row: f64,
    center_col: f64,
    wavelength_m: f64,
}

impl ScanGeometry {
    fn integrator(&self, shape: (usize, usize), method: IntegrationMethod) -> Result<Integrator> {
        let detector = crate::detector::named(&self.detector)?;
        let mut mask = detector.builtin_mask([shape.0, shape.1])?;
        if let Some(path) = env::var_os("DIOPTRIN_HDF5_EXTRA_MASK") {
            let indices: Vec<usize> = serde_json::from_reader(File::open(path)?)?;
            for index in indices {
                *mask
                    .get_mut(index)
                    .ok_or("HDF5 mask index outside detector")? = 1;
            }
        }
        let mut poni = Poni::new();
        poni.pixel1 = self.pixel1;
        poni.pixel2 = self.pixel2;
        poni.distance = self.distance_m;
        poni.poni1 = self.center_row * self.pixel1;
        poni.poni2 = self.center_col * self.pixel2;
        poni.wavelength = self.wavelength_m;
        let geometry = Geometry::new(shape.0, shape.1, poni).with_detector(Arc::new(detector))?;
        Ok(Integrator::new(geometry)
            .with_radial_unit(RadialUnit::TthDeg)
            .with_method(method)
            .with_mask(Mask::from_vec(mask, shape.0, shape.1)?))
    }
}

#[derive(Default, Serialize)]
struct Phases {
    read_decode_ms: f64,
    widen_ms: f64,
    cpu_integration_ms: f64,
    upload_ms: f64,
    kernel_ms: f64,
    download_ms: f64,
}

struct Work {
    first: usize,
    frames: usize,
    tile: usize,
    shape: (usize, usize),
}

impl Work {
    fn pixels(&self) -> usize {
        self.shape.0 * self.shape.1
    }
}

fn cpu_batch_hdf5(
    source: &mut Dataset<'_>,
    work: &Work,
    cpu: &Integrator,
    pool: &ThreadPool,
    buffers: &mut [Vec<u32>; 2],
) -> Result<(Vec<f64>, Phases)> {
    let pixels = work.pixels();
    // One serial HDF5 reader and a 14-worker compute pool. The two
    // reusable buffers bound memory and overlap the next read with work.
    thread::scope(|scope| {
        let (free_tx, free_rx) = sync_channel(2);
        let (ready_tx, ready_rx) = sync_channel::<(usize, &mut Vec<u32>)>(2);
        for buffer in buffers {
            free_tx
                .send(buffer)
                .map_err(|_| "CPU recycle queue closed")?;
        }
        let worker = scope.spawn(move || {
            let mut output = Vec::with_capacity(work.frames * PATTERN);
            let mut integrate_ms = 0.;
            for (count, buffer) in ready_rx {
                let start = Instant::now();
                let patterns: Vec<_> = pool.install(|| {
                    buffer[..count * pixels]
                        .par_chunks_exact(pixels)
                        .map(|frame| integrate_pixels(cpu, work.shape, frame, true))
                        .collect()
                });
                for (_, intensity, sigma) in patterns {
                    output.extend(intensity.into_iter().zip(sigma).flat_map(|(i, s)| [i, s]));
                }
                integrate_ms += elapsed_ms(start);
                let _ = free_tx.send(buffer);
            }
            (output, integrate_ms)
        });
        let mut phases = Phases::default();
        let read: Result<()> = (|| {
            for first in (0..work.frames).step_by(work.tile) {
                let count = (work.frames - first).min(work.tile);
                let buffer = free_rx.recv().map_err(|_| "CPU worker stopped")?;
                let start = Instant::now();
                source.read(work.first + first, &mut buffer[..count * pixels])?;
                phases.read_decode_ms += elapsed_ms(start);
                ready_tx
                    .send((count, buffer))
                    .map_err(|_| "CPU worker stopped")?;
            }
            Ok(())
        })();
        drop(ready_tx);
        let (output, integration) = worker.join().map_err(|_| "CPU worker panicked")?;
        read?;
        phases.cpu_integration_ms = integration;
        Ok((output, phases))
    })
}

struct Events {
    first: usize,
    start: CudaEvent,
    upload: CudaEvent,
    kernel: CudaEvent,
    download: CudaEvent,
}

struct Slot<'a, I> {
    batch: Batch<'a, f64, I>,
    input: PinnedHostSlice<I>,
    output: PinnedHostSlice<f64>,
    events: Option<Events>,
}

impl<'a, I: DeviceRepr + ValidAsZeroBits> Slot<'a, I> {
    fn new(plan: &'a Plan<'a, f64>, frames: usize, pixels: usize) -> Result<Self> {
        Ok(Self {
            batch: plan.batch_with_input(
                frames.try_into()?,
                true,
                plan.cuda.context.new_stream()?,
            )?,
            input: pinned(&plan.cuda.context, frames * pixels)?,
            output: pinned(&plan.cuda.context, frames * PATTERN)?,
            events: None,
        })
    }

    fn finish(&mut self, output: &mut [f64], phases: &mut Phases) -> Result<()> {
        if let Some(events) = self.events.take() {
            events.download.synchronize()?;
            phases.upload_ms += f64::from(events.start.elapsed_ms(&events.upload)?);
            phases.kernel_ms += f64::from(events.upload.elapsed_ms(&events.kernel)?);
            phases.download_ms += f64::from(events.kernel.elapsed_ms(&events.download)?);
            let data = self.output.as_slice()?;
            let first = events.first * PATTERN;
            output[first..first + data.len()].copy_from_slice(data);
        }
        Ok(())
    }
}

fn cuda_batch_hdf5<I: DeviceRepr + ValidAsZeroBits>(
    source: &mut Dataset<'_>,
    work: &Work,
    kernel: &Kernel<f64, I>,
    slots: &mut [Slot<'_, I>],
    mut fill: impl FnMut(&mut Dataset<'_>, usize, &mut [I], &mut Phases) -> Result<()>,
) -> Result<(Vec<f64>, Phases)> {
    let mut output = vec![0.; work.frames * PATTERN];
    let mut phases = Phases::default();
    let result: Result<()> = (|| {
        for (i, first) in (0..work.frames).step_by(work.tile).enumerate() {
            let count = (work.frames - first).min(work.tile);
            let slot = &mut slots[if count < work.tile { 2 } else { i % 2 }];
            slot.finish(&mut output, &mut phases)?;
            fill(
                source,
                work.first + first,
                slot.input.as_mut_slice()?,
                &mut phases,
            )?;
            let stream = Arc::clone(&slot.batch.stream);
            let event = || stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT));
            let start = event()?;
            slot.batch.upload(0, &slot.input)?;
            let upload = event()?;
            slot.batch.run(kernel)?;
            let kernel = event()?;
            slot.batch.download(&mut slot.output)?;
            slot.events = Some(Events {
                first,
                start,
                upload,
                kernel,
                download: event()?,
            });
        }
        for slot in slots.iter_mut() {
            slot.finish(&mut output, &mut phases)?;
        }
        Ok(())
    })();
    if result.is_err() {
        for slot in slots {
            let _ = slot.batch.stream.synchronize();
        }
    }
    result?;
    Ok((output, phases))
}

fn slots<'a, I: DeviceRepr + ValidAsZeroBits>(
    plan: &'a Plan<'a, f64>,
    work: &Work,
) -> Result<Vec<Slot<'a, I>>> {
    [work.tile, work.tile, work.frames % work.tile]
        .into_iter()
        .filter(|&n| n != 0)
        .map(|n| Slot::new(plan, n, work.pixels()))
        .collect()
}

fn measure(
    opened_ms: f64,
    setup: Instant,
    repeats: usize,
    mut run: impl FnMut() -> Result<(Vec<f64>, Phases)>,
) -> Result<()> {
    let prepare_ms = elapsed_ms(setup);
    let mut output = Vec::new();
    for iteration in 0..=repeats {
        let start = Instant::now();
        let (values, phases) = run()?;
        let elapsed = elapsed_ms(start);
        let row = json!({
            "iteration": iteration, "open_ms": opened_ms, "prepare_ms": prepare_ms,
            "batch_ms": elapsed, "phases": phases,
        });
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "DIOPTRIN_HDF5_BATCH {row}")?;
        stdout.flush()?;
        output = values;
    }
    let mut file = BufWriter::new(File::create(
        env::var_os("DIOPTRIN_HDF5_OUTPUT").ok_or("missing output path")?,
    )?);
    for value in output {
        if !value.is_finite() {
            return Err("nonfinite HDF5 integration output".into());
        }
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(file.flush()?)
}

#[test]
#[ignore = "HDF5 pipeline benchmark; use docs/benchmarks/2026-11-18-cuda/hdf5_batch.py"]
fn eiger_hdf5() -> Result<()> {
    let geometry: ScanGeometry = serde_json::from_str(&env::var("DIOPTRIN_HDF5_GEOMETRY")?)?;
    let file = env::var("DIOPTRIN_HDF5_FILE")?;
    let dataset = env::var("DIOPTRIN_HDF5_DATASET")?;
    let method = env::var("DIOPTRIN_HDF5_METHOD")?;
    let backend = env::var("DIOPTRIN_HDF5_BACKEND")?;
    let repeats = env::var("DIOPTRIN_HDF5_REPEATS")?.parse()?;
    let method = METHODS
        .iter()
        .find_map(|&(m, name)| (name == method).then_some(m))
        .ok_or("unknown HDF5 benchmark method")?;
    let start = Instant::now();
    let library = PathBuf::from(env::var_os("DIOPTRIN_HDF5_LIBRARY").ok_or("missing HDF5 DLL")?);
    let api = LibraryApi::load(&library)?;
    let mut source = api.dataset(&file, &dataset)?;
    let opened_ms = elapsed_ms(start);
    let [available, rows, columns] = source.dims;
    let first = env::var("DIOPTRIN_HDF5_FIRST").map_or(Ok(0), |n| n.parse::<usize>())?;
    let requested = env::var("DIOPTRIN_HDF5_FRAMES")?.parse::<usize>()?;
    let tile = env::var("DIOPTRIN_HDF5_TILE")?.parse::<usize>()?;
    if requested == 0
        || first > available
        || requested > available - first
        || !(1..=64).contains(&tile)
    {
        return Err("invalid HDF5 frame count or tile size".into());
    }
    let work = Work {
        first,
        frames: requested,
        tile: tile.min(requested),
        shape: (rows, columns),
    };
    let setup = Instant::now();
    let mut cpu = geometry.integrator(work.shape, method)?;
    if matches!(backend.as_str(), "cpu_csc" | "cpu_csr") {
        if backend == "cpu_csr" {
            cpu.set_sparse_layout(SparseLayout::Csr);
        }
        let threads =
            env::var("DIOPTRIN_HDF5_CPU_WORKERS").map_or(Ok(14), |n| n.parse::<usize>())?;
        if !(1..=14).contains(&threads) {
            return Err("expected 1 to 14 CPU integration workers".into());
        }
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        pool.install(|| {
            black_box(integrate_pixels(
                &cpu,
                work.shape,
                &vec![0u32; work.pixels()],
                true,
            ));
        });
        let mut buffers = std::array::from_fn(|_| vec![0u32; work.pixels() * work.tile]);
        return measure(opened_ms, setup, repeats, || {
            cpu_batch_hdf5(&mut source, &work, &cpu, &pool, &mut buffers)
        });
    }
    let cuda = Cuda::open()?.ok_or("CUDA driver, GPU or NVRTC unavailable")?;
    let plan = cuda.prepare_f64(&cpu, work.shape, BINS)?;
    let (layout, native) = match backend.as_str() {
        "cuda_native_block" => (Layout::Block, true),
        "cuda_native_warp" => (Layout::Warp, true),
        "cuda_f64" => (Layout::Block, false),
        _ => return Err("unknown HDF5 benchmark backend".into()),
    };
    if native {
        let kernel = plan.kernel_u32(layout, true)?;
        let mut slots = slots(&plan, &work)?;
        measure(opened_ms, setup, repeats, || {
            cuda_batch_hdf5(
                &mut source,
                &work,
                &kernel,
                &mut slots,
                |source, first, dst, phases| {
                    let start = Instant::now();
                    source.read(first, dst)?;
                    phases.read_decode_ms += elapsed_ms(start);
                    Ok(())
                },
            )
        })
    } else {
        let kernel = plan.kernel(layout, true)?;
        let mut slots = slots(&plan, &work)?;
        let mut native = vec![0u32; work.pixels() * work.tile];
        let pool = ThreadPoolBuilder::new().num_threads(8).build()?;
        measure(opened_ms, setup, repeats, || {
            cuda_batch_hdf5(
                &mut source,
                &work,
                &kernel,
                &mut slots,
                |source, first, dst, phases| {
                    let start = Instant::now();
                    let src = &mut native[..dst.len()];
                    source.read(first, src)?;
                    phases.read_decode_ms += elapsed_ms(start);
                    let start = Instant::now();
                    pool.install(|| {
                        dst.par_chunks_mut(65_536)
                            .zip(src.par_chunks(65_536))
                            .for_each(|(dst, src)| {
                                for (d, &s) in dst.iter_mut().zip(src) {
                                    *d = f64::from(s);
                                }
                            });
                    });
                    phases.widen_ms += elapsed_ms(start);
                    Ok(())
                },
            )
        })
    }
}
