use super::*;
use crate::{
    frame::{Frame, Pixel},
    geometry::Geometry,
    integration::{RadialUnit, SparseLayout},
    mask::Mask,
    poni::Poni,
};
use cudarc::driver::{DeviceRepr, PinnedHostSlice, ValidAsZeroBits, sys::CUevent_flags};
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File},
    hint::black_box,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

mod file_batch;
mod float32;
mod full_f64;
mod hdf5;
mod headroom;
mod inputs;
mod paper;
mod startup;
mod transfers;

const BINS: u32 = 3_000;
const HOST_LIMIT: usize = 15_000_000_000;
const GPU_RESERVE: usize = 128 * 1024 * 1024;
const METHODS: [(IntegrationMethod, &str); 3] = [
    (IntegrationMethod::Counting, "counting"),
    (IntegrationMethod::PixelSplit, "pixel_split"),
    (IntegrationMethod::PyfaiFull, "pyfai_full"),
];

#[derive(Serialize)]
struct Samples {
    upload_ms: f64,
    device_kernel_ms: f64,
    download_ms: f64,
    resident_wall_ms: f64,
    host_f32_ms: f64,
    host_f64_ms: f64,
}

struct Workload<'a> {
    data: &'a [f64],
    repeats: usize,
    casting_pool: &'a ThreadPool,
}

trait StagingInput: HostSlice<f32> {
    fn pixels_mut(&mut self) -> Result<&mut [f32]>;
}

impl StagingInput for Vec<f32> {
    fn pixels_mut(&mut self) -> Result<&mut [f32]> {
        Ok(self)
    }
}

impl StagingInput for PinnedHostSlice<f32> {
    fn pixels_mut(&mut self) -> Result<&mut [f32]> {
        Ok(self.as_mut_slice()?)
    }
}

struct Staging<I> {
    full: I,
    tail: Option<I>,
}

impl<I: StagingInput> Staging<I> {
    fn new(
        work: &Workload<'_>,
        elements: usize,
        mut allocate: impl FnMut(usize) -> Result<I>,
    ) -> Result<Self> {
        let mut full = allocate(work.data.len())?;
        cast_pixels(full.pixels_mut()?, work.data, work.casting_pool);
        let remaining = elements % full.len();
        let tail = if remaining == 0 {
            None
        } else {
            let mut tail = allocate(remaining)?;
            cast_pixels(
                tail.pixels_mut()?,
                &work.data[..remaining],
                work.casting_pool,
            );
            Some(tail)
        };
        Ok(Self { full, tail })
    }

    fn upload(&mut self, batch: &mut Batch<'_>, convert: Option<&Workload<'_>>) -> Result<()> {
        let full_len = self.full.len();
        let end = batch.input.len() - batch.input.len() % full_len;
        for offset in (0..end).step_by(full_len) {
            if let Some(work) = convert {
                cast_pixels(self.full.pixels_mut()?, work.data, work.casting_pool);
            }
            batch.upload(offset, &self.full)?;
        }
        if let Some(tail) = &mut self.tail {
            if let Some(work) = convert {
                let remaining = tail.len();
                cast_pixels(
                    tail.pixels_mut()?,
                    &work.data[..remaining],
                    work.casting_pool,
                );
            }
            batch.upload(end, tail)?;
        }
        Ok(())
    }
}

fn measure_gpu(
    batch: &mut Batch<'_>,
    layout: Layout,
    mode: Accumulation,
    workload: &Workload<'_>,
    input: &mut Staging<impl StagingInput>,
    output: &mut impl HostSlice<f64>,
) -> Result<Vec<Samples>> {
    let stream = &batch.plan.cuda.stream;
    let kernel = batch.plan.kernel(mode, layout, batch.errors)?;
    let mut samples = Vec::with_capacity(workload.repeats);
    for iteration in 0..workload.repeats + 2 {
        let start = Instant::now();
        input.upload(batch, None)?;
        stream.synchronize()?;
        let upload_ms = elapsed_ms(start);

        let event_start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        batch.run(&kernel)?;
        let event_end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let device_kernel_ms = f64::from(event_start.elapsed_ms(&event_end)?);

        let start = Instant::now();
        batch.download(output)?;
        stream.synchronize()?;
        let download_ms = elapsed_ms(start);

        let start = Instant::now();
        batch.run(&kernel)?;
        stream.synchronize()?;
        let resident_wall_ms = elapsed_ms(start);

        let start = Instant::now();
        input.upload(batch, None)?;
        batch.run(&kernel)?;
        batch.download(output)?;
        stream.synchronize()?;
        let host_f32_ms = elapsed_ms(start);

        let start = Instant::now();
        input.upload(batch, Some(workload))?;
        batch.run(&kernel)?;
        batch.download(output)?;
        stream.synchronize()?;
        let host_f64_ms = elapsed_ms(start);
        if iteration >= 2 {
            samples.push(Samples {
                upload_ms,
                device_kernel_ms,
                download_ms,
                resident_wall_ms,
                host_f32_ms,
                host_f64_ms,
            });
        }
    }
    Ok(samples)
}

fn cast_pixels(dst: &mut [f32], src: &[f64], pool: &ThreadPool) {
    // Keep one frame's conversion in the caller for small inputs. Parallel
    // chunks amortize dispatch on larger batches without per-pixel tasks.
    let convert = |(dst, src): (&mut [f32], &[f64])| {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = s as f32;
        }
    };
    if src.len() < 131_072 {
        convert((dst, src));
    } else {
        pool.install(|| {
            dst.par_chunks_mut(65_536)
                .zip(src.par_chunks(65_536))
                .for_each(convert);
        });
    }
}

fn pinned<T: DeviceRepr + ValidAsZeroBits>(
    context: &Arc<CudaContext>,
    len: usize,
) -> Result<PinnedHostSlice<T>> {
    // Initialize the pinned allocation before creating a typed slice.
    // Flags 0 use cacheable pages for both upload and download.
    let mut host = unsafe { context.alloc_pinned_with_flags::<T>(len, 0)? };
    unsafe { host.as_mut_ptr()?.write_bytes(0, len) };
    Ok(host)
}

fn accuracy(actual: impl Iterator<Item = f64>, expected: &[f64]) -> serde_json::Value {
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut zeros = 0;
    for (a, &b) in actual.zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        if b == 0.0 {
            zeros += 1;
        } else {
            max_rel = max_rel.max(abs / b.abs());
        }
    }
    json!({"max_absolute": max_abs, "max_relative_nonzero": max_rel, "zero_reference_bins": zeros})
}

struct Case {
    name: String,
    geometry: Geometry,
    shape: (usize, usize),
    pixels: Vec<f64>,
    mask: Vec<u8>,
    files: serde_json::Value,
}

impl Case {
    fn load(name: &str, record_hashes: bool) -> Result<Self> {
        let (geometry, shape, pixels, mask, files) = match name {
            "pilatus" => {
                let image = "test_data/CeO2_Pilatus1M.tif";
                let poni = "test_data/v1.poni";
                let frame = Frame::from_tiff(image.into())?;
                let shape = (frame.dim1, frame.dim2);
                let files = if record_hashes {
                    json!({image: sha256(image)?, poni: sha256(poni)?})
                } else {
                    json!({})
                };
                let mask = frame
                    .data
                    .iter()
                    .map(|x| u8::from(!x.is_finite() || *x < 0.0))
                    .collect();
                (
                    Geometry::new(shape.0, shape.1, Poni::load(poni)?),
                    shape,
                    frame.data,
                    mask,
                    files,
                )
            }
            "eiger" | "eiger-synthetic" => {
                let dir = env::var_os("DIOPTRIN_CUDA_EIGER_DIR")
                    .map_or_else(|| PathBuf::from("test_data"), PathBuf::from);
                let calibration = "docs/benchmarks/2026-11-05-bslz4/calibration.poni";
                let mask_path = dir.join("eiger_0000.mask");
                let base_poni = dir.join("eiger_0000.poni");
                let mut mask =
                    Mask::from_tiff(mask_path.to_str().ok_or("invalid mask path")?.into())?;
                let shape = (mask.dim1, mask.dim2);
                let mut files = if record_hashes {
                    json!({
                        "eiger_0000.poni": sha256(&base_poni)?,
                        "eiger_0000.mask": sha256(&mask_path)?,
                        calibration: sha256(calibration)?,
                    })
                } else {
                    json!({})
                };
                let pixels = if name == "eiger" {
                    let image = dir.join("eiger_0000.tiff");
                    let frame =
                        Frame::from_tiff(image.to_str().ok_or("invalid TIFF path")?.into())?;
                    if shape != (frame.dim1, frame.dim2) {
                        return Err("Eiger TIFF and mask shapes differ".into());
                    }
                    for (&pixel, value) in frame.data.iter().zip(&mut mask.data) {
                        if !pixel.is_finite() || pixel < 0.0 {
                            *value = 1;
                        }
                    }
                    if record_hashes {
                        files["eiger_0000.tiff"] = json!(sha256(image)?);
                    }
                    frame.data
                } else {
                    synthetic_pixels(mask.data.len())
                };
                let mut poni = Poni::load(base_poni.to_str().ok_or("invalid PONI path")?)?;
                // The original PONI has no pixel sizes. Fill them from the
                // derived calibration used for the Metal benchmark.
                let explicit = Poni::load(calibration)?;
                poni.pixel1 = explicit.pixel1;
                poni.pixel2 = explicit.pixel2;
                (
                    Geometry::new(shape.0, shape.1, poni),
                    shape,
                    pixels,
                    mask.data,
                    files,
                )
            }
            square => {
                let side: usize = square.parse()?;
                let count = side.checked_mul(side).ok_or("synthetic frame overflow")?;
                let mut poni = Poni::new();
                poni.pixel1 = 7.5e-5;
                poni.pixel2 = 7.5e-5;
                poni.distance = 0.2;
                poni.poni1 = 0.47 * side as f64 * poni.pixel1;
                poni.poni2 = 0.53 * side as f64 * poni.pixel2;
                poni.rot1 = 0.02;
                poni.rot2 = -0.015;
                (
                    Geometry::new(side, side, poni),
                    (side, side),
                    synthetic_pixels(count),
                    synthetic_mask(count),
                    json!({}),
                )
            }
        };
        Ok(Self {
            name: name.into(),
            geometry,
            shape,
            pixels,
            mask,
            files,
        })
    }

    fn strict(&self, method: IntegrationMethod) -> Result<Integrator> {
        Ok(Integrator::new(self.geometry.clone())
            .with_method(method)
            .with_mask(Mask::from_vec(
                self.mask.clone(),
                self.shape.0,
                self.shape.1,
            )?)
            .with_radial_unit(RadialUnit::TthDeg)
            .with_polarization_factor(0.965)
            .with_fast_math(false))
    }
}

fn synthetic_pixels(count: usize) -> Vec<f64> {
    let mut seed = 0x1bbd_412b_9aa4_3809_u64;
    (0..count)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let s = (seed % 70_000) as f64 / 97.0 - 3.0;
            if i % 1_681 == 0 { s * 1_000.0 } else { s }
        })
        .collect()
}

fn synthetic_mask(count: usize) -> Vec<u8> {
    (0..count).map(|i| u8::from(i % 131 <= 3)).collect()
}

fn sha256(path: impl AsRef<Path>) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn free_dedicated_bytes(context: &CudaContext) -> Option<usize> {
    // On WDDM, CUDA's allocation budget can exceed free dedicated VRAM.
    // Match the visible GPU by UUID so this also works with CUDA_VISIBLE_DEVICES.
    let bytes = context.uuid().ok()?.bytes.map(|byte| byte as u8);
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    let device = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..],
    );
    let output = Command::new("nvidia-smi")
        .args([
            "--id",
            &device,
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let mib: usize = std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()?;
    mib.checked_mul(1024 * 1024)
}

fn log(out: &mut impl Write, record: serde_json::Value) -> Result<()> {
    serde_json::to_writer(&mut *out, &record)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn sample(repeats: usize, mut work: impl FnMut() -> Result<()>) -> Result<Vec<f64>> {
    let mut times = Vec::with_capacity(repeats);
    for i in 0..repeats + 2 {
        let start = Instant::now();
        work()?;
        if i >= 2 {
            times.push(elapsed_ms(start));
        }
    }
    Ok(times)
}

fn integrate_pixels<P: Pixel>(
    cpu: &Integrator,
    shape: (usize, usize),
    pixels: &[P],
    errors: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let view = cpu.for_shape(shape.0, shape.1);
    let norm = view.normalization();
    let lut = view.ensure_lut1d(cpu.one_d_table_key(BINS as usize).unwrap());
    let mask = cpu.mask().map(|mask| mask.data.as_slice());
    match (cpu.fast_math(), errors) {
        (true, true) => lut.integrate_pixels::<true, true, P>(pixels, mask, &norm),
        (true, false) => lut.integrate_pixels::<true, false, P>(pixels, mask, &norm),
        (false, true) => lut.integrate_pixels::<false, true, P>(pixels, mask, &norm),
        (false, false) => lut.integrate_pixels::<false, false, P>(pixels, mask, &norm),
    }
}

fn cpu_batch(
    integrator: &Integrator,
    shape: (usize, usize),
    data: &[f64],
    count: usize,
    errors: bool,
    pool: &ThreadPool,
) {
    let pixels = shape.0 * shape.1;
    let host_frames = data.len() / pixels;
    pool.install(|| {
        for offset in (0..count).step_by(host_frames) {
            let len = host_frames.min(count - offset) * pixels;
            data[..len].par_chunks_exact(pixels).for_each(|frame| {
                if errors {
                    black_box(integrator.integrate1d_with_errors_from_slice(
                        frame,
                        shape.0,
                        shape.1,
                        BINS as usize,
                    ));
                } else {
                    black_box(integrator.integrate1d_from_slice(
                        frame,
                        shape.0,
                        shape.1,
                        BINS as usize,
                    ));
                }
            });
        }
    });
}

fn thread_counts(default: &str) -> Result<Vec<usize>> {
    let counts = env::var("DIOPTRIN_CUDA_CPU_THREADS").unwrap_or_else(|_| default.into());
    let counts = counts
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if counts.contains(&0) {
        return Err("CPU thread count must be positive".into());
    }
    Ok(counts)
}

#[test]
#[ignore = "hardware benchmark; run docs/benchmarks/2026-11-18-cuda/run.py"]
fn benchmark_cuda() -> Result<()> {
    let mut out = BufWriter::new(File::create(env::var("DIOPTRIN_CUDA_OUT")?)?);
    let cuda = Cuda::open()?.ok_or("CUDA driver, GPU or NVRTC unavailable")?;
    let repeats: usize = env::var("DIOPTRIN_CUDA_REPEATS")
        .unwrap_or_else(|_| "5".into())
        .parse()?;
    if repeats < 3 {
        return Err("use at least three timed repeats".into());
    }
    let host_limit: usize =
        env::var("DIOPTRIN_CUDA_HOST_BYTES").map_or(Ok(HOST_LIMIT), |v| v.parse())?;
    let suite = env::var("DIOPTRIN_CUDA_CASES").unwrap_or_else(|_| "pilatus".into());
    let requested = env::var("DIOPTRIN_CUDA_BATCHES").unwrap_or_else(|_| "1,8,64,256,max".into());
    let layout_names = env::var("DIOPTRIN_CUDA_LAYOUTS").unwrap_or_else(|_| "block,warp".into());
    let method_names = env::var("DIOPTRIN_CUDA_METHODS")
        .unwrap_or_else(|_| "counting,pixel_split,pyfai_full".into());
    let cpu_threads = thread_counts("1,4,8,16")?;
    let layouts: Vec<_> = Layout::ALL
        .into_iter()
        .filter(|layout| layout_names.split(',').any(|name| name == layout.name()))
        .collect();
    if layouts.is_empty() {
        return Err("select block, warp or both CUDA layouts".into());
    }
    let pools: Vec<_> = cpu_threads
        .iter()
        .map(|&n| ThreadPoolBuilder::new().num_threads(n).build())
        .collect::<std::result::Result<_, _>>()?;
    let casting_pool = ThreadPoolBuilder::new().num_threads(8).build()?;
    let mut nvrtc_major = 0;
    let mut nvrtc_minor = 0;
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) };
    log(
        &mut out,
        json!({
            "type": "setup", "gpu": cuda.context.name()?, "compute_capability": cuda.context.compute_capability()?,
            "nvrtc": [nvrtc_major, nvrtc_minor], "compile_ms": cuda.compile_ms,
            "module_ms": cuda.module_ms, "bins": BINS, "repeats": repeats,
            "host_limit_bytes": host_limit, "device_reserve_bytes": GPU_RESERVE,
        }),
    )?;

    for name in suite.split(',') {
        let case = Case::load(name, true)?;
        log(
            &mut out,
            json!({"type": "case", "case": case.name, "shape": case.shape, "sha256": case.files}),
        )?;
        for (method, label) in METHODS {
            if !method_names.split(',').any(|name| name == label) {
                continue;
            }
            eprintln!("{} {label}", case.name);
            let strict = case.strict(method)?;
            let setup = Instant::now();
            let plan = cuda.prepare(&strict, case.shape, BINS)?;
            let prepare_ms = elapsed_ms(setup);
            let (free, total) = cuda.context.mem_get_info()?;
            let dedicated = free_dedicated_bytes(&cuda.context);
            let budget = dedicated.map_or(free, |bytes| bytes.min(free));
            log(
                &mut out,
                json!({
                    "type": "table", "case": case.name, "method": label, "entries": plan.entries,
                    "prepare_ms": prepare_ms, "upload_ms": plan.upload_ms,
                    "free_device_bytes_after_table": free, "free_dedicated_bytes_after_table": dedicated,
                    "device_bytes": total,
                }),
            )?;
            let mut cpu = [case.strict(method)?, case.strict(method)?, strict];
            cpu[0].set_fast_math(true);
            cpu[1].set_fast_math(true);
            cpu[1].set_sparse_layout(SparseLayout::Csr);
            let (_, reference, sigma) = cpu[2].integrate1d_with_errors_from_slice(
                &case.pixels,
                case.shape.0,
                case.shape.1,
                BINS as usize,
            );
            for errors in [false, true] {
                let stride = 1 + usize::from(errors);
                let pixels = case.pixels.len();
                let result_cost = BINS as usize * stride * size_of::<f64>();
                let device_cost = pixels * size_of::<f32>() + result_cost;
                let input_cost = pixels * (size_of::<f64>() + size_of::<f32>());
                let device_batches = budget.saturating_sub(GPU_RESERVE) / device_cost;
                let max = device_batches.min(i32::MAX as usize / BINS as usize);
                let mut prev = None;
                for batch_name in requested.split(',') {
                    let count: usize = if batch_name == "max" {
                        max
                    } else {
                        batch_name.parse()?
                    };
                    let staging_budget =
                        host_limit.saturating_sub(result_cost.saturating_mul(count));
                    let host_frames = if count <= staging_budget / input_cost {
                        count
                    } else {
                        // Large batches reuse a host slab. Budget for a
                        // second f32 buffer in case a final partial slab remains.
                        (staging_budget / (input_cost + pixels * size_of::<f32>())).min(64)
                    };
                    if count == 0 || count > max || host_frames == 0 {
                        log(
                            &mut out,
                            json!({"type":"skip", "case": name, "method": label, "errors": errors, "batch": batch_name, "max": max, "reason": "buffer budget"}),
                        )?;
                        continue;
                    }
                    if prev == Some(count) {
                        continue;
                    }
                    prev = Some(count);
                    eprintln!(
                        "  variance={errors} {batch_name} ({count} frames, {host_frames} on host)"
                    );
                    let data = case.pixels.repeat(host_frames);
                    let workload = Workload {
                        data: &data,
                        repeats,
                        casting_pool: &casting_pool,
                    };
                    let mut batch = plan.batch(u32::try_from(count)?, errors)?;
                    let samples_count = BINS as usize * count * stride;
                    let record = |kind| {
                        json!({
                            "type": kind, "case": name, "method": label, "errors": errors,
                            "batch": batch_name, "frames": count, "max_device_frames": device_batches,
                            "host_frames": host_frames,
                            "host_buffer_bytes": input_cost * host_frames
                                + (count % host_frames) * pixels * size_of::<f32>()
                                + result_cost * count,
                        })
                    };
                    {
                        let mut input =
                            Staging::new(&workload, batch.input.len(), |len| Ok(vec![0.0; len]))?;
                        let mut output = vec![0.0; samples_count];
                        for &layout in &layouts {
                            for mode in Accumulation::ALL {
                                let samples = measure_gpu(
                                    &mut batch,
                                    layout,
                                    mode,
                                    &workload,
                                    &mut input,
                                    &mut output,
                                )?;
                                let pattern = BINS as usize * stride;
                                assert_eq!(&output[..pattern], &output[output.len() - pattern..]);
                                let mut row = record("gpu");
                                row["memory"] = json!("pageable");
                                row["layout"] = json!(layout.name());
                                row["mode"] = json!(mode.name());
                                row["samples_ms_per_batch"] = json!(samples);
                                row["intensity_error"] = accuracy(
                                    output.iter().step_by(stride).take(BINS as usize).copied(),
                                    &reference,
                                );
                                if errors {
                                    row["std_error"] = accuracy(
                                        output
                                            .iter()
                                            .skip(1)
                                            .step_by(stride)
                                            .take(BINS as usize)
                                            .copied(),
                                        &sigma,
                                    );
                                }
                                log(&mut out, row)?;
                            }
                        }
                    }
                    match Staging::new(&workload, batch.input.len(), |len| {
                        pinned::<f32>(&cuda.context, len)
                    })
                    .and_then(|input| Ok((input, pinned::<f64>(&cuda.context, samples_count)?)))
                    {
                        Ok((mut input, mut output)) => {
                            for &layout in &layouts {
                                for mode in Accumulation::ALL {
                                    let samples = measure_gpu(
                                        &mut batch,
                                        layout,
                                        mode,
                                        &workload,
                                        &mut input,
                                        &mut output,
                                    )?;
                                    let mut row = record("gpu");
                                    row["memory"] = json!("pinned");
                                    row["layout"] = json!(layout.name());
                                    row["mode"] = json!(mode.name());
                                    row["samples_ms_per_batch"] = json!(samples);
                                    log(&mut out, row)?;
                                }
                            }
                        }
                        Err(err) => {
                            let mut row = record("skip");
                            row["memory"] = json!("pinned");
                            row["reason"] = json!(err.to_string());
                            log(&mut out, row)?;
                        }
                    }
                    drop(batch);
                    for (cpu, layout) in cpu.iter().zip(["fast_csc", "fast_csr", "strict_csr"]) {
                        for (pool, threads) in pools.iter().zip(&cpu_threads) {
                            let mut samples = Vec::with_capacity(repeats);
                            for i in 0..repeats + 2 {
                                let start = Instant::now();
                                cpu_batch(cpu, case.shape, &data, count, errors, pool);
                                if i >= 2 {
                                    samples.push(elapsed_ms(start));
                                }
                            }
                            let mut row = record("cpu");
                            row["layout"] = json!(layout);
                            row["threads"] = json!(threads);
                            row["samples_ms_per_batch"] = json!(samples);
                            log(&mut out, row)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
