use super::*;
use crate::{frame::Frame, geometry::Geometry, integration::SparseLayout, mask::Mask, poni::Poni};
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
    path::Path,
};

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

fn measure_gpu(
    batch: &mut Batch<'_>,
    layout: Layout,
    mode: Accumulation,
    workload: &Workload<'_>,
    input: &mut impl HostSlice<f32>,
    fill: impl Fn(&mut [f32], &[f64], &ThreadPool),
    output: &mut impl HostSlice<f64>,
) -> Result<Vec<Samples>> {
    let stream = &batch.plan.cuda.stream;
    let function = batch.plan.function(mode, layout, batch.errors)?;
    let mut samples = Vec::with_capacity(workload.repeats);
    for iteration in 0..workload.repeats + 2 {
        let start = Instant::now();
        batch.upload(input)?;
        stream.synchronize()?;
        let upload_ms = elapsed_ms(start);

        let event_start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        batch.run(&function, layout)?;
        let event_end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let device_kernel_ms = f64::from(event_start.elapsed_ms(&event_end)?);

        let start = Instant::now();
        batch.download(output)?;
        stream.synchronize()?;
        let download_ms = elapsed_ms(start);

        let start = Instant::now();
        batch.run(&function, layout)?;
        stream.synchronize()?;
        let resident_wall_ms = elapsed_ms(start);

        let start = Instant::now();
        batch.upload(input)?;
        batch.run(&function, layout)?;
        batch.download(output)?;
        stream.synchronize()?;
        let host_f32_ms = elapsed_ms(start);

        let start = Instant::now();
        // The host buffer is initialized and the stream was synchronized.
        let (destination, _) = unsafe { input.stream_synced_mut_slice(stream) };
        fill(destination, workload.data, workload.casting_pool);
        batch.upload(input)?;
        batch.run(&function, layout)?;
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
    fn load(name: &str) -> Result<Self> {
        let (geometry, shape, pixels, mask, files) = match name {
            "pilatus" | "eiger" => {
                let (image, poni) = if name == "pilatus" {
                    ("test_data/CeO2_Pilatus1M.tif", "test_data/v1.poni")
                } else {
                    ("test_data/eiger_0000.tiff", "test_data/eiger_0000.poni")
                };
                let frame = Frame::from_tiff(image.into())?;
                let shape = (frame.dim1, frame.dim2);
                let files = json!({image: sha256(image)?, poni: sha256(poni)?});
                let (mask, files) = if name == "eiger" {
                    let raw = env::var("DIOPTRIN_CUDA_EIGER_MASK_U8")?;
                    let mask = fs::read(raw)?;
                    let mut files = files;
                    files["test_data/eiger_0000.mask"] =
                        json!(sha256("test_data/eiger_0000.mask")?);
                    (mask, files)
                } else {
                    (synthetic_mask(frame.data.len()), files)
                };
                (
                    Geometry::new(shape.0, shape.1, Poni::load(poni)?),
                    shape,
                    frame.data,
                    mask,
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
                let mut seed = 0x1bbd_412b_9aa4_3809_u64;
                let data = (0..count)
                    .map(|i| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let s = (seed % 70_000) as f64 / 97.0 - 3.0;
                        if i % 1_681 == 0 { s * 1_000.0 } else { s }
                    })
                    .collect();
                (
                    Geometry::new(side, side, poni),
                    (side, side),
                    data,
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
            .with_polarization_factor(0.95)
            .with_fast_math(false))
    }
}

fn synthetic_mask(count: usize) -> Vec<u8> {
    (0..count).map(|i| u8::from(i % 131 <= 3)).collect()
}

fn sha256(path: impl AsRef<Path>) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn log(out: &mut impl Write, record: serde_json::Value) -> Result<()> {
    serde_json::to_writer(&mut *out, &record)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
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
    let suite = env::var("DIOPTRIN_CUDA_CASES").unwrap_or_else(|_| "256,pilatus,1024,2048".into());
    let requested = env::var("DIOPTRIN_CUDA_BATCHES").unwrap_or_else(|_| "1,8,64,256,max".into());
    let pools: Vec<_> = [1, 4, 8, 16]
        .into_iter()
        .map(|n| ThreadPoolBuilder::new().num_threads(n).build())
        .collect::<std::result::Result<_, _>>()?;
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
        let case = Case::load(name)?;
        log(
            &mut out,
            json!({"type": "case", "case": case.name, "shape": case.shape, "sha256": case.files}),
        )?;
        for (method, label) in METHODS {
            eprintln!("{} {label}", case.name);
            let strict = case.strict(method)?;
            let setup = Instant::now();
            let plan = cuda.prepare(&strict, case.shape, BINS)?;
            let (free, total) = cuda.context.mem_get_info()?;
            log(
                &mut out,
                json!({
                    "type": "table", "case": case.name, "method": label, "entries": plan.entries,
                    "prepare_ms": elapsed_ms(setup), "upload_ms": plan.upload_ms,
                    "free_device_bytes_after_table": free, "device_bytes": total,
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
                let device_cost = case.pixels.len() * size_of::<f32>()
                    + BINS as usize * stride * size_of::<f64>();
                let host_cost = device_cost + case.pixels.len() * size_of::<f64>();
                let device_batches = free.saturating_sub(GPU_RESERVE) / device_cost;
                let max = device_batches
                    .min(host_limit / host_cost)
                    .min(i32::MAX as usize / BINS as usize);
                let mut prev = None;
                for batch_name in requested.split(',') {
                    let count: usize = if batch_name == "max" {
                        max
                    } else {
                        batch_name.parse()?
                    };
                    if count == 0 || count > max {
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
                    eprintln!("  variance={errors} {batch_name} ({count} frames)");
                    let data = case.pixels.repeat(count);
                    let workload = Workload {
                        data: &data,
                        repeats,
                        casting_pool: &pools[2],
                    };
                    let mut batch = plan.batch(u32::try_from(count)?, errors)?;
                    let samples_count = BINS as usize * count * stride;
                    let record = |kind| {
                        json!({
                            "type": kind, "case": name, "method": label, "errors": errors,
                            "batch": batch_name, "frames": count, "max_device_frames": device_batches,
                            "host_buffer_bytes": host_cost * count,
                        })
                    };
                    {
                        let mut input = vec![0.0; data.len()];
                        cast_pixels(&mut input, &data, &pools[2]);
                        let mut output = vec![0.0; samples_count];
                        for layout in Layout::ALL {
                            for mode in Accumulation::ALL {
                                let samples = measure_gpu(
                                    &mut batch,
                                    layout,
                                    mode,
                                    &workload,
                                    &mut input,
                                    cast_pixels,
                                    &mut output,
                                )?;
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
                    match pinned::<f32>(&cuda.context, data.len())
                        .and_then(|input| Ok((input, pinned::<f64>(&cuda.context, samples_count)?)))
                    {
                        Ok((mut input, mut output)) => {
                            cast_pixels(input.as_mut_slice()?, &data, &pools[2]);
                            for layout in Layout::ALL {
                                for mode in Accumulation::ALL {
                                    let samples = measure_gpu(
                                        &mut batch,
                                        layout,
                                        mode,
                                        &workload,
                                        &mut input,
                                        cast_pixels,
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
                        for (pool, threads) in pools.iter().zip([1, 4, 8, 16]) {
                            let mut samples = Vec::with_capacity(repeats);
                            for i in 0..repeats + 2 {
                                let start = Instant::now();
                                pool.install(|| {
                                    data.par_chunks_exact(case.pixels.len()).for_each(|frame| {
                                        if errors {
                                            black_box(cpu.integrate1d_with_errors_from_slice(
                                                frame,
                                                case.shape.0,
                                                case.shape.1,
                                                BINS as usize,
                                            ));
                                        } else {
                                            black_box(cpu.integrate1d_from_slice(
                                                frame,
                                                case.shape.0,
                                                case.shape.1,
                                                BINS as usize,
                                            ));
                                        }
                                    });
                                });
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
