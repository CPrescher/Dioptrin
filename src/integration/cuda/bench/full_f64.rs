use super::*;

#[derive(Serialize)]
struct PhaseSample {
    upload_ms: f64,
    device_kernel_ms: f64,
    download_ms: f64,
    host_ms: f64,
}

fn measure(
    batch: &mut Batch<'_, f64>,
    kernel: &Kernel<f64>,
    input: &(impl HostSlice<f64> + ?Sized),
    output: &mut PinnedHostSlice<f64>,
    repeats: usize,
) -> Result<Vec<PhaseSample>> {
    let stream = Arc::clone(&batch.stream);
    let mut samples = Vec::with_capacity(repeats);
    for iteration in 0..repeats + 2 {
        let start = Instant::now();
        batch.upload(0, input)?;
        stream.synchronize()?;
        let upload_ms = elapsed_ms(start);

        let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        batch.run(kernel)?;
        let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let device_kernel_ms = f64::from(start.elapsed_ms(&end)?);

        let start = Instant::now();
        batch.download(output)?;
        stream.synchronize()?;
        let download_ms = elapsed_ms(start);

        let start = Instant::now();
        batch.upload(0, input)?;
        batch.run(kernel)?;
        batch.download(output)?;
        stream.synchronize()?;
        let host_ms = elapsed_ms(start);
        if iteration >= 2 {
            samples.push(PhaseSample {
                upload_ms,
                device_kernel_ms,
                download_ms,
                host_ms,
            });
        }
    }
    Ok(samples)
}

#[test]
#[ignore = "hardware benchmark; use run.py --gpu-full-f64"]
fn gpu_full_f64() -> Result<()> {
    let cuda = Cuda::open(0)?;
    let mut out = BufWriter::new(File::create(env::var("DIOPTRIN_CUDA_OUT")?)?);
    let cases = env::var("DIOPTRIN_CUDA_CASES").unwrap_or_else(|_| "eiger".into());
    let methods = env::var("DIOPTRIN_CUDA_METHODS")
        .unwrap_or_else(|_| "counting,pixel_split,pyfai_full".into());
    let layout_names = env::var("DIOPTRIN_CUDA_LAYOUTS").unwrap_or_else(|_| "block,warp".into());
    let layouts: Vec<_> = Layout::ALL
        .into_iter()
        .filter(|layout| layout_names.split(',').any(|name| name == layout.name()))
        .collect();
    if layouts.is_empty() {
        return Err("select block, warp or both CUDA layouts".into());
    }
    let batches = env::var("DIOPTRIN_CUDA_BATCHES").unwrap_or_else(|_| "64".into());
    let repeats = env::var("DIOPTRIN_CUDA_REPEATS").map_or(Ok(7), |s| s.parse::<usize>())?;
    if repeats < 3 {
        return Err("use at least three timed repeats".into());
    }
    let host_limit =
        env::var("DIOPTRIN_CUDA_HOST_BYTES").map_or(Ok(HOST_LIMIT), |s| s.parse::<usize>())?;
    let pools: Vec<_> = thread_counts("14")?
        .into_iter()
        .map(|n| ThreadPoolBuilder::new().num_threads(n).build())
        .collect::<std::result::Result<_, _>>()?;
    log(
        &mut out,
        json!({
            "type": "setup", "gpu": cuda.context.name()?,
            "compute_capability": cuda.context.compute_capability()?,
            "compile_ms": cuda.compile_ms, "module_ms": cuda.module_ms,
            "bins": BINS, "repeats": repeats, "host_limit_bytes": host_limit,
        }),
    )?;

    for name in cases.split(',') {
        let case = Case::load(name, true)?;
        log(
            &mut out,
            json!({"type": "case", "case": name, "shape": case.shape, "sha256": case.files}),
        )?;
        for (method, label) in METHODS {
            if !methods.split(',').any(|name| name == label) {
                continue;
            }
            eprintln!("{name} {label}");
            let mut cpu = case.strict(method)?;
            let start = Instant::now();
            let plan = cuda.prepare_f64(&cpu, case.shape, BINS)?;
            let prepare_ms = elapsed_ms(start);
            let (free, _) = cuda.context.mem_get_info()?;
            let dedicated = free_dedicated_bytes(&cuda.context);
            let budget = dedicated.map_or(free, |bytes| bytes.min(free));
            log(
                &mut out,
                json!({
                    "type": "table", "case": name, "method": label, "entries": plan.entries,
                    "prepare_ms": prepare_ms, "upload_ms": plan.upload_ms,
                    "free_device_bytes_after_table": free, "free_dedicated_bytes_after_table": dedicated,
                }),
            )?;
            let (_, intensity, sigma) = cpu.integrate1d_with_errors_from_slice(
                &case.pixels,
                case.shape.0,
                case.shape.1,
                BINS as usize,
            );
            let pattern = BINS as usize * 2;
            let bytes_per_frame = (case.pixels.len() + pattern) * size_of::<f64>();
            let max_frames = host_limit.min(budget.saturating_sub(GPU_RESERVE)) / bytes_per_frame;
            let max_frames = max_frames.min((i32::MAX as u32 / BINS) as usize);
            for frames in batches.split(',').map(str::parse::<u32>) {
                let frames = frames?;
                let count = frames as usize;
                if frames == 0 || count > max_frames {
                    return Err(format!(
                        "full-f64 batch of {frames} frames exceeds buffer budget or is empty"
                    )
                    .into());
                }
                let (input_len, output_len) = (case.pixels.len() * count, pattern * count);
                let pageable = case.pixels.repeat(count);
                for (label_cpu, fast_math, layout) in [
                    ("fast_csc", true, SparseLayout::Csc),
                    ("fast_csr", true, SparseLayout::Csr),
                    ("strict_csr", false, SparseLayout::Csr),
                ] {
                    cpu.set_fast_math(fast_math);
                    cpu.set_sparse_layout(layout);
                    for pool in &pools {
                        // Prepare lazy layout data from a single caller before
                        // dispatching frames across the pool.
                        pool.install(|| {
                            black_box(cpu.integrate1d_with_errors_from_slice(
                                &case.pixels,
                                case.shape.0,
                                case.shape.1,
                                BINS as usize,
                            ));
                        });
                        let samples = sample(repeats, || {
                            cpu_batch(&cpu, case.shape, &pageable, count, true, pool);
                            Ok(())
                        })?;
                        log(
                            &mut out,
                            json!({
                                "type": "cpu_f64", "case": name, "method": label, "frames": frames,
                                "layout": label_cpu, "threads": pool.current_num_threads(),
                                "samples_ms_per_batch": samples,
                            }),
                        )?;
                    }
                }
                let mut batch = plan.batch(frames, true)?;
                let mut output = pinned(&cuda.context, output_len)?;
                let mut record = |memory, layout: Layout, samples, output: &[f64]| {
                    let first = &output[..pattern];
                    assert!(output.chunks_exact(pattern).all(|frame| frame == first));
                    log(
                        &mut out,
                        json!({
                            "type": "gpu_full_f64", "case": name, "method": label, "frames": frames,
                            "memory": memory, "layout": layout.name(), "samples_ms_per_batch": samples,
                            "intensity_error": accuracy(first.iter().step_by(2).copied(), &intensity),
                            "std_error": accuracy(first[1..].iter().step_by(2).copied(), &sigma),
                        }),
                    )
                };
                for &layout in &layouts {
                    let kernel = plan.kernel(layout, true)?;
                    let samples = measure(&mut batch, &kernel, &pageable, &mut output, repeats)?;
                    record("pageable", layout, samples, output.as_slice()?)?;
                }
                drop(pageable);
                // Model a producer that already supplies pinned f64 input.
                let mut input = pinned(&cuda.context, input_len)?;
                for frame in input.as_mut_slice()?.chunks_exact_mut(case.pixels.len()) {
                    frame.copy_from_slice(&case.pixels);
                }
                for &layout in &layouts {
                    let kernel = plan.kernel(layout, true)?;
                    let samples = measure(&mut batch, &kernel, &input, &mut output, repeats)?;
                    record("pinned", layout, samples, output.as_slice()?)?;
                }
            }
        }
    }
    Ok(())
}
