use super::*;

struct Chunk {
    frames: u32,
    input: PinnedHostSlice<f32>,
    output: PinnedHostSlice<f64>,
}

impl Chunk {
    fn new(plan: &Plan<'_>, frames: u32, errors: bool) -> Result<Self> {
        Ok(Self {
            frames,
            input: pinned(&plan.cuda.context, frames as usize * plan.pixels as usize)?,
            output: pinned(
                &plan.cuda.context,
                frames as usize * plan.bins as usize * (1 + usize::from(errors)),
            )?,
        })
    }
}

fn run_pipeline(
    slots: &mut [Batch<'_>],
    chunks: &mut [Chunk],
    kernel: &Kernel,
    mut upload: impl FnMut(usize, &mut Batch<'_>, &mut PinnedHostSlice<f32>) -> Result<()>,
) -> Result<()> {
    let queued: Result<()> = (|| {
        let n = slots.len();
        for (i, chunk) in chunks.iter_mut().enumerate() {
            let slot = &mut slots[i % n];
            upload(i, slot, &mut chunk.input)?;
            slot.run_frames(chunk.frames, kernel)?;
            slot.download_frames(chunk.frames, &mut chunk.output)?;
        }
        Ok(())
    })();
    // The immutable table is shared across streams. Complete all queued reads
    // before returning, including when a later operation fails to enqueue.
    let mut completed = Ok(());
    for slot in slots {
        completed = completed.and(slot.stream.synchronize());
    }
    queued?;
    Ok(completed?)
}

fn slots<'a>(plan: &'a Plan<'a>, count: usize, tile: u32, errors: bool) -> Result<Vec<Batch<'a>>> {
    (0..count)
        .map(|_| plan.batch_on(tile, errors, plan.cuda.context.new_stream()?))
        .collect()
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

#[test]
fn gpu_transfer_variants_preserve_rounded_input_and_frame_order() -> Result<()> {
    let Some(cuda) = Cuda::open()? else {
        assert!(
            env::var_os("DIOPTRIN_REQUIRE_CUDA").is_none(),
            "CUDA driver, GPU or NVRTC unavailable"
        );
        return Ok(());
    };
    let case = Case::load("31", false)?;
    let cpu = case.strict(IntegrationMethod::PixelSplit)?;
    let plan = cuda.prepare(&cpu, case.shape, 137)?;
    let cast = cuda.module.load_function("cast_f64_input")?;
    let wide: Vec<_> = (0..case.pixels.len() * 5)
        .map(|i| ((i * 37 % 311) as f64 - 14.3) / 9.7)
        .collect();
    let narrow: Vec<_> = wide.iter().map(|&x| x as f32).collect();
    for errors in [false, true] {
        let kernel = plan.kernel(Accumulation::Compensated, Layout::Warp, errors)?;
        let mut full = plan.batch(5, errors)?;
        full.upload(0, &narrow)?;
        full.run(&kernel)?;
        let mut reference = vec![0.; full.output.len()];
        full.download(&mut reference)?;
        full.stream.synchronize()?;
        let mut staging = full.stream.alloc_zeros(wide.len())?;
        full.upload_f64(&wide, &mut staging, &cast)?;
        full.run(&kernel)?;
        let mut actual = vec![0.; full.output.len()];
        full.download(&mut actual)?;
        full.stream.synchronize()?;
        assert_eq!(actual, reference);
        let mut slots = slots(&plan, 2, 2, errors)?;
        let mut io = Vec::new();
        for data in narrow.chunks(2 * case.pixels.len()) {
            let mut part = Chunk::new(&plan, (data.len() / case.pixels.len()) as u32, errors)?;
            part.input.as_mut_slice()?.copy_from_slice(data);
            io.push(part);
        }
        run_pipeline(&mut slots, &mut io, &kernel, |_, slot, data| {
            slot.upload(0, data)
        })?;
        let mut offset = 0;
        for part in io {
            let values = part.output.as_slice()?;
            assert_eq!(values, &reference[offset..offset + values.len()]);
            offset += values.len();
        }
        assert_eq!(offset, reference.len());
    }
    Ok(())
}

#[test]
#[ignore = "hardware benchmark; use run.py --gpu-transfers"]
fn gpu_transfers() -> Result<()> {
    let cuda = Cuda::open()?.ok_or("CUDA driver, GPU or NVRTC unavailable")?;
    let cast = cuda.module.load_function("cast_f64_input")?;
    let mut out = BufWriter::new(File::create(env::var("DIOPTRIN_CUDA_OUT")?)?);
    let casting_pool = ThreadPoolBuilder::new().num_threads(8).build()?;
    let repeats = env::var("DIOPTRIN_CUDA_REPEATS").map_or(Ok(7), |s| s.parse::<usize>())?;
    if repeats < 3 {
        return Err("use at least three timed repeats".into());
    }
    let cases = env::var("DIOPTRIN_CUDA_CASES").unwrap_or_else(|_| "pilatus,eiger".into());
    let methods = env::var("DIOPTRIN_CUDA_METHODS")
        .unwrap_or_else(|_| "counting,pixel_split,pyfai_full".into());
    let batches = env::var("DIOPTRIN_CUDA_BATCHES").unwrap_or_else(|_| "1,8,64".into());
    let host_limit =
        env::var("DIOPTRIN_CUDA_HOST_BYTES").map_or(Ok(HOST_LIMIT), |s| s.parse::<usize>())?;
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
            let cpu = case.strict(method)?;
            let plan = cuda.prepare(&cpu, case.shape, BINS)?;
            let layout = if case.pixels.len() > 2_000_000 && label != "counting" {
                Layout::Block
            } else {
                Layout::Warp
            };
            let kernel = plan.kernel(Accumulation::Compensated, layout, true)?;
            let (cuda_free, _) = cuda.context.mem_get_info()?;
            let free = free_dedicated_bytes(&cuda.context).map_or(cuda_free, |n| n.min(cuda_free));
            for count in batches.split(',') {
                let frames: u32 = count.parse()?;
                if frames == 0 || frames > 64 {
                    return Err("the transfer experiment takes batches of 1 to 64 frames".into());
                }
                let pixels = plan.pixels as usize * frames as usize;
                let results = BINS as usize * 2 * frames as usize;
                let bytes =
                    pixels * (size_of::<f64>() + size_of::<f32>()) + results * size_of::<f64>();
                if bytes > host_limit || bytes > free.saturating_sub(GPU_RESERVE) {
                    log(
                        &mut out,
                        json!({"type": "skip", "case": name, "method": label, "frames": frames, "reason": "buffer budget"}),
                    )?;
                    continue;
                }
                let mut record = |path: &str, tile: u32, streams: usize, times: Vec<f64>| {
                    log(
                        &mut out,
                        json!({
                            "type": "transfer", "case": name, "method": label, "frames": frames,
                            "layout": layout.name(), "path": path, "tile_frames": tile, "streams": streams,
                            "samples_ms_per_batch": times,
                        }),
                    )
                };
                let mut full = plan.batch(frames, true)?;
                let mut output = pinned(&cuda.context, results)?;
                let reference = {
                    let wide = case.pixels.repeat(frames as usize);
                    {
                        let mut narrow = pinned(&cuda.context, pixels)?;
                        cast_pixels(narrow.as_mut_slice()?, &wide, &casting_pool);
                        let baseline32 = sample(repeats, || {
                            full.upload(0, &narrow)?;
                            full.run(&kernel)?;
                            full.download(&mut output)?;
                            Ok(full.stream.synchronize()?)
                        })?;
                        record("pinned_f32", frames, 1, baseline32)?;
                        let baseline64 = sample(repeats, || {
                            cast_pixels(narrow.as_mut_slice()?, &wide, &casting_pool);
                            full.upload(0, &narrow)?;
                            full.run(&kernel)?;
                            full.download(&mut output)?;
                            Ok(full.stream.synchronize()?)
                        })?;
                        record("host_cast_f64", frames, 1, baseline64)?;
                    }
                    let reference = output.as_slice()?.to_vec();
                    let mut staging = full.stream.alloc_zeros(pixels)?;
                    let pageable = sample(repeats, || {
                        full.upload_f64(&wide, &mut staging, &cast)?;
                        full.run(&kernel)?;
                        full.download(&mut output)?;
                        Ok(full.stream.synchronize()?)
                    })?;
                    assert_eq!(output.as_slice()?, &reference);
                    record("pageable_device_cast_f64", frames, 1, pageable)?;
                    reference
                };
                // Fill the original f64 input directly in pinned pages. No
                // host copy is timed for an already-pinned input producer.
                {
                    let mut wide = pinned(&cuda.context, pixels)?;
                    for frame in wide.as_mut_slice()?.chunks_exact_mut(case.pixels.len()) {
                        frame.copy_from_slice(&case.pixels);
                    }
                    let mut staging = full.stream.alloc_zeros(pixels)?;
                    let pinned64 = sample(repeats, || {
                        full.upload_f64(&wide, &mut staging, &cast)?;
                        full.run(&kernel)?;
                        full.download(&mut output)?;
                        Ok(full.stream.synchronize()?)
                    })?;
                    assert_eq!(output.as_slice()?, &reference);
                    record("pinned_device_cast_f64", frames, 1, pinned64)?;
                }
                drop(full);
                let narrow: Vec<_> = case.pixels.iter().map(|&x| x as f32).collect();
                let wide = case.pixels.repeat(frames as usize);
                for tile in [4, 8, 16].into_iter().filter(|&tile| tile < frames) {
                    let mut io = Vec::new();
                    for first in (0..frames).step_by(tile as usize) {
                        let mut chunk = Chunk::new(&plan, tile.min(frames - first), true)?;
                        for frame in chunk.input.as_mut_slice()?.chunks_exact_mut(narrow.len()) {
                            frame.copy_from_slice(&narrow);
                        }
                        io.push(chunk);
                    }
                    for streams in [1, 2] {
                        let mut slots = slots(&plan, streams, tile, true)?;
                        let times = sample(repeats, || {
                            run_pipeline(&mut slots, &mut io, &kernel, |_, slot, data| {
                                slot.upload(0, data)
                            })
                        })?;
                        record("pinned_f32_pipeline", tile, streams, times)?;
                        let times = sample(repeats, || {
                            run_pipeline(&mut slots, &mut io, &kernel, |i, slot, data| {
                                let start = i * tile as usize * case.pixels.len();
                                let end = start + data.len();
                                cast_pixels(data.as_mut_slice()?, &wide[start..end], &casting_pool);
                                slot.upload(0, data)
                            })
                        })?;
                        let mut offset = 0;
                        for chunk in &io {
                            let values = chunk.output.as_slice()?;
                            assert_eq!(values, &reference[offset..offset + values.len()]);
                            offset += values.len();
                        }
                        assert_eq!(offset, reference.len());
                        record("host_cast_f64_pipeline", tile, streams, times)?;
                    }
                }
            }
        }
    }
    Ok(())
}
