use super::*;

#[derive(serde::Deserialize)]
struct PaperCase {
    shape: (usize, usize),
    radial_range: (f64, f64),
    pixels: PathBuf,
    mask: PathBuf,
    poni: String,
    frames: u32,
    bins: u32,
    errors: bool,
    repeats: usize,
    layout: String,
    result: PathBuf,
}

fn time_upload(
    batch: &mut Batch<'_, f64, u32>,
    kernel: &Kernel<f64, u32>,
    input: &(impl HostSlice<u32> + ?Sized),
    output: &mut PinnedHostSlice<f64>,
    expected: &[f64],
    repeats: usize,
) -> Result<Vec<f64>> {
    let timings = sample(repeats, || {
        batch.upload(0, input)?;
        batch.run(kernel)?;
        batch.download(output)?;
        Ok(batch.stream.synchronize()?)
    })?;
    assert!(
        output
            .as_slice()?
            .chunks_exact(expected.len())
            .all(|row| row == expected)
    );
    Ok(timings.into_iter().map(|ms| ms / 1e3).collect())
}

fn time_device(
    batch: &mut Batch<'_, f64, u32>,
    kernel: &Kernel<f64, u32>,
    input: &PinnedHostSlice<u32>,
    repeats: usize,
) -> Result<Vec<f64>> {
    let mut samples = Vec::with_capacity(repeats);
    for i in 0..repeats + 2 {
        batch.upload(0, input)?;
        let start = batch
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        batch.run(kernel)?;
        let end = batch
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        if i >= 2 {
            samples.push(f64::from(start.elapsed_ms(&end)?) / 1e3);
        }
    }
    Ok(samples)
}

#[test]
#[ignore = "paper fixture GPU benchmark; use benchmarks/comparison/gpu_compare.py"]
fn paper_native_cuda() -> Result<()> {
    let config = env::var_os("DIOPTRIN_PAPER_CUDA_CASE").ok_or("missing benchmark fixture")?;
    let case: PaperCase = serde_json::from_reader(File::open(config)?)?;
    if case.frames == 0 || case.bins == 0 || case.repeats < 3 {
        return Err("expected positive frames and bins and at least three repeats".into());
    }
    let (rows, columns) = case.shape;
    let layout = Layout::ALL
        .into_iter()
        .find(|layout| layout.name() == case.layout)
        .ok_or("invalid paper CUDA layout")?;
    let count = rows
        .checked_mul(columns)
        .ok_or("invalid frame dimensions")?;
    let input_bytes = fs::read(&case.pixels)?;
    if count.checked_mul(4) != Some(input_bytes.len()) {
        return Err("wrong uint32 pixel buffer length".into());
    }
    let pixels: Vec<_> = input_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    drop(input_bytes);
    let setup = Instant::now();
    let mask = Mask::from_vec(fs::read(&case.mask)?, rows, columns)?;
    let poni = Poni::load(&case.poni)?;
    let integrator = Integrator::new(Geometry::new(rows, columns, poni))
        .with_radial_unit(RadialUnit::TthDeg)
        .with_radial_range(case.radial_range.0, case.radial_range.1)
        .with_method(IntegrationMethod::PyfaiFull)
        .with_polarization_factor(0.965)
        .with_fast_math(false)
        .with_mask(mask);
    let cuda = Cuda::open()?.ok_or("CUDA or NVRTC unavailable")?;
    let plan = cuda.prepare_f64(&integrator, case.shape, case.bins)?;
    let kernel = plan.kernel_u32(layout, case.errors)?;
    let stride = 1 + usize::from(case.errors);
    let pattern = case.bins as usize * stride;
    let mut first = plan.batch_with_input(1, case.errors, Arc::clone(&cuda.stream))?;
    let mut first_output = vec![0.; pattern];
    first.upload(0, &pixels)?;
    first.run(&kernel)?;
    first.download(&mut first_output)?;
    cuda.stream.synchronize()?;
    let first_s = setup.elapsed().as_secs_f64();
    let device = cuda.context.name()?;
    let (free, _) = cuda.context.mem_get_info()?;
    let pixels_total = count
        .checked_mul(case.frames as usize)
        .ok_or("batch size overflow")?;
    let output_len = pattern
        .checked_mul(case.frames as usize)
        .ok_or("output length overflow")?;
    let output_bytes = output_len
        .checked_mul(size_of::<f64>())
        .ok_or("output bytes overflow")?;
    let bytes = pixels_total
        .checked_mul(size_of::<u32>())
        .and_then(|b| b.checked_add(output_bytes))
        .ok_or("device bytes overflow")?;
    let free = free_dedicated_bytes(&cuda.context).map_or(free, |dedicated| dedicated.min(free));
    if bytes > free.saturating_sub(GPU_RESERVE) || bytes > HOST_LIMIT / 2 {
        return Err("paper batch exceeds host or device memory budget".into());
    }
    let mut input = pinned::<u32>(&cuda.context, pixels_total)?;
    for frame in input.as_mut_slice()?.chunks_exact_mut(count) {
        frame.copy_from_slice(&pixels);
    }
    let ordinary = pixels.repeat(case.frames as usize);
    let mut output = pinned::<f64>(&cuda.context, output_len)?;
    let mut batch = plan.batch_with_input(case.frames, case.errors, Arc::clone(&cuda.stream))?;
    let mut records = Vec::new();
    {
        for (backend, host_memory, timings) in [
            (
                "dioptrin-cuda",
                "pinned",
                time_upload(
                    &mut batch,
                    &kernel,
                    &input,
                    &mut output,
                    &first_output,
                    case.repeats,
                )?,
            ),
            (
                "dioptrin-cuda-pageable",
                "pageable",
                time_upload(
                    &mut batch,
                    &kernel,
                    &ordinary,
                    &mut output,
                    &first_output,
                    case.repeats,
                )?,
            ),
        ] {
            records.push(json!({
                "backend": backend, "host_memory": host_memory, "layout": layout.name(),
                "device": device, "batch_seconds": timings,
            }));
        }
        records.push(json!({
            "backend": "dioptrin-cuda-kernels", "layout": layout.name(),
            "device": device, "batch_seconds": time_device(&mut batch, &kernel, &input, case.repeats)?,
        }));
    }
    let mut single_output = vec![0.; output_len];
    let samples = sample(case.repeats, || {
        for row in single_output.chunks_exact_mut(pattern) {
            first.upload(0, &pixels)?;
            first.run(&kernel)?;
            first.download(row)?;
            first.stream.synchronize()?;
        }
        Ok(())
    })?;
    assert!(
        single_output
            .chunks_exact(pattern)
            .all(|row| row == first_output)
    );
    records.push(json!({
        "backend": "dioptrin-cuda-single", "host_memory": "pageable", "layout": layout.name(),
        "device": device, "setup_first_s": first_s,
        "batch_seconds": samples.into_iter().map(|ms| ms / 1e3).collect::<Vec<_>>(),
    }));

    // Check both the measurement frame and a distinct frame with the same
    // mask. A multi-frame launch verifies that output slots stay ordered.
    let distinct: Vec<_> = pixels
        .iter()
        .enumerate()
        .map(|(i, &value)| value.saturating_add((i % 7) as u32))
        .collect();
    let mut validation = plan.batch_with_input(2, case.errors, Arc::clone(&cuda.stream))?;
    validation.upload(0, &pixels)?;
    validation.upload(count, &distinct)?;
    validation.run(&kernel)?;
    let mut checked = vec![0.; pattern * 2];
    validation.download(&mut checked)?;
    cuda.stream.synchronize()?;
    assert_eq!(checked[..pattern], first_output);
    let mut cpu_patterns = Vec::new();
    for (image, actual) in [&pixels, &distinct]
        .into_iter()
        .zip(checked.chunks_exact(pattern))
    {
        let wide: Vec<_> = image.iter().copied().map(f64::from).collect();
        let (radial, intensity, sigma) = if case.errors {
            integrator.integrate1d_with_errors_from_slice(&wide, rows, columns, case.bins as usize)
        } else {
            let (radial, intensity) =
                integrator.integrate1d_from_slice(&wide, rows, columns, case.bins as usize);
            (radial, intensity, Vec::new())
        };
        for (bin, row) in actual.chunks_exact(stride).enumerate() {
            for (&value, reference) in row
                .iter()
                .zip([intensity[bin]].into_iter().chain(sigma.get(bin).copied()))
            {
                assert!(
                    value.is_finite()
                        && (value - reference).abs() <= 1e-12 * reference.abs().max(1.0)
                        && (reference != 0.0 || value == 0.0),
                    "{} CUDA vs CPU bin {bin}: {value} vs {reference}",
                    layout.name()
                );
            }
        }
        cpu_patterns.push(json!({"radial": radial, "intensity": intensity, "sigma": sigma}));
    }
    let view = integrator.for_shape(rows, columns);
    let lut = view.ensure_lut1d(
        integrator
            .one_d_table_key(case.bins as usize)
            .ok_or("missing lookup key")?,
    );
    let gpu_patterns: Vec<_> = checked.chunks_exact(pattern).map(|frame| json!({
        "radial": lut.bins(), "intensity": frame.iter().step_by(stride).copied().collect::<Vec<_>>(),
        "sigma": if case.errors { frame[1..].iter().step_by(stride).copied().collect::<Vec<_>>() } else { Vec::new() },
    })).collect();
    let row =
        json!({"gpu_patterns": gpu_patterns, "cpu_patterns": cpu_patterns, "records": records});
    serde_json::to_writer(BufWriter::new(File::create(case.result)?), &row)?;
    Ok(())
}
