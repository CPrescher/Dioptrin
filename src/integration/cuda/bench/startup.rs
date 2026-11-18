use super::*;

#[test]
#[ignore = "fresh-process benchmark; use docs/benchmarks/2026-11-18-cuda/startup.py"]
fn first_pattern() -> Result<()> {
    let case_name = env::var("DIOPTRIN_CUDA_STARTUP_CASE")?;
    let method_name = env::var("DIOPTRIN_CUDA_STARTUP_METHOD")?;
    let selection = env::var("DIOPTRIN_CUDA_STARTUP_BACKEND")?;
    let method = METHODS
        .iter()
        .find_map(|&(method, name)| (name == method_name).then_some(method))
        .ok_or("unknown startup method")?;

    let start = Instant::now();
    let case = Case::load(&case_name, false)?;
    let fixture_ms = elapsed_ms(start);
    let work_start = Instant::now();
    let mut integrator = case.strict(method)?;
    let mut phases = if selection == "cuda" {
        let stage = Instant::now();
        let cuda = Cuda::open()?.ok_or("CUDA driver, GPU or NVRTC unavailable")?;
        let context_and_compilation_ms = elapsed_ms(stage);
        let stage = Instant::now();
        let plan = cuda.prepare(&integrator, case.shape, BINS)?;
        let prepare_ms = elapsed_ms(stage);

        let stage = Instant::now();
        let layout_name = env::var("DIOPTRIN_CUDA_STARTUP_LAYOUT")?;
        let layout = Layout::ALL
            .into_iter()
            .find(|layout| layout.name() == layout_name)
            .ok_or("unknown startup layout")?;
        let kernel = plan.kernel(Accumulation::Compensated, layout, true)?;
        let mut batch = plan.batch(1, true)?;
        let casting = ThreadPoolBuilder::new().num_threads(8).build()?;
        let memory = env::var("DIOPTRIN_CUDA_STARTUP_MEMORY")?;
        let report = |allocate_ms, first_frame_ms| {
            emit(json!({
                "fixture_ms": fixture_ms, "after_fixture_ms": elapsed_ms(work_start),
                "driver_context_ms": context_and_compilation_ms - cuda.compile_ms - cuda.module_ms,
                "nvrtc_ms": cuda.compile_ms, "module_load_ms": cuda.module_ms,
                "prepare_ms": prepare_ms, "table_upload_ms": plan.upload_ms,
                "allocate_ms": allocate_ms, "first_frame_ms": first_frame_ms,
            }))
        };
        // Signal the parent before destroying either host staging buffer or
        // the CUDA context so process teardown stays outside the interval.
        return if memory == "pinned" {
            let mut input = pinned::<f32>(&cuda.context, case.pixels.len())?;
            let mut output = pinned::<f64>(&cuda.context, batch.output.len())?;
            let allocate_ms = elapsed_ms(stage);
            let start = Instant::now();
            cast_pixels(input.pixels_mut()?, &case.pixels, &casting);
            batch.upload(0, &input)?;
            batch.run(&kernel)?;
            batch.download(&mut output)?;
            cuda.stream.synchronize()?;
            black_box(output.as_slice()?);
            report(allocate_ms, elapsed_ms(start))
        } else if memory == "pageable" {
            let mut input = vec![0.0; case.pixels.len()];
            let mut output = vec![0.0; batch.output.len()];
            let allocate_ms = elapsed_ms(stage);
            let start = Instant::now();
            cast_pixels(&mut input, &case.pixels, &casting);
            batch.upload(0, &input)?;
            batch.run(&kernel)?;
            batch.download(&mut output)?;
            cuda.stream.synchronize()?;
            black_box(&output);
            report(allocate_ms, elapsed_ms(start))
        } else {
            Err("unknown startup host memory mode".into())
        };
    } else {
        let threads: usize = env::var("DIOPTRIN_CUDA_STARTUP_THREADS")?.parse()?;
        match selection.as_str() {
            "fast_csc" => integrator.set_fast_math(true),
            "fast_csr" => {
                integrator.set_fast_math(true);
                integrator.set_sparse_layout(SparseLayout::Csr);
            }
            "strict_csr" => {}
            _ => return Err("unknown CPU startup policy".into()),
        }
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        let setup_ms = elapsed_ms(work_start);
        let stage = Instant::now();
        let pattern = pool.install(|| {
            integrator.integrate1d_with_errors_from_slice(
                &case.pixels,
                case.shape.0,
                case.shape.1,
                BINS as usize,
            )
        });
        black_box(&pattern);
        json!({"cpu_pool_and_integrator_ms": setup_ms, "first_call_ms": elapsed_ms(stage)})
    };
    phases["fixture_ms"] = json!(fixture_ms);
    phases["after_fixture_ms"] = json!(elapsed_ms(work_start));
    emit(phases)
}

fn emit(phases: serde_json::Value) -> Result<()> {
    let mut output = std::io::stdout().lock();
    writeln!(output, "DIOPTRIN_FIRST_PATTERN {phases}")?;
    output.flush()?;
    Ok(())
}
