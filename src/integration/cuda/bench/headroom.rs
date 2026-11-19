use super::*;

#[test]
#[ignore = "CPU worker-count benchmark; use run.py --cpu-only"]
fn cpu_headroom() -> Result<()> {
    let mut out = BufWriter::new(File::create(env::var("DIOPTRIN_CUDA_OUT")?)?);
    let counts = thread_counts("14,16")?;
    let pools: Vec<_> = counts
        .iter()
        .map(|&n| ThreadPoolBuilder::new().num_threads(n).build())
        .collect::<std::result::Result<_, _>>()?;
    let repeats = env::var("DIOPTRIN_CUDA_REPEATS").map_or(Ok(7), |x| x.parse::<usize>())?;
    if repeats < 3 {
        return Err("use at least three timed repeats".into());
    }
    let cases = env::var("DIOPTRIN_CUDA_CASES").unwrap_or_else(|_| "pilatus,eiger".into());
    let batches = env::var("DIOPTRIN_CUDA_BATCHES").unwrap_or_else(|_| "1,64".into());
    let methods = env::var("DIOPTRIN_CUDA_METHODS")
        .unwrap_or_else(|_| "counting,pixel_split,pyfai_full".into());
    let host_limit: usize =
        env::var("DIOPTRIN_CUDA_HOST_BYTES").map_or(Ok(HOST_LIMIT), |x| x.parse())?;
    for name in cases.split(',') {
        let case = Case::load(name, true)?;
        log(
            &mut out,
            json!({"type":"case", "case":name, "shape":case.shape, "sha256":case.files}),
        )?;
        for (method, label) in METHODS {
            if !methods.split(',').any(|m| m == label) {
                continue;
            }
            eprintln!("{name} {label}");
            for (policy, fast, layout) in [
                ("fast_csc", true, SparseLayout::Csc),
                ("fast_csr", true, SparseLayout::Csr),
                ("strict_csr", false, SparseLayout::Csr),
            ] {
                let cpu = case
                    .strict(method)?
                    .with_fast_math(fast)
                    .with_sparse_layout(layout);
                for batch in batches.split(',') {
                    let frames = batch.parse::<usize>()?;
                    if frames == 0 || frames > host_limit / (case.pixels.len() * size_of::<f64>()) {
                        return Err(
                            "CPU-only batch exceeds the host memory budget or is empty".into()
                        );
                    }
                    let source = case.pixels.repeat(frames);
                    for errors in [false, true] {
                        let mut samples = vec![Vec::with_capacity(repeats); counts.len()];
                        for iteration in 0..repeats + 2 {
                            // Alternate worker-count order across paired runs.
                            for offset in 0..pools.len() {
                                let index = (iteration + offset) % pools.len();
                                let start = Instant::now();
                                cpu_batch(&cpu, case.shape, &source, frames, errors, &pools[index]);
                                if iteration >= 2 {
                                    samples[index].push(elapsed_ms(start));
                                }
                            }
                        }
                        for (threads, trial) in counts.iter().zip(samples) {
                            log(
                                &mut out,
                                json!({
                                    "type":"cpu", "case":name, "method":label, "layout":policy,
                                    "threads":threads, "batch":batch, "frames":frames,
                                    "errors":errors, "samples_ms_per_batch":trial,
                                }),
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
