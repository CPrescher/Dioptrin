use super::float32::Float32;
use super::*;
use crate::frame::Pixel;

fn integrate<P: Pixel>(
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

fn batch<P: Pixel>(
    cpu: &Integrator,
    shape: (usize, usize),
    data: &[P],
    errors: bool,
    pool: &ThreadPool,
) {
    pool.install(|| {
        data.par_chunks_exact(shape.0 * shape.1).for_each(|frame| {
            black_box(integrate(cpu, shape, frame, errors));
        });
    });
}

#[test]
#[ignore = "paired CPU input precision benchmark; use run.py --cpu-inputs"]
fn cpu_inputs() -> Result<()> {
    let mut out = BufWriter::new(File::create(env::var("DIOPTRIN_CUDA_OUT")?)?);
    let pools = thread_counts("14")?
        .into_iter()
        .map(|threads| ThreadPoolBuilder::new().num_threads(threads).build())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let repeats = env::var("DIOPTRIN_CUDA_REPEATS").map_or(Ok(7), |s| s.parse::<usize>())?;
    if repeats < 3 {
        return Err("use at least three timed repeats".into());
    }
    let cases = env::var("DIOPTRIN_CUDA_CASES").unwrap_or_else(|_| "pilatus,eiger".into());
    let methods = env::var("DIOPTRIN_CUDA_METHODS")
        .unwrap_or_else(|_| "counting,pixel_split,pyfai_full".into());
    let batches = env::var("DIOPTRIN_CUDA_BATCHES").unwrap_or_else(|_| "1,64".into());
    let host_limit =
        env::var("DIOPTRIN_CUDA_HOST_BYTES").map_or(Ok(HOST_LIMIT), |s| s.parse::<usize>())?;
    for name in cases.split(',') {
        let case = Case::load(name, true)?;
        let input32: Vec<_> = case.pixels.iter().map(|&x| x as f32).collect();
        log(
            &mut out,
            json!({"type": "case", "case": name, "shape": case.shape, "sha256": case.files}),
        )?;
        for (method, label) in METHODS {
            if !methods.split(',').any(|name| name == label) {
                continue;
            }
            eprintln!("{name} {label}");
            let policies = [
                ("fast_csc", case.strict(method)?.with_fast_math(true)),
                (
                    "fast_csr",
                    case.strict(method)?
                        .with_fast_math(true)
                        .with_sparse_layout(SparseLayout::Csr),
                ),
                ("strict_csr", case.strict(method)?),
            ];
            for count in batches.split(',') {
                let frames: usize = count.parse()?;
                let bytes_per_frame = case.pixels.len() * (size_of::<f64>() + size_of::<f32>());
                if frames == 0 || frames >= host_limit / bytes_per_frame {
                    return Err("paired CPU input buffers exceed host budget or are empty".into());
                }
                let data64 = case.pixels.repeat(frames);
                let data32 = input32.repeat(frames);
                for (layout, cpu) in &policies {
                    let view = cpu.for_shape(case.shape.0, case.shape.1);
                    let lut = view.ensure_lut1d(cpu.one_d_table_key(BINS as usize).unwrap());
                    let plan32 = Float32::new(&lut, cpu, &view.normalization())?;
                    for errors in [false, true] {
                        let (height, width) = case.shape;
                        let (reference, rounded, all_f32) = pools[0].install(|| {
                            let reference = integrate(cpu, case.shape, &case.pixels, errors);
                            let public = if errors {
                                cpu.integrate1d_with_errors_from_slice(
                                    &case.pixels,
                                    height,
                                    width,
                                    BINS as usize,
                                )
                            } else {
                                let (bins, intensity) = cpu.integrate1d_from_slice(
                                    &case.pixels,
                                    height,
                                    width,
                                    BINS as usize,
                                );
                                (bins, intensity, Vec::new())
                            };
                            assert_eq!(reference, public);
                            (
                                reference,
                                integrate(cpu, case.shape, &input32, errors),
                                plan32.integrate(&input32, errors),
                            )
                        });
                        assert_eq!(reference.0, all_f32.0);
                        assert_eq!(reference.1.len(), all_f32.1.len());
                        assert_eq!(reference.2.len(), all_f32.2.len());
                        for pool in &pools {
                            let mut samples =
                                std::array::from_fn::<_, 3, _>(|_| Vec::with_capacity(repeats));
                            for i in 0..repeats + 2 {
                                for offset in 0..samples.len() {
                                    let mode = (i + offset) % samples.len();
                                    let start = Instant::now();
                                    match mode {
                                        0 => batch(cpu, case.shape, &data64, errors, pool),
                                        1 => batch(cpu, case.shape, &data32, errors, pool),
                                        _ => pool.install(|| {
                                            data32.par_chunks_exact(case.pixels.len()).for_each(
                                                |frame| {
                                                    black_box(plan32.integrate(frame, errors));
                                                },
                                            );
                                        }),
                                    }
                                    if i >= 2 {
                                        samples[mode].push(elapsed_ms(start));
                                    }
                                }
                            }
                            log(
                                &mut out,
                                json!({
                                    "type": "paired_input", "case": name, "method": label,
                                    "frames": frames, "layout": layout, "threads": pool.current_num_threads(),
                                    "errors": errors, "f64_ms_per_batch": samples[0],
                                    "f32_ms_per_batch": samples[1],
                                    "all_f32_ms_per_batch": samples[2],
                                    "intensity_error": accuracy(rounded.1.iter().copied(), &reference.1),
                                    "std_error": errors.then(|| accuracy(rounded.2.iter().copied(), &reference.2)),
                                    "all_f32_intensity_error": accuracy(all_f32.1.iter().copied().map(f64::from), &reference.1),
                                    "all_f32_std_error": errors.then(|| accuracy(all_f32.2.iter().copied().map(f64::from), &reference.2)),
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
