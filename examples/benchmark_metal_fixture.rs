//! Standalone Metal benchmark fixture and current, prepared CPU baselines.
//! See docs/benchmarks/2026-11-10-metal/README.md. No production GPU dependency.
use dioptrin::{
    frame::Frame,
    geometry::Geometry,
    integration::{IntegrationMethod, Integrator, RadialUnit, SparseLayout},
    lut::{self, Lut1d},
    mask::Mask,
    poni::Poni,
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    hint::black_box,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

fn write_values<T>(path: &Path, values: &[T]) -> std::io::Result<()> {
    // Only primitive numeric slices on the little-endian benchmark host.
    const { assert!(cfg!(target_endian = "little")) };
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(bytes)?;
    file.flush()
}

fn stats(mut times: Vec<f64>) -> Value {
    times.sort_by(f64::total_cmp);
    json!({"median_ms": times[times.len()/2], "min_ms": times[0],
        "p95_ms": times[((times.len()-1) as f64 * 0.95).ceil() as usize],
        "samples_ms": times})
}

fn edge_case() -> (Frame, Lut1d, Vec<f64>, Mask) {
    // Independent hand-checkable rows: empty, masked NaN, fractional weights,
    // negative counts, large counts, cancellation, FP32 input rounding, zero.
    let data = vec![
        f64::NAN,
        4.,
        8.,
        -3.,
        -5.,
        4_000_000_000.,
        2.,
        100_000_000.,
        1.,
        -100_000_000.,
        16_777_217.,
        -16_777_216.,
        0.,
    ];
    let lut = Lut1d {
        row_ptr: vec![0, 0, 1, 3, 5, 7, 10, 12, 13],
        col_idx: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        values: Some(vec![1., 0.25, 0.75, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        hist_min: 0.,
        hist_step: 1.,
    };
    let mask = Mask {
        data: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dim1: 1,
        dim2: 13,
    };
    (
        Frame {
            name: "edge".into(),
            data,
            dim1: 1,
            dim2: 13,
        },
        lut,
        vec![1.; 13],
        mask,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(
            "OUT edge | OUT IMAGE PONI counting|pixel_split BINS [MASK|-] [SAMPLES]".into(),
        );
    }
    let out = Path::new(&args[1]);
    fs::create_dir_all(out)?;
    let edge = args[2] == "edge";
    if !edge && args.len() < 6 {
        return Err("expected OUT IMAGE PONI METHOD BINS [MASK|-] [SAMPLES]".into());
    }
    let method = if !edge && args[4] == "counting" {
        IntegrationMethod::Counting
    } else {
        IntegrationMethod::PixelSplit
    };
    if !edge && !matches!(args[4].as_str(), "counting" | "pixel_split") {
        return Err("unsupported method".into());
    }
    let samples: usize = args.get(7).map(|v| v.parse()).transpose()?.unwrap_or(31);
    if samples == 0 {
        return Err("samples must be positive".into());
    }
    let mut geometry = None;
    let start = Instant::now();
    let (frame, table, norm, mask) = if edge {
        edge_case()
    } else {
        let frame = Frame::from_tiff(args[2].clone())?;
        let poni = Poni::load(&args[3])?;
        assert!(
            poni.pixel1 > 0. && poni.pixel2 > 0.,
            "explicit pixel sizes required"
        );
        let g = Geometry::new(frame.dim1, frame.dim2, poni);
        let tth = g.calc_tth_all();
        let radial = RadialUnit::TthDeg.convert_from_tth(&tth, g.poni.wavelength);
        let bins: usize = args[5].parse()?;
        if bins == 0 {
            return Err("bins must be positive".into());
        }
        let table = match method {
            IntegrationMethod::Counting => lut::build_lut1d_csr(&radial, bins),
            _ => lut::build_lut1d_ps_csr(&g, &radial, bins, RadialUnit::TthDeg),
        };
        let sa = g.calc_solid_angle();
        let pol = g.calc_polarization(0.965, &tth, &g.calc_azimuth_all());
        let norm: Vec<_> = pol.iter().zip(&sa).map(|(p, s)| p * s).collect();
        let mut mask = if let Some(path) = args.get(6).filter(|s| s.as_str() != "-") {
            Mask::from_tiff(path.clone())?
        } else {
            Mask {
                data: vec![0; frame.data.len()],
                dim1: frame.dim1,
                dim2: frame.dim2,
            }
        };
        mask.assert_compatible(frame.dim1, frame.dim2);
        // Real detector fixtures: exclude negative/nonfinite invalid pixels.
        for (&x, m) in frame.data.iter().zip(&mut mask.data) {
            if !x.is_finite() || x < 0. {
                *m = 1;
            }
        }
        geometry = Some(g);
        (frame, table, norm, mask)
    };
    let fixture_setup_ms = start.elapsed().as_secs_f64() * 1000.;
    let bins = table.n_points();
    let variance: Vec<_> = frame.data.iter().map(|v| v.max(0.)).collect();
    let (radial, reference, reference_std) =
        lut::integrate1d_with_errors(&frame.data, &variance, &table, Some(&mask.data), &norm);
    drop(variance);
    if edge {
        let expected = [0., 0., 7., -4., 2_000_000_001., 1. / 3., 0.5, 0.];
        for (a, b) in reference.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((reference_std[2] - 4.75_f64.sqrt()).abs() < 1e-12);
    }
    // Bake the fixed mask into exactly the same CSR row order; cache the f64
    // denominator before converting weights to f32. No geometry approximation.
    let mut ptr = vec![0_u32];
    let mut indices = Vec::<u32>::new();
    let mut weights = Vec::<f32>::new();
    let mut bin_norm = vec![0_f64; bins];
    for (b, denominator) in bin_norm.iter_mut().enumerate() {
        for k in table.row_ptr[b]..table.row_ptr[b + 1] {
            let p = table.col_idx[k];
            if mask.data[p] != 0 {
                continue;
            }
            let w = table.values.as_ref().map_or(1., |v| v[k]);
            indices.push(u32::try_from(p)?);
            weights.push(w as f32);
            *denominator += w * norm[p];
        }
        ptr.push(u32::try_from(indices.len())?);
    }
    write_values(&out.join("rows.u32"), &ptr)?;
    write_values(&out.join("pixels.u32"), &indices)?;
    write_values(&out.join("weights.f32"), &weights)?;
    write_values(&out.join("normalization.f64"), &bin_norm)?;
    write_values(&out.join("signal.f64"), &frame.data)?;
    write_values(
        &out.join("signal.f32"),
        &frame.data.iter().map(|&x| x as f32).collect::<Vec<_>>(),
    )?;
    let mut cpu = Vec::new();
    if let Some(g) = geometry {
        for (label, layout, fast) in [
            ("default_csc", SparseLayout::Csc, true),
            ("fast_csr", SparseLayout::Csr, true),
            ("strict_csr", SparseLayout::Csr, false),
        ] {
            let mut ai = Integrator::new(g.clone())
                .with_method(method)
                .with_radial_unit(RadialUnit::TthDeg)
                .with_polarization_factor(0.965)
                .with_sparse_layout(layout)
                .with_fast_math(fast);
            ai.set_mask(Mask {
                data: mask.data.clone(),
                dim1: mask.dim1,
                dim2: mask.dim2,
            });
            for threads in [1, 4, 10, 14] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()?;
                pool.install(|| {
                    let setup = Instant::now();
                    let (_, y, e) = ai.integrate1d_with_errors(&frame, bins);
                    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
                    for ((a, b), (c, d)) in
                        y.iter().zip(&reference).zip(e.iter().zip(&reference_std))
                    {
                        assert!((a - b).abs() <= 1e-10 * b.abs().max(1.), "CPU LUT mismatch");
                        assert!((c - d).abs() <= 1e-10 * d.abs().max(1.), "CPU std mismatch");
                    }
                    for errors in [false, true] {
                        let run = || {
                            if errors {
                                black_box(ai.integrate1d_with_errors(black_box(&frame), bins));
                            } else {
                                black_box(ai.integrate1d(black_box(&frame), bins));
                            }
                        };
                        let warm = Instant::now();
                        while warm.elapsed().as_millis() < 100 {
                            run();
                        }
                        let mut times = Vec::new();
                        for _ in 0..samples {
                            let t = Instant::now();
                            run();
                            times.push(t.elapsed().as_secs_f64() * 1000.);
                        }
                        cpu.push(json!({"layout":label,"threads":threads,"errors":errors,
                            "first_call_ms":setup_ms,"timing":stats(times)}));
                    }
                });
            }
        }
    }
    let metadata = json!({"image":args[2], "method":if edge {"edge"} else {&args[4]},
        "height":frame.dim1,"width":frame.dim2,"bins":bins,"entries":indices.len(),
        "masked_pixels":mask.data.iter().filter(|&&m| m != 0).count(),
        "unmasked_i32_max_pixels":frame.data.iter().zip(&mask.data)
            .filter(|(x,m)| **m == 0 && **x == i32::MAX as f64).count(),
        "fixture_setup_ms":fixture_setup_ms,"weighted":table.values.is_some(),
        "max_unmasked_signal":frame.data.iter().zip(&mask.data).filter(|(_,m)| **m==0)
            .map(|(x,_)| *x).fold(f64::NEG_INFINITY, f64::max),
        "reference_intensity":reference,"reference_std":reference_std,"radial":radial,"cpu":cpu});
    fs::write(
        out.join("fixture.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    println!(
        "{}: {}x{}, {bins} bins, {} entries",
        args[2],
        frame.dim1,
        frame.dim2,
        indices.len()
    );
    Ok(())
}
