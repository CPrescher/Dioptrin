//! Compare cached execution against the corrected ordinary paths on one fixture.
//! RAYON_NUM_THREADS=8 cargo run --release --example benchmark_consolidated
use dioptrin::{
    frame::Frame,
    geometry::Geometry,
    integration::{
        FixedCsr1dInputPlan, FixedCsr1dPlan, InputOrder, IntegrationGrid2d, IntegrationMethod,
        Integrator, RadialUnit,
    },
    lut,
    mask::Mask,
    poni::Poni,
};
use serde_json::json;
use std::{hint::black_box, sync::Arc, time::Instant};

fn median_ms<T>(mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let mut times = Vec::new();
    for _ in 0..15 {
        let start = Instant::now();
        black_box(f());
        times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

fn main() {
    let frame = Frame::from_tiff("test_data/CeO2_Pilatus1M.tif".into()).unwrap();
    let geometry = Geometry::new(
        frame.dim1,
        frame.dim2,
        Poni::load("test_data/v1.poni").unwrap(),
    );
    let mask: Vec<_> = frame.data.iter().map(|v| u8::from(*v < 0.)).collect();
    let pixels: Vec<u32> = frame.data.iter().map(|v| v.max(0.) as u32).collect();
    let data: Vec<_> = pixels.iter().map(|&v| v as f64).collect();
    let norm = geometry.calc_solid_angle();
    let table =
        lut::build_lut1d_pyfai_csr(&geometry, RadialUnit::TthDeg, 1000, None, None).unwrap();
    let csr =
        Arc::new(FixedCsr1dPlan::new(table.clone(), norm.clone(), Some(mask.clone())).unwrap());
    let typed =
        FixedCsr1dInputPlan::new(csr.clone(), (frame.dim1, frame.dim2), InputOrder::Detector)
            .unwrap();
    let engine = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_unit(RadialUnit::TthDeg)
        .with_mask(Mask {
            data: mask.clone(),
            dim1: frame.dim1,
            dim2: frame.dim2,
        });
    let grid = IntegrationGrid2d::new(
        300,
        180,
        (0., 35.),
        (-std::f64::consts::PI, std::f64::consts::PI),
    )
    .unwrap();
    let start = Instant::now();
    let cake = engine
        .prepare_fixed2d_on_grid((frame.dim1, frame.dim2), &grid)
        .unwrap();
    let cake_setup_ms = start.elapsed().as_secs_f64() * 1000.;
    let mut results = Vec::new();
    for errors in [false, true] {
        let reference = if errors {
            lut::integrate1d_with_errors(&data, &data, &table, Some(&mask), &norm)
        } else {
            let (x, y) = lut::integrate1d(&data, &table, Some(&mask), &norm);
            (x, y, Vec::new())
        };
        for result in [
            csr.integrate_frame(&data, errors).unwrap(),
            typed.integrate_u32(&pixels, errors).unwrap(),
            typed.integrate_u32_parallel(&pixels, errors).unwrap(),
        ] {
            same(&reference.0, &result.radial);
            same(&reference.1, &result.intensity);
            if errors {
                same(&reference.2, result.errors.as_ref().unwrap());
            }
        }
        let ordinary = || {
            let raw = engine
                .integrate2d_raw_on_grid_from_slice(&data, frame.dim1, frame.dim2, &grid, errors)
                .unwrap();
            let intensity: Vec<_> = raw
                .sum_signal
                .iter()
                .zip(&raw.sum_norm)
                .map(|(s, n)| if *n == 0. { 0. } else { s / n })
                .collect();
            let sigma = raw.sum_variance.map(|v| {
                v.iter()
                    .zip(&raw.sum_norm)
                    .map(|(v, n)| {
                        if *n == 0. {
                            0.
                        } else {
                            (v / (n * n)).max(0.).sqrt()
                        }
                    })
                    .collect::<Vec<_>>()
            });
            (intensity, sigma)
        };
        let expected = ordinary();
        let actual = cake.integrate_frame(&data, errors).unwrap();
        same(&expected.0, &actual.0);
        if errors {
            same(expected.1.as_ref().unwrap(), actual.1.as_ref().unwrap());
        }
        results.push(json!({
            "errors": errors,
            "strict_f64_csr_ms": median_ms(|| csr.integrate_frame(&data, errors).unwrap()),
            "strict_u32_serial_ms": median_ms(|| typed.integrate_u32(&pixels, errors).unwrap()),
            "strict_u32_parallel_ms": median_ms(|| typed.integrate_u32_parallel(&pixels, errors).unwrap()),
            "ordinary_grid2d_ms": median_ms(ordinary),
            "cached_grid2d_ms": median_ms(|| cake.integrate_frame(&data, errors).unwrap()),
            "bitwise_equal": true,
        }));
    }
    println!("{}", serde_json::to_string_pretty(&json!({
        "fixture": "CeO2_Pilatus1M.tif", "poni": "v1.poni", "shape": [frame.dim1, frame.dim2],
        "threads": rayon::current_num_threads(), "trials": 15, "warmups": 3,
        "cake_setup_ms": cake_setup_ms, "results": results,
        "scope": "Warm in-memory execution, decoding/conversion excluded; negative TIFF pixels masked. Both paths use corrected main LUTs."
    })).unwrap());
}
