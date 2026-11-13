//! One `Integrator` shared by concurrent calls with images of different shapes.
//!
//! Geometry arrays and lookup tables are cached per integrator, not per
//! shape. Every call must still integrate with tables for its own image, and
//! give the result a fresh integrator gives, whatever the other threads do.
use std::sync::{Arc, mpsc};

use rayon::prelude::*;

use super::{IntegrationGrid2d, IntegrationMethod, Integrator, RadialUnit};
use crate::batch::{
    Integration1D, Integration2D, WorkItem, process_files_streaming, process_queue_streaming,
};
use crate::frame::{Frame, TiffFrame};
use crate::geometry::Geometry;
use crate::mask::Mask;
use crate::poni::Poni;

/// Includes two shapes with the same pixel count, which a length check alone
/// cannot tell apart.
const SHAPES: [(usize, usize); 3] = [(48, 36), (20, 30), (36, 48)];
const METHODS: [IntegrationMethod; 5] = [
    IntegrationMethod::Counting,
    IntegrationMethod::PixelSplit,
    IntegrationMethod::PyfaiFull,
    IntegrationMethod::Supersampled { n_ss: 2 },
    IntegrationMethod::Histogram,
];
const N_POINTS: usize = 25;
const N_RADIAL: usize = 12;
const N_AZIMUTHAL: usize = 6;

fn integrator(method: IntegrationMethod) -> Integrator {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 40.0;
    poni.poni1 = 9.3;
    poni.poni2 = 13.7;
    poni.wavelength = 1e-10;
    Integrator::new(Geometry::new(2, 2, poni)).with_method(method)
}

fn frame((dim1, dim2): (usize, usize)) -> Frame {
    Frame {
        name: format!("{dim1}x{dim2}"),
        data: (0..dim1 * dim2)
            .map(|p| ((p * 37 + dim2) % 101) as f64 + 1.0)
            .collect(),
        dim1,
        dim2,
    }
}

type Pattern = (Vec<f64>, Vec<f64>);
/// A frame with its 1D pattern and 2D cake from a fresh integrator.
type Reference = (Frame, (Pattern, Option<Vec<f64>>));

/// 1D pattern and, except for Histogram (which has no 2D binning), 2D cake.
fn integrate(integrator: &Integrator, frame: &Frame) -> (Pattern, Option<Vec<f64>>) {
    let pattern = integrator.integrate1d_from_slice(&frame.data, frame.dim1, frame.dim2, N_POINTS);
    let cake = (!matches!(integrator.method(), IntegrationMethod::Histogram)).then(|| {
        integrator.integrate2d_from_slice(
            &frame.data,
            frame.dim1,
            frame.dim2,
            N_RADIAL,
            N_AZIMUTHAL,
        )
    });
    (pattern, cake)
}

/// Bit-identical, except for the 2D pixel-split table: its builder does not
/// fix the order of the pixels within a bin, so two builds of the same table
/// sum in different orders and differ in the last digits.
fn assert_same(method: IntegrationMethod, actual: &[f64], expected: &[f64], what: &str) {
    if matches!(method, IntegrationMethod::PixelSplit) {
        assert_eq!(actual.len(), expected.len(), "{method:?} {what}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= 1e-12 * e.abs(),
                "{method:?} {what}: {a} vs {e}"
            );
        }
    } else {
        assert_eq!(actual, expected, "{method:?} {what}");
    }
}

fn assert_integrates_like_fresh(shared: &Integrator, (frame, (pattern, cake)): &Reference) {
    let (actual_pattern, actual_cake) = integrate(shared, frame);
    assert_eq!(&actual_pattern, pattern, "{:?} 1D", shared.method());
    assert_eq!(actual_cake.is_some(), cake.is_some());
    if let (Some(actual), Some(expected)) = (actual_cake, cake) {
        assert_same(shared.method(), &actual, expected, &frame.name);
    }
}

/// Each shape integrated on its own fresh integrator.
fn references(method: IntegrationMethod) -> Vec<Reference> {
    SHAPES
        .iter()
        .map(|&shape| {
            let frame = frame(shape);
            let expected = integrate(&integrator(method), &frame);
            (frame, expected)
        })
        .collect()
}

#[test]
fn histogram_preparation_warms_geometry_without_a_lookup_table() {
    let ai = integrator(IntegrationMethod::Histogram);
    assert!(ai.tth().is_empty());
    ai.prepare1d(SHAPES[0].0, SHAPES[0].1, N_POINTS);
    assert_eq!(ai.tth().len(), SHAPES[0].0 * SHAPES[0].1);
    assert!(ai.cache_1d.read().unwrap().lut.is_none());
    let prepared = ai.geometry_for(SHAPES[0].0, SHAPES[0].1);
    ai.integrate1d(&frame(SHAPES[0]), N_POINTS);
    assert!(Arc::ptr_eq(
        &prepared,
        &ai.geometry_for(SHAPES[0].0, SHAPES[0].1)
    ));
}

#[test]
fn radial_unit_builder_refreshes_an_already_used_integrator() {
    for method in METHODS {
        let frame = frame(SHAPES[0]);
        let mut ai = integrator(method);
        ai.integrate1d(&frame, N_POINTS);
        for unit in [
            RadialUnit::TthDeg,
            RadialUnit::QNm,
            RadialUnit::QAngstrom,
            RadialUnit::TthRad,
        ] {
            ai = ai.with_radial_unit(unit);
            let fresh = integrator(method).with_radial_unit(unit);
            let actual = ai.integrate1d_with_errors(&frame, N_POINTS);
            let expected = fresh.integrate1d_with_errors(&frame, N_POINTS);
            assert_eq!(actual.0, expected.0);
            assert_same(method, &actual.1, &expected.1, "unit-builder intensity");
            assert_same(method, &actual.2, &expected.2, "unit-builder errors");
        }
    }
}

#[test]
fn incompatible_mask_rejects_a_shape_before_replacing_geometry() {
    let shape = SHAPES[0];
    let ai = integrator(IntegrationMethod::Counting).with_mask(Mask {
        data: vec![0; shape.0 * shape.1],
        dim1: shape.0,
        dim2: shape.1,
    });
    let expected = ai.integrate1d(&frame(shape), N_POINTS);
    let prepared = ai.geometry_for(shape.0, shape.1);
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ai.integrate1d(&frame(SHAPES[1]), N_POINTS)
    }));
    assert!(failed.is_err());
    assert!(Arc::ptr_eq(&prepared, &ai.state.read().unwrap()));
    assert_eq!(ai.integrate1d(&frame(shape), N_POINTS), expected);
}

#[test]
fn typed_tiff_threads_match_float_references_across_shapes() {
    use std::{
        fs::File,
        time::{SystemTime, UNIX_EPOCH},
    };
    use tiff::encoder::{TiffEncoder, colortype};
    let path = std::env::temp_dir().join(format!(
        "dioptrin-mixed-typed-{}-{}.tif",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let typed: Vec<_> = SHAPES
        .into_iter()
        .map(|(h, w)| {
            let pixels: Vec<u32> = (0..h * w)
                .map(|p| ((p * 37 + w) % 100_003) as u32)
                .collect();
            TiffEncoder::new(File::create(&path).unwrap())
                .unwrap()
                .write_image::<colortype::Gray32>(w as u32, h as u32, &pixels)
                .unwrap();
            let mut image = TiffFrame::default();
            image.reload(path.to_str().unwrap()).unwrap();
            assert!(image.legacy().is_none());
            assert!(matches!(
                image.pixels,
                tiff::decoder::DecodingResult::U32(_)
            ));
            image
        })
        .collect();
    std::fs::remove_file(path).unwrap();
    let angles = [None, Some((-2.8, 0.9))];
    for method in METHODS.into_iter().take(4) {
        for fast in [false, true] {
            let new = || integrator(method).with_fast_math(fast);
            let references: Vec<_> = typed
                .iter()
                .map(|image| {
                    let frame = image.to_frame();
                    let ai = new();
                    let pattern = ai.integrate1d_with_errors(&frame, N_POINTS);
                    let cakes: Vec<_> = angles
                        .into_iter()
                        .map(|range| {
                            if let Some((start, step)) = range {
                                ai.integrate2d_with_azimuthal_range_and_errors(
                                    &frame,
                                    N_RADIAL,
                                    N_AZIMUTHAL,
                                    start,
                                    step,
                                )
                            } else {
                                ai.integrate2d_with_errors(&frame, N_RADIAL, N_AZIMUTHAL)
                            }
                        })
                        .collect();
                    (pattern, cakes)
                })
                .collect();
            let shared = new();
            (0..36).into_par_iter().for_each(|i| {
                let k = i % typed.len();
                let a = (i / typed.len()) % angles.len();
                let actual = shared.integrate_tiff1d::<true>(&typed[k], N_POINTS);
                let expected = &references[k].0;
                assert_eq!(actual.0, expected.0);
                assert_same(method, &actual.1, &expected.1, "typed 1D intensity");
                assert_same(method, &actual.2, &expected.2, "typed 1D errors");
                let actual =
                    shared.integrate_tiff2d::<true>(&typed[k], N_RADIAL, N_AZIMUTHAL, angles[a]);
                let expected = &references[k].1[a];
                assert_same(method, &actual.0, &expected.0, "typed 2D intensity");
                assert_same(method, &actual.1, &expected.1, "typed 2D errors");
            });
        }
    }
}

#[test]
fn threads_sharing_an_integrator_integrate_each_shape_with_its_own_tables() {
    for method in METHODS {
        let references = references(method);
        let shared = integrator(method);
        std::thread::scope(|scope| {
            for thread in 0..8 {
                let (references, shared) = (&references, &shared);
                scope.spawn(move || {
                    for i in 0..30 {
                        let reference = &references[(thread + i) % references.len()];
                        assert_integrates_like_fresh(shared, reference);
                    }
                });
            }
        });
    }
}

/// Frames of one Rayon job can be stolen by a thread that is itself waiting
/// inside another frame's integration, so no lock may be held across the work.
#[test]
fn one_rayon_pool_integrates_mixed_shapes_without_deadlock() {
    for method in METHODS {
        let references = references(method);
        let shared = integrator(method);
        (0..96).into_par_iter().for_each(|i| {
            assert_integrates_like_fresh(&shared, &references[i % references.len()]);
        });
    }
}

#[test]
fn concurrent_shapes_preserve_true_ranges_errors_and_supplied_variances() {
    let close = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected) {
            assert!(
                a == e || (a - e).abs() <= 1e-12 * (1. + e.abs()),
                "{a} != {e}"
            );
        }
    };
    for method in METHODS.into_iter().take(4) {
        for fast in [false, true] {
            for unit in [RadialUnit::TthDeg, RadialUnit::QNm] {
                let new = || {
                    integrator(method)
                        .with_fast_math(fast)
                        .with_radial_unit(unit)
                };
                let evaluate = |ai: &Integrator, frame: &Frame| {
                    let range = match frame.dim1 {
                        20 => (2.5, 2.5 + std::f64::consts::FRAC_PI_2),
                        36 => (-3. * std::f64::consts::PI, -std::f64::consts::PI),
                        _ => (0., std::f64::consts::TAU),
                    };
                    let pattern = ai.integrate1d_with_errors_from_slice(
                        &frame.data,
                        frame.dim1,
                        frame.dim2,
                        N_POINTS,
                    );
                    let cake = ai
                        .integrate2d_on_azimuthal_range_from_slice(
                            &frame.data,
                            frame.dim1,
                            frame.dim2,
                            N_RADIAL,
                            N_AZIMUTHAL,
                            range,
                            true,
                        )
                        .unwrap();
                    let grid =
                        IntegrationGrid2d::new(N_RADIAL, N_AZIMUTHAL, (0., 120.), range).unwrap();
                    let variance: Vec<_> = frame.data.iter().map(|s| 2. * s + 3.).collect();
                    let raw = ai
                        .integrate2d_raw_on_grid_with_variances_from_slice(
                            &frame.data,
                            &variance,
                            frame.dim1,
                            frame.dim2,
                            &grid,
                        )
                        .unwrap();
                    (pattern, cake, raw)
                };
                let references: Vec<_> = SHAPES
                    .into_iter()
                    .map(|shape| {
                        let frame = frame(shape);
                        let expected = evaluate(&new(), &frame);
                        (frame, expected)
                    })
                    .collect();
                let shared = new();
                (0..24).into_par_iter().for_each(|i| {
                    let (frame, (pattern, cake, raw)) = &references[i % references.len()];
                    let (actual_pattern, actual_cake, actual_raw) = evaluate(&shared, frame);
                    close(&actual_pattern.0, &pattern.0);
                    close(&actual_pattern.1, &pattern.1);
                    close(&actual_pattern.2, &pattern.2);
                    assert_eq!(actual_cake.radial, cake.radial);
                    assert_eq!(actual_cake.azimuthal, cake.azimuthal);
                    close(&actual_cake.intensity, &cake.intensity);
                    close(
                        actual_cake.errors.as_ref().unwrap(),
                        cake.errors.as_ref().unwrap(),
                    );
                    close(&actual_raw.sum_signal, &raw.sum_signal);
                    close(&actual_raw.sum_norm, &raw.sum_norm);
                    close(&actual_raw.count, &raw.count);
                    close(
                        actual_raw.sum_variance.as_ref().unwrap(),
                        raw.sum_variance.as_ref().unwrap(),
                    );
                });
            }
        }
    }
}

#[test]
fn queue_batch_with_workers_handles_mixed_shapes() {
    for method in [IntegrationMethod::Counting, IntegrationMethod::PixelSplit] {
        let references = references(method);
        let n_frames = 48;
        let (work_tx, work_rx) = mpsc::channel();
        for index in 0..n_frames {
            let frame = &references[index % references.len()].0;
            work_tx
                .send(WorkItem::Frame {
                    index,
                    name: frame.name.clone(),
                    frame: Frame {
                        name: frame.name.clone(),
                        data: frame.data.clone(),
                        dim1: frame.dim1,
                        dim2: frame.dim2,
                    },
                })
                .unwrap();
        }
        drop(work_tx);
        let task = Integration2D {
            n_radial: N_RADIAL,
            n_azimuthal: N_AZIMUTHAL,
        };
        let batch = process_queue_streaming(work_rx, Arc::new(integrator(method)), task, 4, 2);
        let results: Vec<_> = batch.into_iter().collect();
        assert_eq!(results.len(), n_frames, "a worker died: {method:?}");
        for (index, _, result) in results {
            let expected = references[index % references.len()].1.1.as_ref();
            assert_same(method, &result.unwrap(), expected.unwrap(), "batch frame");
        }
    }
}

/// Rust streaming helpers decode TIFFs to float64 Frames; the direct typed
/// adapters are exercised separately above.
#[test]
fn file_batch_with_workers_handles_tiffs_of_different_sizes() {
    crate::util::download_pilatus_test_file();
    let files = [
        "test_data/CeO2_Pilatus1M.tif", // 1043 x 981
        "test_data/ceo2_side.tif",
    ];
    let shapes: Vec<_> = files
        .iter()
        .map(|f| {
            let frame = Frame::from_tiff(f.to_string()).unwrap();
            (frame.dim1, frame.dim2)
        })
        .collect();
    assert_ne!(shapes[0], shapes[1], "fixtures must differ in shape");
    let filenames: Vec<String> = (0..12).map(|i| files[i % 2].to_string()).collect();
    let poni = Poni::load("test_data/v1.poni").unwrap();
    let new = || {
        Integrator::new(Geometry::new(2, 2, poni.clone())).with_method(IntegrationMethod::Counting)
    };

    let task = Integration2D {
        n_radial: 50,
        n_azimuthal: 12,
    };
    let expected: Vec<_> = files
        .iter()
        .map(|f| {
            process_files_streaming(&[f.to_string()], Arc::new(new()), task, 1, 1)
                .into_iter()
                .next()
                .unwrap()
                .2
                .unwrap()
        })
        .collect();
    let results: Vec<_> = process_files_streaming(&filenames, Arc::new(new()), task, 4, 1)
        .into_iter()
        .collect();
    assert_eq!(results.len(), filenames.len(), "a worker died");
    for (index, name, result) in results {
        assert_eq!(result.as_ref(), Ok(&expected[index % 2]), "{name} #{index}");
    }

    let task = Integration1D { n_bins: 200 };
    let expected: Vec<_> = files
        .iter()
        .map(|f| {
            process_files_streaming(&[f.to_string()], Arc::new(new()), task, 1, 1)
                .into_iter()
                .next()
                .unwrap()
                .2
                .unwrap()
        })
        .collect();
    let results: Vec<_> = process_files_streaming(&filenames, Arc::new(new()), task, 4, 1)
        .into_iter()
        .collect();
    assert_eq!(results.len(), filenames.len(), "a worker died");
    for (index, name, result) in results {
        assert_eq!(result.as_ref(), Ok(&expected[index % 2]), "{name} #{index}");
    }
}
