//! Physical detector models: the per-geometry `Detector` operations and
//! integration with a detector attached through `Geometry::with_detector`.
//!
//! Detectors come from the bundled catalogue, binned and transformed with
//! `Detector::transformed` (`horizontal` is a flip, `left`/`right` quarter
//! turns). A transform of a uniform map (Pilatus, Eiger, MAR, Rayonix) only
//! swaps shape, pixel sizes and mask; a transform of a non-uniform map
//! (Jungfrau 500k, a separable map) builds a `PixelMap::Transformed`, whose
//! positions go through the inverse transform. Frames are synthetic and
//! deterministic, except for `none` and `pilatus1m`, which use the committed
//! CeO2 image. IDs:
//!
//! - `detectors/builtin_mask/<detector>`: `Detector::builtin_mask` for the
//!   circular-mask MAR 345 and Rayonix133, unbinned, binned and transformed,
//!   and the gapped Eiger2 4M (interval mask) for reference.
//! - `detectors/position/<detector>`: `Detector::position` at every pixel
//!   centre of the image (0.5M to 1M calls) on uniform, separable,
//!   transformed, binned and binned transformed maps.
//! - `detectors/pyfai_lut1d/<detector>`: the public 1D `pyfai_full` table
//!   builder on the attached detector; with a detector attached this runs
//!   the corner-sign vote.
//! - `detectors/build1d/<method>/<detector>`: cold, a new `Integrator` and
//!   its first `integrate1d` (pixel corners and geometry arrays from the
//!   detector, corrections, table). The detector's built-in mask is not
//!   applied by the `Integrator`.
//! - `detectors/integrate1d/pyfai_full/<detector>`: warm.
//!
//! `<detector>` in the integration and position cases: `none` (PONI only,
//! the CeO2 setup), `pilatus1m` (uniform, module gaps), `pilatus1m_left`
//! (uniform, quarter turn), `eiger2_4m_bin2` (uniform, gaps, binned 2x2),
//! `jungfrau500k` (separable map, 512 x 1024), `jungfrau500k_horizontal`,
//! `jungfrau500k_left`, `jungfrau500k_bin2`, `jungfrau500k_bin2_right`
//! (transformed, binned and binned transformed separable maps) and `cirpad`
//! (curved separable map, 11200 x 120).

mod common;

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dioptrin::detector::{self, Detector, DetectorTransform};
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::lut;

const N_1D: usize = 1250;

fn transformed(name: &str, op: Option<DetectorTransform>) -> Detector {
    let det = detector::named(name).expect("catalogue detector");
    match op {
        Some(op) => det.transformed(op).expect("transform"),
        None => det,
    }
}

fn binned_shape(det: &Detector, bin: usize) -> [usize; 2] {
    let full = det.max_shape.expect("max shape");
    [full[0] / bin, full[1] / bin]
}

type Case = (&'static str, &'static str, usize, Option<DetectorTransform>);

const LEFT: Option<DetectorTransform> = Some(DetectorTransform::Left);
const RIGHT: Option<DetectorTransform> = Some(DetectorTransform::Right);
const HORIZONTAL: Option<DetectorTransform> = Some(DetectorTransform::Horizontal);

/// Catalogue cases with a synthetic frame: (label, catalogue name, bin, transform).
const CATALOGUE_CASES: [Case; 7] = [
    ("pilatus1m_left", "Pilatus 1M", 1, LEFT),
    ("jungfrau500k", "Jungfrau 500k", 1, None),
    ("jungfrau500k_horizontal", "Jungfrau 500k", 1, HORIZONTAL),
    ("jungfrau500k_left", "Jungfrau 500k", 1, LEFT),
    ("jungfrau500k_bin2", "Jungfrau 500k", 2, None),
    ("jungfrau500k_bin2_right", "Jungfrau 500k", 2, RIGHT),
    ("cirpad", "CirPAD", 1, None),
];

/// The integration cases: (label, geometry, frame).
fn integration_cases() -> Vec<(&'static str, Geometry, Frame)> {
    let (plain, ceo2) = common::ceo2();
    let pilatus = Arc::new(detector::named("Pilatus 1M").unwrap());
    let with_pilatus = plain.clone().with_detector(pilatus).unwrap();
    let mut cases = vec![
        ("none", plain, common::copy_frame(&ceo2)),
        ("pilatus1m", with_pilatus, ceo2),
    ];
    for (label, name, bin, op) in CATALOGUE_CASES {
        let (_, geometry, frame) = common::catalogue(name, bin, op);
        cases.push((label, geometry, frame));
    }
    cases
}

fn detector_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("detectors/builtin_mask");
    group.sample_size(10);
    let masks: [Case; 7] = [
        ("mar345", "MAR 345", 1, None),
        ("mar345_bin2", "MAR 345", 2, None),
        ("mar345_bin2_left", "MAR 345", 2, LEFT),
        ("rayonix133", "Rayonix133", 1, None),
        ("rayonix133_bin2", "Rayonix133", 2, None),
        ("rayonix133_bin2_horizontal", "Rayonix133", 2, HORIZONTAL),
        ("eiger2_4m", "Eiger2 4M", 1, None),
    ];
    for (label, name, bin, op) in masks {
        let det = transformed(name, op);
        if op.is_some() && name == "Jungfrau 500k" {
            // A transformed non-uniform map goes through the inverse transform.
            assert!(
                matches!(det.map, detector::PixelMap::Transformed { .. }),
                "{label}: not a transformed map"
            );
        }
        let shape = binned_shape(&det, bin);
        let mask = det.builtin_mask(shape).unwrap();
        assert!(mask.iter().any(|&m| m != 0), "{label}: empty mask");
        group.bench_function(label, |b| {
            b.iter_with_large_drop(|| det.builtin_mask(shape).unwrap())
        });
    }
    group.finish();

    let mut group = c.benchmark_group("detectors/position");
    group.sample_size(10);
    let positions = [
        ("pilatus1m", "Pilatus 1M", 1, None),
        ("eiger2_4m_bin2", "Eiger2 4M", 2, None),
    ]
    .into_iter()
    .chain(CATALOGUE_CASES);
    for (label, name, bin, op) in positions {
        let det = transformed(name, op);
        let shape = binned_shape(&det, bin);
        group.bench_function(label, |b| {
            b.iter(|| {
                let mut sum = 0.0;
                for row in 0..shape[0] {
                    for col in 0..shape[1] {
                        let p = det.position(shape, row as f64 + 0.5, col as f64 + 0.5);
                        sum += p[1] + p[2];
                    }
                }
                black_box(sum)
            })
        });
    }
    group.finish();
}

fn attached(c: &mut Criterion) {
    let cases = integration_cases();

    let mut group = common::cold_group(c, "detectors/pyfai_lut1d");
    for (label, geometry, _) in &cases {
        group.bench_function(*label, |b| {
            b.iter_with_large_drop(|| {
                lut::build_lut1d_pyfai_csr(geometry, RadialUnit::TthRad, N_1D, None, None).unwrap()
            })
        });
    }
    group.finish();

    // The cold and warm Integrator cases use a subset, to keep the run short.
    let subset = |names: &[&str]| -> Vec<&(&'static str, Geometry, Frame)> {
        cases.iter().filter(|(l, _, _)| names.contains(l)).collect()
    };
    let mut group = common::cold_group(c, "detectors/build1d");
    for (method_label, method, names) in [
        (
            "pixel_split",
            IntegrationMethod::PixelSplit,
            &["none", "pilatus1m", "jungfrau500k_left"][..],
        ),
        (
            "pyfai_full",
            IntegrationMethod::PyfaiFull,
            &[
                "none",
                "pilatus1m",
                "jungfrau500k",
                "jungfrau500k_left",
                "jungfrau500k_bin2_right",
            ][..],
        ),
    ] {
        for (label, geometry, frame) in subset(names) {
            group.bench_function(format!("{method_label}/{label}"), |b| {
                b.iter_batched(
                    || Integrator::new(geometry.clone()).with_method(method),
                    |ai| {
                        let out = ai.integrate1d(frame, N_1D);
                        (out, ai)
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("detectors/integrate1d");
    for (label, geometry, frame) in subset(&["none", "pilatus1m", "jungfrau500k_left"]) {
        let ai = Integrator::new(geometry.clone()).with_method(IntegrationMethod::PyfaiFull);
        black_box(ai.integrate1d(frame, N_1D));
        group.bench_function(format!("pyfai_full/{label}"), |b| {
            b.iter(|| ai.integrate1d(frame, N_1D))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::config();
    targets = detector_ops, attached
}
criterion_main!(benches);
