//! `MultiIntegrator`: merging two or three detectors on a shared grid.
//!
//! Detectors: the committed Pilatus 1M CeO2 image with `v1.poni`, a
//! synthetic 1024 x 1024 frame (100 um pixels) and a synthetic 512 x 1024
//! frame (75 um pixels), each with its own beam position and tilt. All
//! integrators share one 2-theta range covering every detector. 1D uses
//! 1250 bins, 2D 1000 x 360 over the full circle. IDs:
//!
//! - `multi/integrate1d/<n>det/<method>`, `multi/integrate2d/<n>det/<method>`:
//!   warm, the per-detector tables cached by an untimed first call.
//! - `multi/build1d/3det/<method>`, `multi/build2d/3det/<method>`: cold, new
//!   integrators and a new `MultiIntegrator` per iteration, then the first
//!   merge.
//!
//! `<n>` is 2 or 3; `<method>` is `pixel_split` or `pyfai_full`.

mod common;

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::lut;
use dioptrin::multi::{Multi2dOptions, MultiIntegrator};

const N_1D: usize = 1250;
const N_RADIAL: usize = 1000;
const N_AZIMUTHAL: usize = 360;

const METHODS: [(&str, IntegrationMethod); 2] = [
    ("pixel_split", IntegrationMethod::PixelSplit),
    ("pyfai_full", IntegrationMethod::PyfaiFull),
];

fn detectors() -> (Vec<(Geometry, Frame)>, (f64, f64)) {
    let ceo2 = common::ceo2();
    let mut poni = common::centred_poni(1024, 1024, 100e-6, 100e-6);
    poni.distance = 0.25;
    poni.poni1 = 0.02;
    poni.rot1 = 0.3;
    let square = (
        Geometry::new(1024, 1024, poni),
        common::synthetic_frame(1024, 1024, 11),
    );
    let mut poni = common::centred_poni(512, 1024, 75e-6, 75e-6);
    poni.poni1 = -0.01;
    poni.rot2 = 0.25;
    let wide = (
        Geometry::new(512, 1024, poni),
        common::synthetic_frame(512, 1024, 12),
    );
    let set = vec![ceo2, square, wide];
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (g, _) in &set {
        let r = lut::pyfai_radial_range(g, RadialUnit::TthRad);
        lo = lo.min(r[0]);
        hi = hi.max(r[1]);
    }
    (set, (lo, hi))
}

fn make(
    set: &[(Geometry, Frame)],
    range: (f64, f64),
    method: IntegrationMethod,
) -> MultiIntegrator {
    MultiIntegrator::try_new(
        set.iter()
            .map(|(g, _)| {
                Integrator::new(g.clone())
                    .with_method(method)
                    .with_radial_range(range.0, range.1)
            })
            .collect(),
    )
    .unwrap()
}

fn warm(c: &mut Criterion) {
    let (all, range) = detectors();
    let options = Multi2dOptions::default();

    let mut group = c.benchmark_group("multi/integrate1d");
    for n in [2, 3] {
        let set = &all[..n];
        let frames: Vec<&Frame> = set.iter().map(|(_, f)| f).collect();
        for (label, method) in METHODS {
            let multi = make(set, range, method);
            black_box(multi.integrate1d(&frames, N_1D));
            group.bench_function(format!("{n}det/{label}"), |b| {
                b.iter(|| multi.integrate1d(&frames, N_1D))
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("multi/integrate2d");
    for n in [2, 3] {
        let set = &all[..n];
        let frames: Vec<&Frame> = set.iter().map(|(_, f)| f).collect();
        for (label, method) in METHODS {
            let multi = make(set, range, method);
            black_box(
                multi
                    .integrate2d(&frames, N_RADIAL, N_AZIMUTHAL, &options)
                    .unwrap(),
            );
            group.bench_function(format!("{n}det/{label}"), |b| {
                b.iter(|| {
                    multi
                        .integrate2d(&frames, N_RADIAL, N_AZIMUTHAL, &options)
                        .unwrap()
                })
            });
        }
    }
    group.finish();
}

fn cold(c: &mut Criterion) {
    let (set, range) = detectors();
    let frames: Vec<&Frame> = set.iter().map(|(_, f)| f).collect();
    let options = Multi2dOptions::default();

    for dims in ["1d", "2d"] {
        let mut group = common::cold_group(c, &format!("multi/build{dims}"));
        for (label, method) in METHODS {
            group.bench_function(format!("3det/{label}"), |b| {
                b.iter_batched(
                    || make(&set, range, method),
                    |multi| {
                        if dims == "1d" {
                            black_box(multi.integrate1d(&frames, N_1D));
                        } else {
                            black_box(
                                multi
                                    .integrate2d(&frames, N_RADIAL, N_AZIMUTHAL, &options)
                                    .unwrap(),
                            );
                        }
                        multi
                    },
                    BatchSize::PerIteration,
                )
            });
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = common::config();
    targets = warm, cold
}
criterion_main!(benches);
