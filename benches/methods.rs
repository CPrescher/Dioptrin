//! The four table-based integration methods side by side, cold and warm, in
//! 1D and 2D. Compiles against v0.12.0 as well, so the methods can be compared
//! with that release; the 0.13 azimuthal-range APIs are in `azimuthal.rs`.
//!
//! Input: the committed Pilatus 1M CeO2 image (1043 x 981) with `v1.poni`,
//! no detector attached. IDs:
//!
//! - `methods/build1d/<method>`, `methods/build2d/<method>`: cold. A new
//!   `Integrator` per iteration, then its first `integrate1d` (1250 bins) or
//!   `integrate2d` (1000 x 360): geometry arrays, corrections, the table and
//!   one integration, as a user's first frame sees it. The `Integrator` is
//!   dropped outside the timed region.
//! - `methods/lut1d/pyfai_full`: the public 1D `pyfai_full` table builder
//!   alone (the 2D one is crate-private).
//! - `methods/integrate1d/<method>`, `methods/integrate2d/<method>`: warm,
//!   the table cached by an untimed first call.
//!
//! `<method>` is one of `counting`, `pixel_split`, `supersampled4` (n_ss = 4)
//! and `pyfai_full`.

mod common;

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::lut;

const N_1D: usize = 1250;
const N_RADIAL: usize = 1000;
const N_AZIMUTHAL: usize = 360;

const METHODS: [(&str, IntegrationMethod); 4] = [
    ("counting", IntegrationMethod::Counting),
    ("pixel_split", IntegrationMethod::PixelSplit),
    ("supersampled4", IntegrationMethod::Supersampled { n_ss: 4 }),
    ("pyfai_full", IntegrationMethod::PyfaiFull),
];

fn build(c: &mut Criterion) {
    let (geometry, frame) = common::ceo2();

    let mut group = common::cold_group(c, "methods/build1d");
    for (name, method) in METHODS {
        group.bench_function(name, |b| {
            b.iter_batched(
                || Integrator::new(geometry.clone()).with_method(method),
                |ai| {
                    let out = ai.integrate1d(&frame, N_1D);
                    (out, ai)
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();

    let mut group = common::cold_group(c, "methods/build2d");
    for (name, method) in METHODS {
        group.bench_function(name, |b| {
            b.iter_batched(
                || Integrator::new(geometry.clone()).with_method(method),
                |ai| {
                    let out = ai.integrate2d(&frame, N_RADIAL, N_AZIMUTHAL);
                    (out, ai)
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();

    let mut group = common::cold_group(c, "methods/lut1d");
    group.bench_function("pyfai_full", |b| {
        b.iter_with_large_drop(|| {
            lut::build_lut1d_pyfai_csr(&geometry, RadialUnit::TthRad, N_1D, None, None).unwrap()
        })
    });
    group.finish();
}

fn integrate(c: &mut Criterion) {
    let (geometry, frame) = common::ceo2();

    let mut group = c.benchmark_group("methods/integrate1d");
    for (name, method) in METHODS {
        let ai = Integrator::new(geometry.clone()).with_method(method);
        black_box(ai.integrate1d(&frame, N_1D));
        group.bench_function(name, |b| b.iter(|| ai.integrate1d(&frame, N_1D)));
    }
    group.finish();

    let mut group = c.benchmark_group("methods/integrate2d");
    for (name, method) in METHODS {
        let ai = Integrator::new(geometry.clone()).with_method(method);
        black_box(ai.integrate2d(&frame, N_RADIAL, N_AZIMUTHAL));
        group.bench_function(name, |b| {
            b.iter(|| ai.integrate2d(&frame, N_RADIAL, N_AZIMUTHAL))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::config();
    targets = build, integrate
}
criterion_main!(benches);
