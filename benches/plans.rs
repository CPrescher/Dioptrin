//! Prepared and fixed integration plans, and the CSC/CSR layout choice, at
//! 1M and 4M frame sizes. All warm: plans are prepared outside the timed
//! region, and only `integrate_frame` (or the `Integrator` call) is timed.
//!
//! Sizes: `1m` is the committed Pilatus 1M CeO2 image (1043 x 981, `v1.poni`);
//! `4m` is a synthetic 2048 x 2048 frame with 75 um pixels. 1D uses 1250 bins,
//! 2D 1000 x 360. IDs:
//!
//! - `plans/fixed1d/<size>/<layout>`: `Fixed1dPlan::integrate_frame`
//!   (`pixel_split`) with `with_csc_layout` (`csc`) and with
//!   `with_auto_layout` (`auto`, which picks CSR for these sizes). The plan
//!   as prepared is left out: on first use it times a gather kernel against
//!   CSC and keeps the winner, so on a loaded machine it can time a
//!   different kernel on each revision.
//! - `plans/layout1d/<size>/<layout>`: `Integrator::integrate1d` with
//!   `SparseLayout::Csc` or `SparseLayout::Csr`.
//! - `plans/fixed_csr1d/<size>/<input>`: `FixedCsr1dPlan::integrate_frame`
//!   on f64 pixels (`f64`) and `FixedCsr1dInputPlan` on u16 and u32 pixels
//!   (`u16`, `u32`) and on u32 pixels with bins in parallel (`u32_parallel`),
//!   with a `pyfai_full` table.
//! - `plans/fixed_grid2d/<size>/<method>`: `FixedGrid2dPlan::integrate_frame`
//!   on a shared 2D grid.

mod common;

use std::f64::consts::PI;
use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::RadialUnit;
use dioptrin::integration::{
    FixedCsr1dInputPlan, FixedCsr1dPlan, InputOrder, IntegrationGrid2d, IntegrationMethod,
    Integrator, SparseLayout,
};
use dioptrin::lut;

const N_1D: usize = 1250;
const N_RADIAL: usize = 1000;
const N_AZIMUTHAL: usize = 360;

fn sizes() -> Vec<(&'static str, Geometry, Frame)> {
    let (g1, f1) = common::ceo2();
    let (g4, f4) = common::synthetic(2048, 75e-6, 4);
    vec![("1m", g1, f1), ("4m", g4, f4)]
}

fn fixed1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("plans/fixed1d");
    for (size, geometry, frame) in sizes() {
        let shape = (frame.dim1, frame.dim2);
        let ai = Integrator::new(geometry.clone());
        for layout in ["csc", "auto"] {
            let plan = ai.prepare_fixed1d(shape, N_1D).unwrap();
            let plan = if layout == "csc" {
                plan.with_csc_layout()
            } else {
                plan.with_auto_layout()
            };
            black_box(plan.integrate_frame(&frame.data, false).unwrap());
            group.bench_function(format!("{size}/{layout}"), |b| {
                b.iter(|| plan.integrate_frame(&frame.data, false).unwrap())
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("plans/layout1d");
    for (size, geometry, frame) in sizes() {
        for (label, layout) in [("csc", SparseLayout::Csc), ("csr", SparseLayout::Csr)] {
            let ai = Integrator::new(geometry.clone()).with_sparse_layout(layout);
            black_box(ai.integrate1d(&frame, N_1D));
            group.bench_function(format!("{size}/{label}"), |b| {
                b.iter(|| ai.integrate1d(&frame, N_1D))
            });
        }
    }
    group.finish();
}

fn fixed_csr(c: &mut Criterion) {
    let mut group = c.benchmark_group("plans/fixed_csr1d");
    for (size, geometry, frame) in sizes() {
        let shape = (frame.dim1, frame.dim2);
        let table =
            lut::build_lut1d_pyfai_csr(&geometry, RadialUnit::TthRad, N_1D, None, None).unwrap();
        let norm = geometry.calc_solid_angle();
        let plan = Arc::new(FixedCsr1dPlan::new(table, norm, None).unwrap());
        let input = FixedCsr1dInputPlan::new(plan.clone(), shape, InputOrder::Detector).unwrap();
        let u16_pixels: Vec<u16> = frame.data.iter().map(|&v| v as u16).collect();
        let u32_pixels: Vec<u32> = frame.data.iter().map(|&v| v as u32).collect();

        group.bench_function(format!("{size}/f64"), |b| {
            b.iter(|| plan.integrate_frame(&frame.data, false).unwrap())
        });
        group.bench_function(format!("{size}/u16"), |b| {
            b.iter(|| input.integrate_u16(&u16_pixels, false).unwrap())
        });
        group.bench_function(format!("{size}/u32"), |b| {
            b.iter(|| input.integrate_u32(&u32_pixels, false).unwrap())
        });
        group.bench_function(format!("{size}/u32_parallel"), |b| {
            b.iter(|| input.integrate_u32_parallel(&u32_pixels, false).unwrap())
        });
    }
    group.finish();
}

fn fixed_grid2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("plans/fixed_grid2d");
    for (size, geometry, frame) in sizes() {
        let shape = (frame.dim1, frame.dim2);
        let radial = lut::pyfai_radial_range(&geometry, RadialUnit::TthRad);
        let grid = IntegrationGrid2d::new(N_RADIAL, N_AZIMUTHAL, (radial[0], radial[1]), (-PI, PI))
            .unwrap();
        for (label, method) in [
            ("pixel_split", IntegrationMethod::PixelSplit),
            ("pyfai_full", IntegrationMethod::PyfaiFull),
        ] {
            let ai = Integrator::new(geometry.clone()).with_method(method);
            let plan = ai.prepare_fixed2d_on_grid(shape, &grid).unwrap();
            black_box(plan.integrate_frame(&frame.data, false).unwrap());
            group.bench_function(format!("{size}/{label}"), |b| {
                b.iter(|| plan.integrate_frame(&frame.data, false).unwrap())
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::config();
    targets = fixed1d, fixed_csr, fixed_grid2d
}
criterion_main!(benches);
