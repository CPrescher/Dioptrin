use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::poni::Poni;

fn setup_integrator() -> (Integrator, Frame) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    let integrator = Integrator::new(geometry);
    (integrator, frame)
}

pub fn integrator_1d_histogram_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::Histogram);
    let n_points = 1250;

    // Warm-up
    let _ = integrator.integrate1d(&frame, n_points);

    c.bench_function("integrator 1d histogram", |b| {
        b.iter(|| integrator.integrate1d(&frame, n_points))
    });
}

pub fn integrator_1d_counting_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::Counting);
    let n_points = 1250;

    // Warm-up
    let _ = integrator.integrate1d(&frame, n_points);

    c.bench_function("integrator 1d counting", |b| {
        b.iter(|| integrator.integrate1d(&frame, n_points))
    });
}

pub fn integrator_1d_pixel_split_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::PixelSplit);
    let n_points = 1250;

    // Warm-up
    let _ = integrator.integrate1d(&frame, n_points);

    c.bench_function("integrator 1d pixel-split", |b| {
        b.iter(|| integrator.integrate1d(&frame, n_points))
    });
}

pub fn integrator_1d_supersampled_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::Supersampled { n_ss: 4 });
    let n_points = 1250;

    // Warm-up
    let _ = integrator.integrate1d(&frame, n_points);

    c.bench_function("integrator 1d supersampled n_ss=4", |b| {
        b.iter(|| integrator.integrate1d(&frame, n_points))
    });
}

pub fn integrator_1d_with_errors_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::PixelSplit);
    let n_points = 1250;

    // Warm-up
    let _ = integrator.integrate1d_with_errors(&frame, n_points);

    c.bench_function("integrator 1d with errors", |b| {
        b.iter(|| integrator.integrate1d_with_errors(&frame, n_points))
    });
}

pub fn integrator_1d_q_unit_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::PixelSplit);
    integrator.set_radial_unit(RadialUnit::QAngstrom);
    let n_points = 1250;

    // Warm-up
    let _ = integrator.integrate1d(&frame, n_points);

    c.bench_function("integrator 1d q unit", |b| {
        b.iter(|| integrator.integrate1d(&frame, n_points))
    });
}

pub fn integrator_1d_with_polarization_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::PixelSplit);
    integrator.set_polarization_factor(Some(0.99));
    let n_points = 1250;

    // Warm-up
    let _ = integrator.integrate1d(&frame, n_points);

    c.bench_function("integrator 1d with polarization", |b| {
        b.iter(|| integrator.integrate1d(&frame, n_points))
    });
}

pub fn integrator_2d_counting_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::Counting);
    let n_radial = 1250;
    let n_azimuthal = 360;

    // Warm-up
    let _ = integrator.integrate2d(&frame, n_radial, n_azimuthal);

    c.bench_function("integrator 2d counting", |b| {
        b.iter(|| integrator.integrate2d(&frame, n_radial, n_azimuthal))
    });
}

pub fn integrator_2d_pixel_split_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::PixelSplit);
    let n_radial = 1250;
    let n_azimuthal = 360;

    // Warm-up
    let _ = integrator.integrate2d(&frame, n_radial, n_azimuthal);

    c.bench_function("integrator 2d pixel-split", |b| {
        b.iter(|| integrator.integrate2d(&frame, n_radial, n_azimuthal))
    });
}

pub fn integrator_2d_supersampled_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::Supersampled { n_ss: 4 });
    let n_radial = 1250;
    let n_azimuthal = 360;

    // Warm-up
    let _ = integrator.integrate2d(&frame, n_radial, n_azimuthal);

    c.bench_function("integrator 2d supersampled n_ss=4", |b| {
        b.iter(|| integrator.integrate2d(&frame, n_radial, n_azimuthal))
    });
}

pub fn integrator_2d_with_errors_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::PixelSplit);
    let n_radial = 1250;
    let n_azimuthal = 360;

    // Warm-up
    let _ = integrator.integrate2d_with_errors(&frame, n_radial, n_azimuthal);

    c.bench_function("integrator 2d with errors", |b| {
        b.iter(|| integrator.integrate2d_with_errors(&frame, n_radial, n_azimuthal))
    });
}

pub fn integrator_2d_azimuthal_range_benchmark(c: &mut Criterion) {
    let (mut integrator, frame) = setup_integrator();
    integrator.set_method(IntegrationMethod::PixelSplit);
    let n_radial = 1250;
    let n_azimuthal = 360;
    let azimuthal_start = 0.0;
    let azimuthal_step = 2.0 * std::f64::consts::PI / n_azimuthal as f64;

    // Warm-up
    let _ = integrator.integrate2d_with_azimuthal_range(
        &frame,
        n_radial,
        n_azimuthal,
        azimuthal_start,
        azimuthal_step,
    );

    c.bench_function("integrator 2d azimuthal range", |b| {
        b.iter(|| {
            integrator.integrate2d_with_azimuthal_range(
                &frame,
                n_radial,
                n_azimuthal,
                azimuthal_start,
                azimuthal_step,
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = integrator_1d_histogram_benchmark,
        integrator_1d_counting_benchmark,
        integrator_1d_pixel_split_benchmark,
        integrator_1d_supersampled_benchmark,
        integrator_1d_with_errors_benchmark,
        integrator_1d_q_unit_benchmark,
        integrator_1d_with_polarization_benchmark,
        integrator_2d_counting_benchmark,
        integrator_2d_pixel_split_benchmark,
        integrator_2d_supersampled_benchmark,
        integrator_2d_with_errors_benchmark,
        integrator_2d_azimuthal_range_benchmark
}
criterion_main!(benches);
