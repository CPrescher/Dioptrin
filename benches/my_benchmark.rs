use criterion::{criterion_group, criterion_main, Criterion};

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::Integrator;
use dioptrin::poni::Poni;

pub fn integrate_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);
    integrator.integrate_csr(&frame, 1250);

    c.bench_function("bench 1d csr", |b| {
        b.iter(|| integrator.integrate_csr(&frame, 1250))
    });
}

pub fn integrate_csr_supersample_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);

    let n_subpixels = 4;
    let n_points = 1250;

    // generate all the geometric calculations before benchmarking
    integrator.integrate_csr_supersampled(&frame, n_points, n_subpixels);

    c.bench_function("bench 1d csr supersampled", |b| {
        b.iter(|| integrator.integrate_csr_supersampled(&frame, n_points, n_subpixels))
    });
}

pub fn integrate2d_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);

    let n_tth = 1250;
    let n_azi = 360;

    // generate all the geometric calculations before benchmarking
    integrator.integrate2d_csr(&frame, n_tth, n_azi);

    c.bench_function("bench 2d csr", |b| {
        b.iter(|| integrator.integrate2d_csr(&frame, n_tth, n_azi))
    });
}

pub fn integrate2d_csr_ss_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);

    let n_tth = 1250;
    let n_azi = 360;
    let n_subpixels = 5;

    // generate all the geometric calculations before benchmarking
    integrator.integrate2d_csr_ss(&frame, n_tth, n_azi, n_subpixels);

    c.bench_function("bench 2d csr with supersampling", |b| {
        b.iter(|| integrator.integrate2d_csr_ss(&frame, n_tth, n_azi, n_subpixels))
    });
}

pub fn read_frame_benchmark(c: &mut Criterion) {
    c.bench_function("bench", |b| {
        b.iter(|| Frame::from_tiff(String::from("test_data/image_001.tif")).unwrap())
    });
}

criterion_group!(
    benches,
    integrate_csr_benchmark,
    integrate_csr_supersample_benchmark,
    integrate2d_csr_benchmark,
    integrate2d_csr_ss_benchmark
);
criterion_main!(benches);
