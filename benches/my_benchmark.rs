use criterion::{criterion_group, criterion_main, Criterion};

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::Integrator;
use dioptrin::poni::Poni;
use dioptrin::lut::csr as lut_csr;

pub fn integrate_lut_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);
    integrator.integrate_lut(&frame, 1250);

    c.bench_function("bench 1d lut", |b| {
        b.iter(|| integrator.integrate_lut(&frame, 1250))
    });
}

pub fn integrate_lut_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_points = 1250;

    // Precompute tth and CSR LUT
    let tth = geometry.calc_tth_all();
    let (csr, hist_min, hist_step) = lut_csr::lut_csr_from_tth(&tth, n_points);

    // Correct intensity (same correction as used by Integrator)
    let chi = geometry.calc_chi_all();
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up run
    let _ = lut_csr::integrate_csr(&intensity, &csr, hist_min, hist_step);

    c.bench_function("bench 1d lut_csr", |b| {
        b.iter(|| lut_csr::integrate_csr(&intensity, &csr, hist_min, hist_step))
    });
}

pub fn integrate_lut_supersample_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);

    let n_subpixels = 4;
    let n_points = 1250;

    // generate all the geometric calculations before benchmarking
    integrator.integrate_lut_ss(&frame, n_points, n_subpixels);

    c.bench_function("bench 1d lut supersampled", |b| {
        b.iter(|| integrator.integrate_lut_ss(&frame, n_points, n_subpixels))
    });
}

pub fn integrate_lut_ss_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_points = 1250;
    let n_subpixels = 4;

    // Precompute tth and CSR-SS LUT
    let tth = geometry.calc_tth_all();
    let (csr_ss, hist_min, hist_step) =
        lut_csr::lut_ss_csr(&geometry, &tth, n_points, n_subpixels);

    // Correct intensity
    let chi = geometry.calc_chi_all();
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up
    let _ = lut_csr::integrate_ss_csr(&intensity, &csr_ss, hist_min, hist_step);

    c.bench_function("bench 1d lut_ss_csr", |b| {
        b.iter(|| lut_csr::integrate_ss_csr(&intensity, &csr_ss, hist_min, hist_step))
    });
}

pub fn integrate2d_lut_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);

    let n_tth = 1250;
    let n_azi = 360;

    // generate all the geometric calculations before benchmarking
    integrator.integrate_2d_lut(&frame, n_tth, n_azi);

    c.bench_function("bench 2d lut", |b| {
        b.iter(|| integrator.integrate_2d_lut(&frame, n_tth, n_azi))
    });
}

pub fn integrate2d_lut_ss_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry);

    let n_tth = 1250;
    let n_azi = 360;
    let n_subpixels = 5;

    // generate all the geometric calculations before benchmarking
    integrator.integrate2d_lut_ss(&frame, n_tth, n_azi, n_subpixels);

    c.bench_function("bench 2d lut with supersampling", |b| {
        b.iter(|| integrator.integrate2d_lut_ss(&frame, n_tth, n_azi, n_subpixels))
    });
}

pub fn read_frame_benchmark(c: &mut Criterion) {
    c.bench_function("bench", |b| {
        b.iter(|| Frame::from_tiff(String::from("test_data/image_001.tif")).unwrap())
    });
}

criterion_group!(
    benches,
    integrate_lut_benchmark,
    integrate_lut_csr_benchmark,
    integrate_lut_supersample_benchmark,
    integrate_lut_ss_csr_benchmark,
    integrate2d_lut_benchmark,
    integrate2d_lut_ss_benchmark
);
criterion_main!(benches);
