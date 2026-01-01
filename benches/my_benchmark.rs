use criterion::{criterion_group, criterion_main, Criterion};

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::poni::Poni;
use dioptrin::lut;
use dioptrin::math::histogram;


pub fn integrate_lut_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_points = 1250;

    // Precompute tth and CSR LUT
    let tth = geometry.calc_tth_all();
    let (csr, hist_min, hist_step) = lut::build_lut1d_csr(&tth, n_points);

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
    let _ = lut::integrate_lut1d(&intensity, &csr, hist_min, hist_step);

    c.bench_function("bench 1d lut_csr", |b| {
        b.iter(|| lut::integrate_lut1d(&intensity, &csr, hist_min, hist_step))
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
    let (csr_ss, hist_min, hist_step) = lut::build_lut1d_ss_csr(&geometry, &tth, n_points, n_subpixels);

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
    let _ = lut::integrate_lut1d(&intensity, &csr_ss, hist_min, hist_step);

    c.bench_function("bench 1d lut_ss_csr", |b| {
        b.iter(|| lut::integrate_lut1d(&intensity, &csr_ss, hist_min, hist_step))
    });
}

pub fn integrate2d_lut_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_tth = 1250;
    let n_azi = 360;

    // Precompute tth, chi and CSR 2d LUT
    let tth = geometry.calc_tth_all();
    let chi = geometry.calc_chi_all();
    let csr2d = lut::build_lut2d_csr(&tth, &chi, n_tth, n_azi);

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up
    let _ = lut::integrate_lut2d(&intensity, &csr2d);

    c.bench_function("bench 2d lut_csr", |b| {
        b.iter(|| lut::integrate_lut2d(&intensity, &csr2d))
    });
}

pub fn integrate2d_lut_ss_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_tth = 1250;
    let n_azi = 360;
    let n_subpixels = 5;

    // Precompute tth, chi and CSR 2d SS LUT
    let tth = geometry.calc_tth_all();
    let chi = geometry.calc_chi_all();
    let csr2d_ss = lut::build_lut2d_ss_csr(&geometry, &tth, &chi, n_tth, n_azi, n_subpixels);

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up
    let _ = lut::integrate_lut2d(&intensity, &csr2d_ss);

    c.bench_function("bench 2d lut_ss_csr", |b| {
        b.iter(|| lut::integrate_lut2d(&intensity, &csr2d_ss))
    });
}

pub fn integrate_lut_ps_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_points = 1250;
    let tth = geometry.calc_tth_all();
    let chi = geometry.calc_chi_all();
    // Note: pixel-split CSR builder returns histogram params

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Pixel-split CSR (fractional weights)
    let (csr_ps, min2, step2) = lut::build_lut1d_ps_csr(&tth, n_points);
    let _ = lut::integrate_lut1d(&intensity, &csr_ps, min2, step2);

    c.bench_function("bench 1d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate_lut1d(&intensity, &csr_ps, min2, step2))
    });
}

pub fn integrate_lut_ps_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_points = 1250;
    let tth = geometry.calc_tth_all();
    let chi = geometry.calc_chi_all();

    let (csr_ps, min, step) = lut::build_lut1d_ps_csr(&tth, n_points);

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    let _ = lut::integrate_lut1d(&intensity, &csr_ps, min, step);

    c.bench_function("bench 1d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate_lut1d(&intensity, &csr_ps, min, step))
    });
}

pub fn integrate2d_lut_ps_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_tth = 1250;
    let n_azi = 360;
    let tth = geometry.calc_tth_all();
    let chi = geometry.calc_chi_all();

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Use unified supersampled CSR instead of HashMap pixel-split
    let csr2d_ps = lut::build_lut2d_ps_csr(&tth, &chi, n_tth, n_azi);
    let _ = lut::integrate_lut2d(&intensity, &csr2d_ps);

    c.bench_function("bench 2d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate_lut2d(&intensity, &csr2d_ps))
    });
}

pub fn integrate2d_lut_ps_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let mut geometry = Geometry::new(0, 0, poni);
    geometry.dim1 = frame.dim1;
    geometry.dim2 = frame.dim2;

    let n_tth = 1250;
    let n_azi = 360;
    let tth = geometry.calc_tth_all();
    let chi = geometry.calc_chi_all();

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &chi);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    let csr2d_ps2 = lut::build_lut2d_ps_csr(&tth, &chi, n_tth, n_azi);
    let _ = lut::integrate_lut2d(&intensity, &csr2d_ps2);

    c.bench_function("bench 2d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate_lut2d(&intensity, &csr2d_ps2))
    });
}


pub fn read_frame_benchmark(c: &mut Criterion) {
    c.bench_function("bench", |b| {
        b.iter(|| Frame::from_tiff(String::from("test_data/image_001.tif")).unwrap())
    });
}

criterion_group!(
    benches,
    // CSR-only benches
    integrate_lut_csr_benchmark,
    integrate_lut_ss_csr_benchmark,
    integrate2d_lut_csr_benchmark,
    integrate2d_lut_ss_csr_benchmark,
    integrate_lut_ps_benchmark,
    integrate_lut_ps_csr_benchmark,
    integrate2d_lut_ps_csr_benchmark,
    integrate2d_lut_ps_benchmark,
);
criterion_main!(benches);
