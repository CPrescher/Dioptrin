use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::RadialUnit;
use dioptrin::lut;
use dioptrin::poni::Poni;

pub fn integrate_lut_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_points = 1250;

    // Precompute tth and CSR LUT
    let tth = geometry.calc_tth_all();
    let csr = lut::build_lut1d_csr(&tth, n_points);

    // Correct intensity (same correction as used by Integrator)
    let azimuthal = geometry.calc_azimuth_all();
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up run
    let _ = lut::integrate1d(&intensity, &csr, None, None);

    c.bench_function("bench 1d lut_csr", |b| {
        b.iter(|| lut::integrate1d(&intensity, &csr, None, None))
    });
}

pub fn integrate_lut_ss_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_points = 1250;
    let n_subpixels = 4;

    // Precompute tth and CSR-SS LUT
    let tth = geometry.calc_tth_all();
    let csr_ss =
        lut::build_lut1d_ss_csr(&geometry, &tth, n_points, n_subpixels, RadialUnit::TthRad);

    // Correct intensity
    let azimuthal = geometry.calc_azimuth_all();
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up
    let _ = lut::integrate1d(&intensity, &csr_ss, None, None);

    c.bench_function("bench 1d lut_ss_csr", |b| {
        b.iter(|| lut::integrate1d(&intensity, &csr_ss, None, None))
    });
}

pub fn integrate2d_lut_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_tth = 1250;
    let n_azimuthal = 360;

    // Precompute tth, azimuthal and CSR 2d LUT
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();
    let csr2d = lut::build_lut2d_csr(&tth, &azimuthal, n_tth, n_azimuthal);

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up
    let _ = lut::integrate2d(&intensity, &csr2d, None, None);

    c.bench_function("bench 2d lut_csr", |b| {
        b.iter(|| lut::integrate2d(&intensity, &csr2d, None, None))
    });
}

pub fn integrate2d_lut_ss_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_tth = 1250;
    let n_azimuthal = 360;
    let n_subpixels = 5;

    // Precompute tth, azimuthal and CSR 2d SS LUT
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();
    let csr2d_ss = lut::build_lut2d_ss_csr(
        &geometry,
        &tth,
        &azimuthal,
        n_tth,
        n_azimuthal,
        n_subpixels,
        RadialUnit::TthRad,
    );

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Warm-up
    let _ = lut::integrate2d(&intensity, &csr2d_ss, None, None);

    c.bench_function("bench 2d lut_ss_csr", |b| {
        b.iter(|| lut::integrate2d(&intensity, &csr2d_ss, None, None))
    });
}

pub fn integrate_lut_ps_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_points = 1250;
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();
    // Note: pixel-split CSR builder returns histogram params

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Pixel-split CSR (fractional weights)
    let csr_ps = lut::build_lut1d_ps_csr(&geometry, &tth, n_points, RadialUnit::TthRad);
    let _ = lut::integrate1d(&intensity, &csr_ps, None, None);

    c.bench_function("bench 1d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate1d(&intensity, &csr_ps, None, None))
    });
}

pub fn integrate_lut_ps_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_points = 1250;
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();

    let csr_ps = lut::build_lut1d_ps_csr(&geometry, &tth, n_points, RadialUnit::TthRad);

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    let _ = lut::integrate1d(&intensity, &csr_ps, None, None);

    c.bench_function("bench 1d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate1d(&intensity, &csr_ps, None, None))
    });
}

pub fn integrate2d_lut_ps_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_tth = 1250;
    let n_azimuthal = 360;
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    // Use unified supersampled CSR instead of HashMap pixel-split
    let csr2d_ps = lut::build_lut2d_ps_csr(
        &geometry,
        &tth,
        &azimuthal,
        n_tth,
        n_azimuthal,
        RadialUnit::TthRad,
    );
    let _ = lut::integrate2d(&intensity, &csr2d_ps, None, None);

    c.bench_function("bench 2d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate2d(&intensity, &csr2d_ps, None, None))
    });
}

pub fn integrate2d_lut_ps_csr_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_tth = 1250;
    let n_azimuthal = 360;
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();

    // Correct intensity
    let solid_angle = geometry.calc_solid_angle();
    let polarization = geometry.calc_polarization(0.99, &tth, &azimuthal);
    let intensity: Vec<f64> = frame
        .data
        .iter()
        .enumerate()
        .map(|(i, y)| y / polarization[i] / solid_angle[i])
        .collect();

    let csr2d_ps2 = lut::build_lut2d_ps_csr(
        &geometry,
        &tth,
        &azimuthal,
        n_tth,
        n_azimuthal,
        RadialUnit::TthRad,
    );
    let _ = lut::integrate2d(&intensity, &csr2d_ps2, None, None);

    c.bench_function("bench 2d lut pixel-split csr", |b| {
        b.iter(|| lut::integrate2d(&intensity, &csr2d_ps2, None, None))
    });
}

pub fn read_frame_benchmark(c: &mut Criterion) {
    c.bench_function("bench", |b| {
        b.iter(|| Frame::from_tiff(String::from("test_data/image_001.tif")).unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(3));
    targets = integrate_lut_csr_benchmark,
        integrate_lut_ss_csr_benchmark,
        integrate2d_lut_csr_benchmark,
        integrate2d_lut_ss_csr_benchmark,
        integrate_lut_ps_benchmark,
        integrate_lut_ps_csr_benchmark,
        integrate2d_lut_ps_csr_benchmark,
        integrate2d_lut_ps_benchmark,
}
criterion_main!(benches);
