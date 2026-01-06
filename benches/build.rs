use criterion::{criterion_group, criterion_main, Criterion};

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::RadialUnit;
use dioptrin::lut;
use dioptrin::poni::Poni;

pub fn build_lut_ss_1d_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_points = 1250;
    let tth = geometry.calc_tth_all();

    c.bench_function("bench build 1d lut supersampled", |b| {
        b.iter(|| lut::build_lut1d_ss_csr(&geometry, &tth, n_points, 5, RadialUnit::TthRad))
    });
}

pub fn build_lut_ss_2d_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_tth = 1250;
    let n_azimuthal = 360;
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();

    c.bench_function("bench build 2d lut supersampled", |b| {
        b.iter(|| {
            lut::build_lut2d_ss_csr(
                &geometry,
                &tth,
                &azimuthal,
                n_tth,
                n_azimuthal,
                5,
                RadialUnit::TthRad,
            )
        })
    });
}

pub fn build_lut_ps_1d_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_points = 1250;
    let tth = geometry.calc_tth_all();

    c.bench_function("bench build 1d lut pixel-split", |b| {
        b.iter(|| lut::build_lut1d_ps_csr(&geometry, &tth, n_points, RadialUnit::TthRad))
    });
}

pub fn build_lut_ps_2d_benchmark(c: &mut Criterion) {
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let n_tth = 1250;
    let n_azimuthal = 360;
    let tth = geometry.calc_tth_all();
    let azimuthal = geometry.calc_azimuth_all();

    c.bench_function("bench build 2d lut pixel-split", |b| {
        b.iter(|| {
            lut::build_lut2d_ps_csr(
                &geometry,
                &tth,
                &azimuthal,
                n_tth,
                n_azimuthal,
                RadialUnit::TthRad,
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = build_lut_ss_1d_benchmark, build_lut_ss_2d_benchmark, build_lut_ps_1d_benchmark, build_lut_ps_2d_benchmark
}
criterion_main!(benches);
