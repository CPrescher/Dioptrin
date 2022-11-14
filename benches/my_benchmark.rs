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

    c.bench_function("bench", |b| {
        b.iter(|| integrator.integrate_csr(&frame, 1250))
    });
}

pub fn read_frame_benchmark(c: &mut Criterion) {
    c.bench_function("bench", |b| {
        b.iter(|| Frame::from_tiff(String::from("test_data/image_001.tif")).unwrap())
    });
}

criterion_group!(benches, integrate_csr_benchmark);
criterion_main!(benches);
