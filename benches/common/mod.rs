//! Inputs shared by the coverage benches (`methods`, `detectors`, `plans`,
//! `multi`, `azimuthal`, `batch`, `massif`). Not a bench target itself: Cargo only discovers
//! `benches/*.rs` and `benches/*/main.rs`.
//!
//! Every input is deterministic: real fixtures from `test_data/`, or synthetic
//! frames from a fixed-seed generator. Benchmark IDs are
//! `<target>/<case>/<variant>` so that a filter such as `cargo bench --bench
//! methods -- 'methods/build1d/'` picks out one family and `critcmp`-style
//! tools line the same IDs up between two revisions.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, SamplingMode};
use dioptrin::detector::{self, DetectorTransform};
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::poni::Poni;

pub const CEO2_TIFF: &str = "test_data/CeO2_Pilatus1M.tif";
pub const CEO2_PONI: &str = "test_data/v1.poni";

/// Short warm-up and measurement: these targets favour coverage over
/// precision. Groups that time cold table builds lower the sample count and
/// switch to flat sampling themselves.
pub fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(1500))
        .sample_size(20)
}

/// A group for cold builds: 10 flat samples (one iteration each) after a
/// short warm-up, since one iteration takes tens to hundreds of milliseconds.
pub fn cold_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(300));
    group
}

/// The committed Pilatus 1M CeO2 image (1043 x 981) and its PONI.
pub fn ceo2() -> (Geometry, Frame) {
    let poni = Poni::load(CEO2_PONI).expect("test_data/v1.poni");
    let frame = Frame::from_tiff(String::from(CEO2_TIFF)).expect("test_data/CeO2_Pilatus1M.tif");
    (Geometry::new(frame.dim1, frame.dim2, poni), frame)
}

/// Deterministic noise in [0, 1000) from a fixed-seed xorshift generator.
pub fn synthetic_frame(dim1: usize, dim2: usize, seed: u64) -> Frame {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let data = (0..dim1 * dim2)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1000) as f64
        })
        .collect();
    Frame {
        name: format!("synthetic_{dim1}x{dim2}_{seed}"),
        data,
        dim1,
        dim2,
    }
}

/// A PONI with the beam near the image centre, a slight tilt and the given
/// pixel size: a generic geometry for frames the fixture cannot provide.
pub fn centred_poni(dim1: usize, dim2: usize, pixel1: f64, pixel2: f64) -> Poni {
    let mut poni = Poni::new();
    poni.pixel1 = pixel1;
    poni.pixel2 = pixel2;
    poni.distance = 0.2;
    poni.poni1 = 0.47 * dim1 as f64 * pixel1;
    poni.poni2 = 0.53 * dim2 as f64 * pixel2;
    poni.rot1 = 0.02;
    poni.rot2 = -0.015;
    poni.wavelength = 0.3344e-10;
    poni
}

/// A synthetic square-pixel geometry and frame: 1024 x 1024 for "1M", 2048 x
/// 2048 for "4M", 256 x 256 for small frames.
pub fn synthetic(dim: usize, pixel: f64, seed: u64) -> (Geometry, Frame) {
    let geometry = Geometry::new(dim, dim, centred_poni(dim, dim, pixel, pixel));
    (geometry, synthetic_frame(dim, dim, seed))
}

/// A catalogue detector, binned by `bin` and transformed by `transform`, with
/// a geometry for the resulting image shape (pixel size taken from the
/// detector, beam near the centre) and a synthetic frame of that shape.
pub fn catalogue(
    name: &str,
    bin: usize,
    transform: Option<DetectorTransform>,
) -> (Arc<detector::Detector>, Geometry, Frame) {
    let mut det = detector::named(name).expect("catalogue detector");
    if let Some(op) = transform {
        det = det.transformed(op).expect("transform");
    }
    let full = det.max_shape.expect("detector with a max shape");
    let shape = [full[0] / bin, full[1] / bin];
    let pixels = det.effective_pixels(shape);
    let det = Arc::new(det);
    let poni = centred_poni(shape[0], shape[1], pixels[0], pixels[1]);
    let geometry = Geometry::new(shape[0], shape[1], poni)
        .with_detector(det.clone())
        .expect("detector fits the shape");
    (det, geometry, synthetic_frame(shape[0], shape[1], 7))
}

/// `Frame` is not `Clone`; copy its fields.
pub fn copy_frame(frame: &Frame) -> Frame {
    Frame {
        name: frame.name.clone(),
        data: frame.data.clone(),
        dim1: frame.dim1,
        dim2: frame.dim2,
    }
}
