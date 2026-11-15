//! The Massif peak search used by calibration, with and without reuse of
//! `PreparedMassif` (new in 0.13, so this target does not compile against
//! v0.12.0).
//!
//! Input: the committed LaB6 MarCCD image (2048 x 2048), default
//! `MassifOptions`, seeds on four rings. IDs:
//!
//! - `massif/find_peaks`: one seed; builds a `PreparedMassif` (blur and
//!   labelling) on every call.
//! - `massif/prepared_new`: that preparation alone.
//! - `massif/prepared_find_peaks`: one seed on a reused `PreparedMassif`.
//! - `massif/find_peaks_4seeds`, `massif/prepared_4seeds`: four seeds without
//!   reuse, and with one preparation (included) shared by the four.

mod common;

use dioptrin::calibration::{MassifOptions, PreparedMassif, find_peaks};
use dioptrin::frame::Frame;

use criterion::{Criterion, criterion_group, criterion_main};

const LAB6_TIFF: &str = "test_data/LaB6_40keV_MarCCD.tif";
// Seeds on LaB6 rings, as (row, column).
const SEEDS: [(usize, usize); 4] = [(1180, 1129), (1268, 1120), (1024, 1400), (700, 1024)];

fn massif(c: &mut Criterion) {
    let frame = Frame::from_tiff(String::from(LAB6_TIFF)).expect("test_data/LaB6_40keV_MarCCD.tif");
    let (rows, cols) = (frame.dim1, frame.dim2);
    let options = MassifOptions::default();
    let prepared = PreparedMassif::new(frame.data.clone(), rows, cols, options);
    for seed in SEEDS {
        assert!(
            !prepared.find_peaks(seed).is_empty(),
            "no peaks at seed {seed:?}"
        );
    }

    let mut group = common::cold_group(c, "massif");
    group.bench_function("find_peaks", |b| {
        b.iter(|| find_peaks(&frame.data, rows, cols, SEEDS[0], &options))
    });
    group.bench_function("prepared_new", |b| {
        b.iter_with_large_drop(|| PreparedMassif::new(frame.data.clone(), rows, cols, options))
    });
    group.bench_function("prepared_find_peaks", |b| {
        b.iter(|| prepared.find_peaks(SEEDS[0]))
    });
    group.bench_function("find_peaks_4seeds", |b| {
        b.iter(|| {
            SEEDS
                .iter()
                .map(|&s| find_peaks(&frame.data, rows, cols, s, &options).len())
                .sum::<usize>()
        })
    });
    group.bench_function("prepared_4seeds", |b| {
        b.iter(|| {
            let p = PreparedMassif::new(frame.data.clone(), rows, cols, options);
            SEEDS.iter().map(|&s| p.find_peaks(s).len()).sum::<usize>()
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::config();
    targets = massif
}
criterion_main!(benches);
