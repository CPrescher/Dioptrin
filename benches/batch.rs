//! The Rust batch helpers on many frames, in frames per second.
//!
//! - `batch/queue1d/<size>/w<workers>`, `batch/queue2d/<size>/w<workers>`:
//!   `process_queue_streaming` on in-memory frames (`WorkItem::Frame`), one
//!   Rayon thread per worker, `pixel_split`, 1000 bins (1D) or 500 x 360
//!   (2D), tables built before timing. Each iteration starts the workers,
//!   integrates the batch and collects every result; the queue is filled
//!   before timing starts. `256` is 200 synthetic 256 x 256 frames, `1m` 16
//!   copies of the CeO2 Pilatus 1M image.
//! - `batch/files/1m/w<workers>`: `process_files` on 16 paths to the CeO2
//!   TIFF (decoding included; the file is in the OS cache after the first
//!   read).
//!
//! `<workers>` is 2 or 8.

mod common;

use std::hint::black_box;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use dioptrin::batch::{
    Integration1D, Integration2D, IntegrationTask, WorkItem, process_files, process_queue_streaming,
};
use dioptrin::frame::Frame;
use dioptrin::integration::{IntegrationMethod, Integrator};

const WORKERS: [usize; 2] = [2, 8];

fn queue<T>(c: &mut Criterion, name: &str, sets: &[(&str, Arc<Integrator>, Vec<Frame>)], task: T)
where
    T: IntegrationTask + Clone + 'static,
{
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    for (size, ai, frames) in sets {
        group.throughput(Throughput::Elements(frames.len() as u64));
        for workers in WORKERS {
            group.bench_function(format!("{size}/w{workers}"), |b| {
                b.iter_batched(
                    || {
                        let (tx, rx) = mpsc::channel();
                        for (index, frame) in frames.iter().enumerate() {
                            tx.send(WorkItem::Frame {
                                index,
                                name: frame.name.clone(),
                                frame: common::copy_frame(frame),
                            })
                            .unwrap();
                        }
                        rx
                    },
                    |rx| {
                        let done =
                            process_queue_streaming(rx, ai.clone(), task.clone(), workers, 1)
                                .into_iter()
                                .filter(|(_, _, r)| r.is_ok())
                                .count();
                        assert_eq!(done, frames.len());
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }
    group.finish();
}

fn batch(c: &mut Criterion) {
    let (g1, ceo2) = common::ceo2();
    let (g256, _) = common::synthetic(256, 172e-6, 0);
    let small: Vec<Frame> = (0..200)
        .map(|i| common::synthetic_frame(256, 256, i))
        .collect();
    let large: Vec<Frame> = (0..16).map(|_| common::copy_frame(&ceo2)).collect();
    let ai256 = Arc::new(Integrator::new(g256).with_method(IntegrationMethod::PixelSplit));
    let ai1m = Arc::new(Integrator::new(g1).with_method(IntegrationMethod::PixelSplit));
    let sets = [("256", ai256, small), ("1m", ai1m, large)];
    // Build the 1D and 2D tables before timing.
    for (_, ai, frames) in &sets {
        black_box(ai.integrate1d(&frames[0], 1000));
        black_box(ai.integrate2d(&frames[0], 500, 360));
    }

    queue(c, "batch/queue1d", &sets, Integration1D { n_bins: 1000 });
    queue(
        c,
        "batch/queue2d",
        &sets,
        Integration2D {
            n_radial: 500,
            n_azimuthal: 360,
        },
    );

    let paths = vec![String::from(common::CEO2_TIFF); 16];
    let ai = &sets[1].1;
    let mut group = c.benchmark_group("batch/files");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.throughput(Throughput::Elements(paths.len() as u64));
    for workers in WORKERS {
        group.bench_function(format!("1m/w{workers}"), |b| {
            b.iter(|| {
                let out = process_files(
                    &paths,
                    ai.clone(),
                    Integration1D { n_bins: 1000 },
                    workers,
                    1,
                );
                assert!(out.iter().all(|(_, r)| r.is_ok()));
                out
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::config();
    targets = batch
}
criterion_main!(benches);
