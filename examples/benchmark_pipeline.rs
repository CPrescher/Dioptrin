//! Example: Benchmark full pipeline (load + integrate) with worker/thread combinations
//!
//! This example measures real throughput when processing files with different
//! configurations of parallel workers and threads per worker.
//!
//! Each worker performs the full pipeline: load image → create integrator → integrate
//!
//! Run with: cargo run --release --example benchmark_pipeline [image.tif] [calibration.poni]

use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::poni::Poni;

const DEFAULT_IMAGE: &str = "test_data/CeO2_Pilatus1M.tif";
const DEFAULT_PONI: &str = "test_data/v1.poni";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = env::args().nth(1).unwrap_or_else(|| DEFAULT_IMAGE.to_string());
    let poni_path = env::args().nth(2).unwrap_or_else(|| DEFAULT_PONI.to_string());

    let max_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    let bench_duration = Duration::from_secs(3);
    let n_bins_1d = 1000;

    // Verify files exist
    let frame = Frame::from_tiff(image_path.clone())?;
    println!("Frame size: {} x {} pixels", frame.dim1, frame.dim2);
    println!("Max threads: {}", max_threads);
    println!("Benchmark duration: {} seconds per config", bench_duration.as_secs());
    println!();
    drop(frame);

    // Generate worker/thread combinations where workers * threads <= max_threads
    let mut configs: Vec<(usize, usize)> = Vec::new();
    for n_workers in 1..=max_threads {
        for threads_per_worker in 1..=max_threads {
            if n_workers * threads_per_worker <= max_threads {
                configs.push((n_workers, threads_per_worker));
            }
        }
    }

    // Sort by total threads used, then by workers
    configs.sort_by_key(|(w, t)| (w * t, *w));
    configs.dedup();

    println!("Testing {} configurations...\n", configs.len());

    let mut results: Vec<(usize, usize, f64, f64)> = Vec::new();

    for (n_workers, threads_per_worker) in &configs {
        let throughput = benchmark_config(
            *n_workers,
            *threads_per_worker,
            &image_path,
            &poni_path,
            n_bins_1d,
            bench_duration,
        )?;

        let total_threads = n_workers * threads_per_worker;
        results.push((*n_workers, *threads_per_worker, total_threads as f64, throughput));

        print!(".");
        std::io::stdout().flush().ok();
    }
    println!("\n");

    // Sort by throughput descending
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    // Print results
    println!("1D Integration Pipeline (load + integrate, {} bins)", n_bins_1d);
    println!(
        "{:>8} {:>12} {:>12} {:>14}",
        "Workers", "Threads/W", "Total Thr", "Throughput"
    );
    println!(
        "{:>8} {:>12} {:>12} {:>14}",
        "", "", "", "(files/sec)"
    );
    println!("{:-<8} {:-<12} {:-<12} {:-<14}", "", "", "", "");

    for (workers, threads, total, throughput) in &results {
        println!(
            "{:>8} {:>12} {:>12} {:>14.1}",
            workers, threads, *total as usize, throughput
        );
    }

    println!();
    let best = &results[0];
    println!(
        "Best config: {} workers × {} threads = {:.1} files/sec",
        best.0, best.1, best.3
    );

    // Show top 5
    println!("\nTop 5 configurations:");
    for (i, (workers, threads, _, throughput)) in results.iter().take(5).enumerate() {
        println!(
            "  {}. {} workers × {} threads → {:.1} files/sec",
            i + 1,
            workers,
            threads,
            throughput
        );
    }

    Ok(())
}

fn benchmark_config(
    n_workers: usize,
    threads_per_worker: usize,
    image_path: &str,
    poni_path: &str,
    n_bins: usize,
    duration: Duration,
) -> Result<f64, Box<dyn std::error::Error>> {
    let counter = Arc::new(AtomicUsize::new(0));
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));

    let mut handles = Vec::new();

    for _ in 0..n_workers {
        let counter = Arc::clone(&counter);
        let running = Arc::clone(&running);
        let image_path = image_path.to_string();
        let poni_path = poni_path.to_string();

        let handle = thread::spawn(move || {
            // Create thread pool for this worker
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads_per_worker)
                .build()
                .unwrap();

            while running.load(Ordering::Relaxed) {
                // Full pipeline: load + integrate
                let frame = match Frame::from_tiff(image_path.clone()) {
                    Ok(f) => f,
                    Err(_) => continue,
                };

                let poni = match Poni::load(&poni_path) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
                let mut integrator = Integrator::new(geometry)
                    .with_method(IntegrationMethod::PixelSplit)
                    .with_radial_unit(RadialUnit::TthDeg);

                pool.install(|| {
                    let _ = integrator.integrate1d(&frame, n_bins);
                });

                counter.fetch_add(1, Ordering::Relaxed);
            }
        });

        handles.push(handle);
    }

    // Let it run for the specified duration
    thread::sleep(duration);
    running.store(false, Ordering::Relaxed);

    // Wait for all workers to finish
    for handle in handles {
        let _ = handle.join();
    }

    let total_processed = counter.load(Ordering::Relaxed);
    let throughput = total_processed as f64 / duration.as_secs_f64();

    Ok(throughput)
}
