//! Example: Benchmark integration with varying thread counts
//!
//! This example measures 1D and 2D integration performance across
//! different thread counts from 1 to the system maximum.
//!
//! Run with: cargo run --release --example benchmark_threads [image.tif] [calibration.poni]

use std::env;
use std::io::Write;
use std::time::Instant;

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

    let n_iterations = 50;
    let n_bins_1d = 1000;
    let n_radial_2d = 100;
    let n_azimuthal_2d = 36;

    // Load frame once
    let frame = Frame::from_tiff(image_path)?;

    println!("Loaded frame: {} x {} pixels", frame.dim1, frame.dim2);
    println!("Max threads: {}", max_threads);
    println!("Iterations per measurement: {}", n_iterations);
    println!();

    // Collect results
    let mut results_1d: Vec<(usize, f64)> = Vec::new();
    let mut results_2d: Vec<(usize, f64)> = Vec::new();

    print!("Benchmarking");
    std::io::stdout().flush().ok();

    for n_threads in 1..=max_threads {
        // Create a new thread pool for this thread count
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()?;

        // Create fresh integrator (reload poni since Geometry takes ownership)
        let poni = Poni::load(&poni_path)?;
        let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::PixelSplit)
            .with_radial_unit(RadialUnit::TthDeg);

        // Warm-up
        pool.install(|| {
            let _ = integrator.integrate1d(&frame, n_bins_1d);
            let _ = integrator.integrate2d(&frame, n_radial_2d, n_azimuthal_2d);
        });

        // Benchmark 1D
        let start = Instant::now();
        pool.install(|| {
            for _ in 0..n_iterations {
                let _ = integrator.integrate1d(&frame, n_bins_1d);
            }
        });
        let elapsed_1d = start.elapsed();
        let per_iter_1d_us = elapsed_1d.as_secs_f64() * 1_000_000.0 / n_iterations as f64;
        results_1d.push((n_threads, per_iter_1d_us));

        // Benchmark 2D
        let start = Instant::now();
        pool.install(|| {
            for _ in 0..n_iterations {
                let _ = integrator.integrate2d(&frame, n_radial_2d, n_azimuthal_2d);
            }
        });
        let elapsed_2d = start.elapsed();
        let per_iter_2d_us = elapsed_2d.as_secs_f64() * 1_000_000.0 / n_iterations as f64;
        results_2d.push((n_threads, per_iter_2d_us));

        print!(".");
        std::io::stdout().flush().ok();
    }
    println!("\n");

    // Print 1D results
    println!("1D Integration ({} bins)", n_bins_1d);
    println!(
        "{:>8} {:>12} {:>12}",
        "Threads", "Time (µs)", "Speedup"
    );
    println!("{:-<8} {:-<12} {:-<12}", "", "", "");
    let baseline_1d = results_1d[0].1;
    for (threads, time_us) in &results_1d {
        let speedup = baseline_1d / time_us;
        println!("{:>8} {:>12.0} {:>12.2}x", threads, time_us, speedup);
    }

    println!();

    // Print 2D results
    println!("2D Integration ({}x{} bins)", n_radial_2d, n_azimuthal_2d);
    println!(
        "{:>8} {:>12} {:>12}",
        "Threads", "Time (µs)", "Speedup"
    );
    println!("{:-<8} {:-<12} {:-<12}", "", "", "");
    let baseline_2d = results_2d[0].1;
    for (threads, time_us) in &results_2d {
        let speedup = baseline_2d / time_us;
        println!("{:>8} {:>12.0} {:>12.2}x", threads, time_us, speedup);
    }

    Ok(())
}
