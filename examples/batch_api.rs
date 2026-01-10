//! Example: Batch processing with the new batch API
//!
//! This example demonstrates how to use the `dioptrin::batch` module
//! to process multiple files in parallel with different integration methods.
//!
//! Run with: cargo run --release --example batch_api [image.tif] [calibration.poni]

use std::env;
use std::sync::Arc;
use std::time::Instant;

use dioptrin::batch::{
    Integration1D, Integration1DWithErrors, Integration2D, process_files, process_files_sequential,
};
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::poni::Poni;

const DEFAULT_IMAGE: &str = "test_data/CeO2_Pilatus1M.tif";
const DEFAULT_PONI: &str = "test_data/v1.poni";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_IMAGE.to_string());
    let poni_path = env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_PONI.to_string());

    // Verify files exist and get dimensions
    let frame = Frame::from_tiff(image_path.clone())?;
    println!("Frame size: {} x {} pixels", frame.dim1, frame.dim2);
    println!("Image: {}", image_path);
    println!("PONI: {}", poni_path);
    println!();

    // Create shared integrator
    let poni = Poni::load(&poni_path)?;
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    let integrator = Arc::new(
        Integrator::new(geometry)
            .with_method(IntegrationMethod::PixelSplit)
            .with_radial_unit(RadialUnit::TthDeg),
    );

    // Simulate processing multiple files (we'll use the same file repeatedly)
    let num_files = 20;
    let files: Vec<String> = (0..num_files).map(|_| image_path.clone()).collect();

    println!("Processing {} files...\n", num_files);

    // Example 1: Sequential processing
    println!("=== Sequential Processing ===");
    let start = Instant::now();
    let task = Integration1D { n_bins: 3000 };
    let results = process_files_sequential(&files, &integrator, task);
    let elapsed = start.elapsed();

    let successful = results.iter().filter(|(_, r)| r.is_ok()).count();
    println!("Processed: {}/{}", successful, num_files);
    println!("Time: {:.2?}", elapsed);
    println!(
        "Throughput: {:.1} files/sec\n",
        num_files as f64 / elapsed.as_secs_f64()
    );

    // Example 2: Parallel processing with 1D integration
    println!("=== Parallel 1D Integration ===");
    let start = Instant::now();
    let task = Integration1D { n_bins: 3000 };
    let results = process_files(&files, integrator.clone(), task, 4, 2);
    let elapsed = start.elapsed();

    let successful = results.iter().filter(|(_, r)| r.is_ok()).count();
    println!("Workers: 4, Threads/worker: 2");
    println!("Processed: {}/{}", successful, num_files);
    println!("Time: {:.2?}", elapsed);
    println!(
        "Throughput: {:.1} files/sec\n",
        num_files as f64 / elapsed.as_secs_f64()
    );

    // Example 3: Parallel processing with errors
    println!("=== Parallel 1D Integration with Errors ===");
    let start = Instant::now();
    let task = Integration1DWithErrors { n_bins: 3000 };
    let results = process_files(&files, integrator.clone(), task, 4, 2);
    let elapsed = start.elapsed();

    let successful = results.iter().filter(|(_, r)| r.is_ok()).count();
    println!("Workers: 4, Threads/worker: 2");
    println!("Processed: {}/{}", successful, num_files);
    println!("Time: {:.2?}", elapsed);
    println!(
        "Throughput: {:.1} files/sec",
        num_files as f64 / elapsed.as_secs_f64()
    );

    if let Some((_, Ok((radial, intensity, errors)))) = results.first() {
        println!("First result: {} bins", radial.len());
        println!(
            "  Intensity range: [{:.1}, {:.1}]",
            intensity.iter().copied().fold(f64::INFINITY, f64::min),
            intensity.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
        println!(
            "  Error range: [{:.2e}, {:.2e}]",
            errors.iter().copied().fold(f64::INFINITY, f64::min),
            errors.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }
    println!();

    // Example 4: Parallel 2D integration
    println!("=== Parallel 2D Integration ===");
    let start = Instant::now();
    let task = Integration2D {
        n_radial: 1000,
        n_azimuthal: 360,
    };
    let results = process_files(&files, integrator.clone(), task, 4, 2);
    let elapsed = start.elapsed();

    let successful = results.iter().filter(|(_, r)| r.is_ok()).count();
    println!("Workers: 4, Threads/worker: 2");
    println!("Processed: {}/{}", successful, num_files);
    println!("Time: {:.2?}", elapsed);
    println!(
        "Throughput: {:.1} files/sec",
        num_files as f64 / elapsed.as_secs_f64()
    );

    if let Some((_, Ok(intensity))) = results.first() {
        println!("First result: {} bins (1000×360)", intensity.len());
    }
    println!();

    // Example 5: Different worker configurations
    println!("=== Testing Different Worker Configurations ===");
    let configs = vec![
        (1, 8), // 1 worker, 8 threads
        (2, 4), // 2 workers, 4 threads each
        (4, 2), // 4 workers, 2 threads each
        (8, 1), // 8 workers, 1 thread each
    ];

    let task = Integration1D { n_bins: 3000 };
    let mut config_results = Vec::new();

    for (workers, threads) in &configs {
        let start = Instant::now();
        let _results = process_files(&files, integrator.clone(), task, *workers, *threads);
        let elapsed = start.elapsed();
        let throughput = num_files as f64 / elapsed.as_secs_f64();
        config_results.push((*workers, *threads, throughput));
    }

    println!(
        "{:>8} {:>12} {:>12} {:>14}",
        "Workers", "Threads/W", "Total Thr", "Throughput"
    );
    println!("{:>8} {:>12} {:>12} {:>14}", "", "", "", "(files/sec)");
    println!("{:-<8} {:-<12} {:-<12} {:-<14}", "", "", "", "");

    for (workers, threads, throughput) in &config_results {
        println!(
            "{:>8} {:>12} {:>12} {:>14.1}",
            workers,
            threads,
            workers * threads,
            throughput
        );
    }

    // Find best configuration
    let best = config_results
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    println!(
        "\nBest config: {} workers × {} threads = {:.1} files/sec",
        best.0, best.1, best.2
    );

    Ok(())
}
