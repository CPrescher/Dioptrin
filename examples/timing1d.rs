//! Example: Timing measurements for 1D integration
//!
//! This example measures the performance of 1D integration
//! with a warm-up phase followed by multiple iterations.
//!
//! Run with: cargo run --release --example timing1d [image.tif] [calibration.poni]

use std::env;
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

    let n_iterations = 100;
    let n_bins = 1000;

    // Load the calibration parameters from a PONI file
    let poni = Poni::load(&poni_path)?;

    // Load the detector image from a TIFF file
    let frame = Frame::from_tiff(image_path)?;

    println!("Loaded frame: {} x {} pixels", frame.dim1, frame.dim2);
    println!("Integration bins: {}", n_bins);
    println!("Iterations: {}\n", n_iterations);

    // Create geometry from frame dimensions and calibration
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    // Create an integrator
    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_unit(RadialUnit::TthDeg);

    // Warm-up: single integration to initialize caches and JIT
    println!("Warming up...");
    let _ = integrator.integrate1d(&frame, n_bins);

    // Timed iterations
    println!("Running {} iterations...\n", n_iterations);

    let start = Instant::now();
    for _ in 0..n_iterations {
        let _ = integrator.integrate1d(&frame, n_bins);
    }
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_iteration_ms = total_ms / n_iterations as f64;
    let per_iteration_us = per_iteration_ms * 1000.0;

    println!("Results:");
    println!("  Total time:     {:.2} ms", total_ms);
    println!("  Per iteration:  {:.2} ms ({:.0} µs)", per_iteration_ms, per_iteration_us);
    println!("  Throughput:     {:.1} integrations/sec", 1000.0 / per_iteration_ms);

    Ok(())
}
